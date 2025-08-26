# interpret_neurons.py
# Pipeline:
# 0) Stratified subsample & cache z(x)
# 1) Feature on/off via per-neuron high-quantile thresholds
# 2) Base-rate-corrected association via PMI with Laplace smoothing
# 3) Logit-lens on decoder directions (W_out @ d_i)
# 4) Causal edit (knockout; optional steer)
# 5) Export sprites + JSONL for LLM consumption

import os, io, math, json, random
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, utils as vutils
from tqdm import tqdm


# ----------------------- Dataset -----------------------

class TuBerlinParquetDataset(Dataset):
    """
    Loads all parquet shards under data_dir and keeps records in memory as (bytes, label).
    Images are PNG bytes (grayscale) at 224x224 already; we enforce Resize(224) anyway.
    Exposes .all_records (bytes) and .all_labels (np.int64), and .indices (selected split).
    """
    def __init__(self, data_dir: str, split: str, transform=None, val_ratio: float = 0.15, seed: int = 1234):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.transform = transform
        parquet_dir = self.data_dir / "data"
        files = sorted([str(parquet_dir / f) for f in parquet_dir.iterdir() if f.suffix == ".parquet"])
        if not files:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
        records, labels = [], []
        for fp in files:
            table = pq.read_table(fp, columns=["image", "label"])
            df = table.to_pandas()
            for img_dict, lab in zip(df["image"].tolist(), df["label"].tolist()):
                b = img_dict["bytes"]  # PNG bytes
                records.append(b)
                labels.append(int(lab))
        self.all_records = records
        self.all_labels = np.array(labels, dtype=np.int64)

        # Stratified split (per-class deterministic slice)
        self.indices_train, self.indices_val = self._stratified_split(self.all_labels, val_ratio, seed)
        self.indices = self.indices_train if split == "train" else self.indices_val

    @staticmethod
    def _stratified_split(labels: np.ndarray, val_ratio: float, seed: int):
        rng = np.random.RandomState(seed)
        idx = np.arange(len(labels))
        train_idx, val_idx = [], []
        for c in np.unique(labels):
            c_idx = idx[labels == c]
            rng.shuffle(c_idx)
            n_val = max(1, int(round(val_ratio * len(c_idx))))
            val_idx.extend(c_idx[:n_val])
            train_idx.extend(c_idx[n_val:])
        rng.shuffle(train_idx); rng.shuffle(val_idx)
        return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        gi = int(self.indices[i])
        b = self.all_records[gi]
        y = int(self.all_labels[gi])
        img = Image.open(io.BytesIO(b)).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, y, gi

    def get_pil_by_global_index(self, gi: int) -> Image.Image:
        b = self.all_records[gi]
        return Image.open(io.BytesIO(b)).convert("L")


# ----------------------- Transforms -----------------------

def build_transforms(mean: Tuple[float] = (0.90,), std: Tuple[float] = (0.25,)):
    tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),                          # -> (1,H,W)
        transforms.Normalize(mean, std),
    ])
    return tf

# ----------------------- Backbone (frozen classifier) -----------------------

class CamFriendlyResNet(nn.Module):
    """
    ResNet-18 with 1-channel input and grayscale-initialized conv1.
    GAP + Linear head for clean Grad-CAM.
    Produces 7x7 feature maps at 224 input (your trained setup).
    """
    def __init__(self, num_classes: int, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=weights)
        if in_channels == 1:
            w = m.conv1.weight.data  # (64,3,7,7)
            gray = w.mean(dim=1, keepdim=True)  # (64,1,7,7)
            m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                m.conv1.weight.copy_(gray)
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feat: bool = False):
        feat = self.stem(x)                  # (B, 512, 7, 7)
        pooled = self.gap(feat).flatten(1)   # (B, 512)
        logits = self.fc(pooled)             # (B, K)
        if return_feat:
            return logits, feat, pooled
        return logits


# ----------------------- SAE (Top-K trained; we use dense ReLU for analysis) -----------------------

class TopKSAE(nn.Module):
    """
    Token-space SAE used for training. For analysis we use:
        s = ReLU(W_e a + b_e)
    without applying top-k masking, to get analog activation magnitudes.
    D contains unit-norm columns learned on standardized tokens.
    """
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.W_e = nn.Parameter(torch.empty(latent_dim, in_dim))
        self.b_e = nn.Parameter(torch.empty(latent_dim))
        self.D   = nn.Parameter(torch.empty(in_dim, latent_dim))  # decoder atoms (C, M)
        self.b_d = nn.Parameter(torch.empty(in_dim))

    def encode_dense(self, a):  # (N,C) -> z (N,M) dense ReLU
        s = torch.addmm(self.b_e, a, self.W_e.t())
        return torch.relu(s)


# ----------------------- Utilities -----------------------

def load_checkpoints(
    classifier_ckpt: str,
    sae_ckpt: str,
    device: torch.device
):
    # Load classifier
    cstate = torch.load(classifier_ckpt, map_location=device)
    num_classes = int(cstate.get("num_classes", 250))
    mean_img = tuple(cstate.get("mean", (0.90,)))
    std_img  = tuple(cstate.get("std", (0.25,)))
    cfg = cstate.get("config", {}) or {}
    val_ratio = float(cfg.get("val_ratio", 0.15))
    seed = int(cfg.get("seed", 1234))

    backbone = CamFriendlyResNet(num_classes=num_classes, in_channels=1, pretrained=False).to(device)
    backbone.load_state_dict(cstate["model"])
    backbone.eval()

    # Classifier head params
    W_out = backbone.fc.weight.detach().clone()   # (K, C)
    b_out = backbone.fc.bias.detach().clone()     # (K,)

    # Load SAE
    sstate = torch.load(sae_ckpt, map_location=device)
    sae_params = sstate["sae_state"]
    # Handle DDP prefixes if any
    remap = lambda k: k.replace("module.", "") if k.startswith("module.") else k
    sae_params = {remap(k): v for k, v in sae_params.items()}
    feature_mean: torch.Tensor = sstate.get("feature_mean", None)
    feature_std:  torch.Tensor = sstate.get("feature_std", None)

    # Infer dims
    in_dim = feature_mean.numel() if feature_mean is not None else W_out.shape[1]
    latent_dim = sae_params["W_e"].shape[0]

    sae = TopKSAE(in_dim=in_dim, latent_dim=latent_dim).to(device)
    with torch.no_grad():
        sae.W_e.copy_(sae_params["W_e"])
        sae.b_e.copy_(sae_params["b_e"])
        sae.D.copy_(sae_params["D"])
        sae.b_d.copy_(sae_params["b_d"])
    sae.eval()

    return backbone, (W_out, b_out), (mean_img, std_img), (feature_mean.to(device), feature_std.to(device)), (val_ratio, seed), sae, num_classes


def build_class_names(class_map_path: Optional[str], num_classes: int) -> List[str]:
    if class_map_path and Path(class_map_path).exists():
        with open(class_map_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) == num_classes:
            return names
    return [f"class_{i:03d}" for i in range(num_classes)]


def to_pil_thumbnail(pil_img: Image.Image, max_side: int = 128) -> Image.Image:
    img = pil_img.copy()
    # pad to square, then resize keeping strokes centered
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), color=255)
    canvas.paste(img, ((side - w)//2, (side - h)//2))
    canvas = canvas.resize((max_side, max_side), resample=Image.BICUBIC)
    return canvas


# ----------------------- Main pipeline -----------------------

def main():
    # ---- Config ----
    classifier_ckpt = "runs/standard_cnn/best.pt"
    sae_ckpt        = "runs/sae_topk/best_sae.pt"
    data_dir        = r"F:\TuBerlin"          # parquet layout
    out_dir         = Path("./neuron_report")
    out_dir.mkdir(parents=True, exist_ok=True)
    sprites_dir = out_dir / "sprites"; sprites_dir.mkdir(exist_ok=True)
    thumbs_dir  = out_dir / "thumbs";  thumbs_dir.mkdir(exist_ok=True)

    # subsample per class
    n_per_class = 60          # 0) n in [30, 100]
    loader_bsz  = 128
    num_workers = 8
    seed        = 1234

    # thresholds
    q = 0.99                  # q in [0.98, 0.995]

    # delta-minus evaluation
    top_classes_for_delta = 5     # per feature
    control_classes_rand  = 8
    m_images_per_class    = 8
    steer_alpha = 0.0            # set >0.0 to also compute Δ^+ (cheap check)

    # top examples / sprites
    k_top_examples = 5
    thumb_side = 128

    # optional class names file (one name per line)
    class_names_file = None  # e.g., "classes.txt"

    # ---- Setup ----
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, (W_out, b_out), (mean_img, std_img), (fmean, fstd), (val_ratio, seed_ckpt), sae, num_classes = \
        load_checkpoints(classifier_ckpt, sae_ckpt, device)
    class_names = build_class_names(class_names_file, num_classes)
    assert fmean is not None and fstd is not None, \
        "Missing feature_mean/feature_std in SAE checkpoint; save them at training or precompute."

    # data & stratified subsample indices
    tf = build_transforms(mean_img, std_img)
    full_train = TuBerlinParquetDataset(data_dir=data_dir, split="train", transform=tf,
                                        val_ratio=val_ratio, seed=seed_ckpt)
    # stratified pick n_per_class
    rng = np.random.RandomState(seed)
    by_class: Dict[int, List[int]] = {}
    for i in range(len(full_train.indices)):
        gi = int(full_train.indices[i])
        lab = int(full_train.all_labels[gi])
        by_class.setdefault(lab, []).append(i)
    picked_local_indices: List[int] = []
    for c, idxs in by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        picked_local_indices.extend(idxs[:n_per_class])
    picked_local_indices = np.array(sorted(picked_local_indices), dtype=np.int64)
    N = len(picked_local_indices)
    print(f"[Subsample] N={N} images ({n_per_class} per class × {len(by_class)} classes)")

    # loader over subsample
    sub_sampler = torch.utils.data.SubsetRandomSampler(picked_local_indices.tolist())
    loader = DataLoader(
        full_train, batch_size=loader_bsz, sampler=sub_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=False, persistent_workers=(num_workers > 0)
    )

    # ---- 0) Cache z(x) (per-image) using dense ReLU & {MAX, AVG} over spatial tokens ----
    # Shapes: per-image z_max, z_avg: (N, M)
    M = sae.latent_dim
    C = W_out.shape[1]  # 512
    H = W = 7
    z_max_per_image = torch.zeros((N, M), dtype=torch.float32, device="cpu")
    z_avg_per_image = torch.zeros((N, M), dtype=torch.float32, device="cpu")
    labels_np = np.zeros(N, dtype=np.int64)
    global_indices = np.zeros(N, dtype=np.int64)

    with torch.no_grad():
        offset = 0
        for x, y, gi in tqdm(loader, desc="Caching z(x) via {MAX, AVG} over spatial", ncols=100):
            b = x.size(0)
            x = x.to(device, non_blocking=True)
            with autocast():
                _, feat, pooled = backbone(x, return_feat=True)     # feat: (B, C, 7, 7)
            B, Cc, Hh, Ww = feat.shape
            assert (Cc, Hh, Ww) == (C, H, W)
            # tokens (B*H*W, C)
            a = feat.permute(0,2,3,1).reshape(-1, C)               # unstandardized tokens
            a_std = (a - fmean) / fstd
            # dense ReLU code (T, M)
            z_dense = sae.encode_dense(a_std)                      # (B*H*W, M)
            # max over spatial tokens per image
            z_dense = z_dense.reshape(B, H*W, M)                   # (B, 49, M)
            z_max   = z_dense.max(dim=1).values.to("cpu")          # (B, M)
            z_avg   = z_dense.mean(dim=1).to("cpu")               # (B, M)
            z_max_per_image[offset:offset+b] = z_max
            z_avg_per_image[offset:offset+b] = z_avg
            labels_np[offset:offset+b] = y.numpy()
            global_indices[offset:offset+b] = gi.numpy()
            offset += b

    # ---- 1) Feature on/off thresholds & stats ----
    # tau_i = quantile over N images; indicator I_i(x) = 1[z_i(x) > tau_i]
    q_tensor = torch.tensor(q, dtype=torch.float32)
    tau = torch.quantile(z_max_per_image, q=q_tensor.item(), dim=0).numpy()         # (M,)
    z_np = z_max_per_image.numpy()                                                  # (N, M)
    I = (z_np > tau.reshape(1, -1))                                                 # (N, M) bool
    n_i = I.sum(axis=0)                                                             # (M,)
    p_hat_i = n_i / float(N)                                                        # sparsity per feature
    # mean-on μ_i (conditioned on I_i==1); handle empty by 0
    mu_i = np.zeros(M, dtype=np.float32)
    for i in range(M):
        on_idx = np.where(I[:, i])[0]
        mu_i[i] = float(z_np[on_idx, i].mean()) if on_idx.size > 0 else 0.0

    # ---- 2) PMI with Laplace smoothing (per feature × class) ----
    classes = np.sort(np.unique(labels_np))
    K = classes.size
    lam = 1.0
    # class counts
    n_c = np.array([np.sum(labels_np == c) for c in classes], dtype=np.int64)       # (K,)
    P_c = (n_c + lam) / (N + lam * K)

    # n_{i,c}: for each feature i and class c: number of images in class c where I_i=1
    # Build I indices per class to vectorize
    PMI = np.zeros((M, K), dtype=np.float32)
    n_i_c = np.zeros((M, K), dtype=np.int64)
    for k_idx, c in enumerate(classes):
        mask_c = (labels_np == c)                                                  # (N,)
        n_c_k = int(mask_c.sum())
        if n_c_k == 0:
            continue
        I_c = I[mask_c, :]                                                         # (n_c, M)
        n_i_c[:, k_idx] = I_c.sum(axis=0)                                          # (M,)
    P_I1 = (n_i + lam) / (N + lam)
    # P(c | I=1): (n_i_c + λ) / (n_i + λ|C|)
    P_c_given_I1 = (n_i_c + lam) / (n_i.reshape(-1,1) + lam * K)                   # (M,K)
    # PMI_i(c) = log ( P(c | I=1) / P(c) )
    # guard tiny
    PMI = np.log(np.maximum(P_c_given_I1, 1e-12)) - np.log(P_c.reshape(1, -1))

    # ---- 3) Logit-lens on decoder direction: w_i = W_out @ d_i_unstd ----
    # D was trained on standardized tokens. Convert decoder columns to unstandardized feature space:
    with torch.no_grad():
        D_std = sae.D.detach().to(device)                 # (C, M)
        D_unstd = (D_std * fstd.view(-1, 1)).to(device)   # (C, M)
        W_matrix = W_out.to(device)                       # (K, C)
        w_all = (W_matrix @ D_unstd).detach().cpu().numpy()      # (K, M) -> transpose later if needed
        w_all = w_all.T                                   # (M, K)

    # ---- 4) Closed-form Causal edit Δ^- using cached z_avg (no forwards) ----
    def closed_form_delta_minus_for_feature(
        feature_i: int,
        top_class_indices: List[int],      # k-indices (0..K-1)
        control_class_indices: List[int],  # k-indices (0..K-1)
        m_images_per_class: int = 8,
        seed_local: int = 1234
    ) -> Dict[int, float]:
        """
        Uses Δy = - mean(z_avg_i) * w_i (exact for linear head).
        Returns {abs_label: mean Δy_c} for evaluated classes.
        Only samples images where the feature is 'on' to cut noise.
        """
        rng = np.random.RandomState(seed_local + feature_i)
        result: Dict[int, float] = {}
        eval_k = top_class_indices + control_class_indices

        for kk in eval_k:
            c_abs = int(classes[kk])
            # indices (in subsample) for this class where feature is ON
            idxs = np.where((labels_np == c_abs) & I[:, feature_i])[0]
            if idxs.size == 0:
                continue
            rng.shuffle(idxs)
            idxs = idxs[:m_images_per_class]

            # mean z_avg over selected images
            idxs_t = torch.from_numpy(idxs).to(torch.long)
            zbar = float(z_avg_per_image.index_select(0, idxs_t)[:, feature_i].mean().item())
            # class-logit delta: - zbar * w_i[c]
            delta = - zbar * float(w_all[feature_i, kk])
            result[c_abs] = delta

        return result

    # ---- 5) Build sprites + JSONL ----
    jsonl_path = out_dir / "neurons.jsonl"
    fout = open(jsonl_path, "w", encoding="utf-8")

    # Precompute per-image tensors for thumbnails only when needed
    def save_sprite_and_thumbs_for_feature(fid: int, top_examples: List[Tuple[int, float]]):
        """
        top_examples: list of (j_idx_in_subsample, z_value)
        Saves individual thumbs and a horizontal sprite; returns paths + metadata.
        """
        thumbs_info = []
        tiles = []
        for k, (j_idx, zval) in enumerate(top_examples):
            gi = int(global_indices[j_idx])
            lab = int(labels_np[j_idx])
            pil = full_train.get_pil_by_global_index(gi)
            thumb = to_pil_thumbnail(pil, max_side=thumb_side)
            thumb_name = f"feature_{fid:05d}_img{k}.webp"
            thumb_path = thumbs_dir / thumb_name
            thumb.save(thumb_path, "WEBP", quality=90)
            tiles.append(thumb.convert("L"))
            thumbs_info.append({
                "thumb": str(thumb_path.as_posix()),
                "label": class_names[lab],
                "z": float(zval),
            })
        # compose sprite (horizontal)
        sprite = Image.new("L", (thumb_side * len(tiles), thumb_side), color=255)
        for i, tile in enumerate(tiles):
            sprite.paste(tile, (i * thumb_side, 0))
        sprite_path = sprites_dir / f"feature_{fid:05d}.webp"
        sprite.save(sprite_path, "WEBP", quality=90)
        return str(sprite_path.as_posix()), thumbs_info

    # precompute some helpers
    # class index mapping: absolute label → 0..K-1 (position in arrays)
    label_to_k = {int(c): int(i) for i, c in enumerate(classes)}
    k_to_label = {int(i): int(c) for i, c in enumerate(classes)}

    # z-score helper
    def zscore(v: np.ndarray, axis=None, eps=1e-8):
        m = v.mean(axis=axis, keepdims=True)
        s = v.std(axis=axis, keepdims=True)
        return (v - m) / (s + eps)

    # For each feature/neuron i:
    for i in tqdm(range(M), desc="Interpreting features", ncols=100):
        # stats from step 1
        sparsity = float(p_hat_i[i])
        threshold = float(tau[i])
        mean_on  = float(mu_i[i])

        # Per-class PMI and w
        pmi_i = PMI[i, :]               # (K,)
        w_i   = w_all[i, :]             # (K,)

        # Δ^- (knockout) on tiny batches for top classes by a quick proxy:
        # pick union of top by PMI and top by w (positives) to evaluate causally
        top_by_pmi_idx = np.argsort(-pmi_i)[:top_classes_for_delta]
        top_by_w_idx   = np.argsort(-w_i)[:top_classes_for_delta]
        cand_k_idx = np.unique(np.concatenate([top_by_pmi_idx, top_by_w_idx], axis=0)).tolist()

        # add random control classes
        all_k_idx = list(range(K))
        random.shuffle(all_k_idx)
        control_k_idx = []
        for kk in all_k_idx:
            if kk not in cand_k_idx:
                control_k_idx.append(kk)
            if len(control_k_idx) >= control_classes_rand:
                break

        # Compute closed-form Δ^- (no forward pass)
        delta_minus_map = closed_form_delta_minus_for_feature(
            i, cand_k_idx, control_k_idx, m_images_per_class=m_images_per_class, seed_local=seed
        )
        # expand to vector over evaluated classes only
        delta_vec = np.zeros(K, dtype=np.float32)
        have_delta = np.zeros(K, dtype=bool)
        for c_abs, val in delta_minus_map.items():
            kk = label_to_k[c_abs]
            delta_vec[kk] = float(val)
            have_delta[kk] = True

        # z-scores across classes (defined over available entries)
        zPMI = zscore(pmi_i.copy())
        zW   = zscore(w_i.copy())
        # For zΔ^-, compute z-score only over entries we computed; others keep 0
        if have_delta.any():
            sub = delta_vec[have_delta]
            zsub = zscore(sub)
            zDelta = np.zeros_like(delta_vec)
            zDelta[have_delta] = zsub
        else:
            zDelta = np.zeros_like(delta_vec)

        # Combined class ranking score S_i(c)
        S = zPMI + 2.0 * zDelta + 1.0 * zW

        # Step 5.1: top-k examples for this feature by z_i(x)
        z_col = z_np[:, i]                    # (N,)
        top_j = np.argsort(-z_col)[:k_top_examples]
        top_examples = [(int(j_idx), float(z_col[int(j_idx)])) for j_idx in top_j]
        sprite_path, thumbs_info = save_sprite_and_thumbs_for_feature(i, top_examples)

        # Per-class counts n_{i,c} for classes shown
        i_counts = n_i_c[i, :]  # (K,)

        # Build top classes table (keep top 5 by S)
        topK = 5
        top_classes_idx = np.argsort(-S)[:topK]
        top_classes = []
        for kk in top_classes_idx:
            top_classes.append({
                "class": class_names[int(k_to_label[kk])],
                "S": float(S[kk]),
                "PMI": float(pmi_i[kk]),
                "delta_minus": float(delta_vec[kk]) if have_delta[kk] else 0.0,
                "w": float(w_i[kk]),
                "count_on": int(i_counts[kk]),
            })

        # Autosummary (deterministic, short)
        # Choose the best class label by S
        if len(top_classes) > 0:
            best = top_classes[0]
            best_label = best["class"]
            dm = best["delta_minus"]
            sign_word = "reduces" if dm < 0 else "increases"
            summary = (f"Neuron {i:05d} fires on {sparsity*100:.1f}% of images; "
                       f"knockout {sign_word} '{best_label}' logit by {dm:.2f}. "
                       f"Combined score ranks: " +
                       ", ".join([tc['class'] for tc in top_classes]))
        else:
            summary = (f"Neuron {i:05d} fires on {sparsity*100:.1f}% of images; "
                       f"insufficient evidence for class association.")

        # JSON line
        obj = {
            "neuron_id": int(i),
            "sparsity": float(sparsity),
            "threshold": float(threshold),
            "mean_on_activation": float(mean_on),
            "top_classes": top_classes,
            "sprite_path": sprite_path,
            "top_examples": thumbs_info,
            "auto_summary": summary
        }
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    fout.close()
    print(f"\nDone. Wrote JSONL to: {jsonl_path}")
    print(f"Sprites: {sprites_dir}")
    print(f"Thumbs : {thumbs_dir}")


if __name__ == "__main__":
    main()
