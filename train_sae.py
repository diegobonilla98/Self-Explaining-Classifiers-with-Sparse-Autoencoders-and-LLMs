# sae_train.py
import os, math, time, io, random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image, ImageFilter
import pyarrow.parquet as pq
from tqdm import tqdm

# =========================
# Dataset (same layout you used)
# =========================

class TuBerlinParquetDataset(Dataset):
    """
    Loads all parquet shards under data_dir and keeps records in memory as (bytes, label).
    Images are PNG bytes (grayscale) at 224x224 already; we enforce Resize(224) anyway.
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
                b = img_dict["bytes"]
                records.append(b)
                labels.append(int(lab))
        self.all_records = records
        self.all_labels = np.array(labels, dtype=np.int64)
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

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        gi = int(self.indices[i])
        b = self.all_records[gi]
        y = int(self.all_labels[gi])
        img = Image.open(io.BytesIO(b)).convert("L")
        if self.transform: img = self.transform(img)
        return img, y

def build_transforms(mean: Tuple[float] = (0.90,), std: Tuple[float] = (0.25,)):
    val_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return val_tf

# =========================
# Frozen backbone (your ResNet-18, last conv is 7x7)
# =========================

class CamFriendlyResNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=weights)
        if in_channels == 1:
            w = m.conv1.weight.data
            gray = w.mean(dim=1, keepdim=True)
            m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad(): m.conv1.weight.copy_(gray)
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        self.gap  = nn.AdaptiveAvgPool2d((1,1))
        self.fc   = nn.Linear(512, num_classes)

    def forward(self, x, return_feat: bool = False):
        feat = self.stem(x)                # (B, 512, 7, 7) for your current model
        pooled = self.gap(feat).flatten(1)
        logits = self.fc(pooled)
        return (logits, feat) if return_feat else logits

# =========================
# Top-K SAE
# =========================

class TopKSAE(nn.Module):
    """
    Token-space SAE: a ∈ R^C  ->  z = ReLU(W_e a + b_e), keep top-k per sample,
    recon = D z + b_d, where D ∈ R^{C×M}. Decoder columns are unit-norm.
    """
    def __init__(self, in_dim: int, latent_dim: int, k: int, init_scale: float = 0.02):
        super().__init__()
        self.in_dim  = in_dim
        self.latent_dim = latent_dim
        self.k = k

        self.W_e = nn.Parameter(torch.randn(latent_dim, in_dim) * init_scale)
        self.b_e = nn.Parameter(torch.zeros(latent_dim))
        # Dictionary D with unit-norm columns; we store as (C, M)
        D = torch.randn(in_dim, latent_dim) * init_scale
        D = D / (D.norm(dim=0, keepdim=True) + 1e-8)
        self.D = nn.Parameter(D)
        self.b_d = nn.Parameter(torch.zeros(in_dim))

        # Usage tracking for dead-latent resampling (EMA of nonzero activations)
        self.register_buffer("usage_ema", torch.zeros(latent_dim))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def renorm_decoder(self):
        self.D.data /= (self.D.data.norm(dim=0, keepdim=True) + 1e-8)

    def encode_pre(self, a):  # (N,C) -> pre-activation s (N,M)
        return torch.addmm(self.b_e, a, self.W_e.t())  # (N,M)

    def topk_mask(self, s):   # s (N,M) -> values/indices/masked activations
        s = torch.relu(s)
        k = min(self.k, s.size(1))
        vals, idx = torch.topk(s, k=k, dim=1)
        # Build sparse-like z without materializing full dense z
        return vals, idx

    def decode_from_topk(self, vals, idx):
        """
        vals, idx: (N, k)
        D: (C, M) -> D^T: (M, C)
        recon = sum_j vals[n,j] * D[:, idx[n,j]]
        """
        # Gather atoms: (N, k, C)
        atoms = self.D.t()[idx]            # (N,k,C)
        recon = (vals.unsqueeze(-1) * atoms).sum(dim=1) + self.b_d  # (N,C)
        return recon

    def forward(self, a):
        # a: (N, C)
        s = self.encode_pre(a)
        vals, idx = self.topk_mask(s)
        recon = self.decode_from_topk(vals, idx)
        return recon, vals, idx

    @torch.no_grad()
    def update_usage_ema(self, idx, vals, beta=0.99):
        # idx/vals: (N,k). Count a latent as "used" if val>0 (they are >=0 here).
        used_counts = torch.zeros_like(self.usage_ema)
        # Each latent gets increments equal to the number of times it appears (could weight by vals.mean())
        used_counts.scatter_add_(0, idx.flatten(), torch.ones_like(vals.flatten(), dtype=used_counts.dtype))
        used_counts = used_counts / max(1, idx.size(0))  # normalize by batch tokens
        self.usage_ema.mul_(beta).add_((1 - beta) * used_counts)
        self.step += 1

    @torch.no_grad()
    def resample_dead(self, token_batch, dead_threshold=1e-4, min_steps=200, max_per_step=8):
        """
        token_batch: (N, C), standardized tokens from the current step.
        Find latents with very low usage_ema and reinitialize from random tokens.
        """
        if self.step.item() < min_steps: return 0
        dead = (self.usage_ema < dead_threshold).nonzero(as_tuple=False).flatten()
        if dead.numel() == 0: return 0
        n_re = min(max_per_step, dead.numel())
        sel = dead[:n_re]

        # pick random tokens and set decoder atoms to normalized tokens
        ridx = torch.randint(0, token_batch.size(0), (n_re,), device=token_batch.device)
        new_atoms = token_batch[ridx]               # (n_re, C)
        new_atoms = new_atoms / (new_atoms.norm(dim=1, keepdim=True) + 1e-8)
        self.D.data[:, sel] = new_atoms.t()        # columns set

        # align encoder rows roughly to decoder atoms (plus tiny noise)
        self.W_e.data[sel] = new_atoms + 0.01 * torch.randn_like(new_atoms)
        self.b_e.data[sel] = 0.0
        self.usage_ema[sel] = 0.05                 # bump usage so they don't immediately resample
        return int(n_re)

# =========================
# DDP boilerplate
# =========================

def ddp_setup():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 1

# Broadcast helper to keep all DDP replicas in sync after resampling
def broadcast_module_(m: nn.Module, src: int = 0):
    if not dist.is_available() or not dist.is_initialized():
        return
    for p in m.parameters(recurse=True):
        dist.broadcast(p.data, src=src)
    for _, buf in m.named_buffers(recurse=True):
        dist.broadcast(buf.data, src=src)

# =========================
# Feature stats (whitening)
# =========================

@torch.no_grad()
def compute_feature_stats(backbone, loader, device, is_ddp: bool = False, world_size: int = 1):
    """
    Computes per-channel mean/std over last conv features A (C,H,W) using
    precise running sums. DDP-safe by all-reducing Σx, Σx^2, and N.
    Returns tensors of shape (C,)
    """
    backbone.eval()
    sum_x = None
    sum_x2 = None
    N = 0
    for x, _ in tqdm(loader, desc="Estimating feature mean/std", leave=False):
        x = x.to(device, non_blocking=True)
        # no autocast for stats: keep full precision
        _, feat = backbone(x, return_feat=True)  # (B, C, H, W)
        B, C, H, W = feat.shape
        feats = feat.permute(0, 2, 3, 1).reshape(-1, C).to(torch.float64)
        if sum_x is None:
            sum_x = feats.sum(dim=0)
            sum_x2 = (feats * feats).sum(dim=0)
        else:
            sum_x += feats.sum(dim=0)
            sum_x2 += (feats * feats).sum(dim=0)
        N += feats.size(0)

    if is_ddp and dist.is_initialized():
        C = sum_x.numel()
        buf = torch.empty(2 * C + 1, dtype=torch.float64, device=device)
        buf[:C] = sum_x
        buf[C:2*C] = sum_x2
        buf[-1] = float(N)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        sum_x = buf[:C]
        sum_x2 = buf[C:2*C]
        N = int(buf[-1].item())

    mean64 = sum_x / max(1, N)
    var64 = (sum_x2 / max(1, N)) - (mean64 ** 2)
    var64 = torch.clamp(var64, min=1e-6)
    mean = mean64.to(torch.float32)
    std = torch.sqrt(var64).to(torch.float32)
    return mean.detach(), std.detach()

# =========================
# Training loop
# =========================

def main():
    # -------- Config --------
    checkpoint_path = "runs/standard_cnn/best.pt"   # your trained classifier
    data_dir = r"F:\TuBerlin"
    save_dir = Path("./runs/sae_topk")
    save_dir.mkdir(parents=True, exist_ok=True)

    # SAE hyperparams
    latent_dim  = 2048
    target_density = 0.01          # 0.5–2% → choose 1% by default
    top_k = max(1, int(round(latent_dim * target_density)))
    epochs = 40
    lr = 1e-3
    weight_decay = 1e-4
    per_gpu_imgs = 128             # images per batch (each gives H*W tokens)
    token_subsample = 1.0          # 1.0 = use all H*W tokens, else e.g. 0.5
    num_workers = 8

    # Resampling & decoder renorm
    dead_threshold = 1e-4
    resample_after_steps = 200
    max_resamples_per_step = 8
    renorm_every = 1               # after every optimizer step

    # DDP
    is_ddp, rank, world = ddp_setup()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)
    torch.backends.cudnn.benchmark = True

    # -------- Load classifier checkpoint --------
    state = torch.load(checkpoint_path, map_location=device)
    num_classes = int(state.get("num_classes", 250))
    mean_img = tuple(state.get("mean", (0.90,)))
    std_img  = tuple(state.get("std", (0.25,)))
    cfg = state.get("config", {}) or {}
    val_ratio = float(cfg.get("val_ratio", 0.15))
    seed = int(cfg.get("seed", 1234))

    # backbone
    backbone = CamFriendlyResNet(num_classes=num_classes, in_channels=1, pretrained=False).to(device)
    backbone.load_state_dict(state["model"])
    backbone.eval()  # frozen

    # -------- Data (no aug; consistent stats) --------
    tf = build_transforms(mean_img, std_img)
    train_set = TuBerlinParquetDataset(data_dir=data_dir, split="train", transform=tf,
                                       val_ratio=val_ratio, seed=seed)
    if is_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, drop_last=True)
    else:
        sampler = None
    loader = DataLoader(
        train_set, batch_size=per_gpu_imgs, shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0), drop_last=True
    )

    # -------- Feature stats (whitening in feature space) --------
    # Dedicated deterministic stats loader; DDP-safe reduction of sums
    with torch.no_grad():
        if is_ddp:
            stats_sampler = torch.utils.data.distributed.DistributedSampler(
                train_set, shuffle=False, drop_last=False
            )
        else:
            stats_sampler = None
        feat_stats_loader = DataLoader(
            train_set, batch_size=per_gpu_imgs, shuffle=False, sampler=stats_sampler,
            num_workers=num_workers, pin_memory=True, drop_last=False
        )
        fmean, fstd = compute_feature_stats(backbone, feat_stats_loader, device, is_ddp=is_ddp, world_size=world)

    if is_main:
        print(f"[SAE] Feature stats: mean shape={tuple(fmean.shape)}, std shape={tuple(fstd.shape)}")

    C = fmean.numel()
    # -------- SAE --------
    sae = TopKSAE(in_dim=C, latent_dim=latent_dim, k=top_k).to(device)

    if is_ddp:
        sae = nn.parallel.DistributedDataParallel(sae, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    base_sae = sae.module if is_ddp else sae
    optimizer = torch.optim.AdamW(
        [
            {"params": [base_sae.W_e, base_sae.b_e], "weight_decay": weight_decay},
            {"params": [base_sae.D, base_sae.b_d], "weight_decay": 0.0},
        ],
        lr=lr, betas=(0.9, 0.999), eps=1e-8
    )
    scaler = GradScaler()

    # -------- Train --------
    global_step = 0
    best_loss = float("inf")

    def tokens_from_images(x):
        # x: (B,1,224,224) -> A: (B, C, H, W) (C=512, H=W=7)
        with torch.no_grad(), autocast():
            _, feat = backbone(x, return_feat=True)
        B, Cc, H, W = feat.shape
        a = feat.permute(0,2,3,1).reshape(-1, Cc)   # (B*H*W, C)
        # standardize per-channel using feature mean/std
        a = (a - fmean) / fstd
        if token_subsample < 1.0:
            N = a.size(0)
            keep = int(max(1, round(N * token_subsample)))
            idx = torch.randperm(N, device=a.device)[:keep]
            a = a[idx]
        return a

    for epoch in range(1, epochs+1):
        if is_ddp and hasattr(loader, "sampler") and loader.sampler is not None:
            loader.sampler.set_epoch(epoch)

        sae.train()
        epoch_loss, epoch_n = 0.0, 0
        pbar = tqdm(loader, disable=not is_main, ncols=100, desc=f"SAE epoch {epoch:03d}")

        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            tokens = tokens_from_images(x)                  # (Ntok, C)
            # Split tokens into manageable chunks if very large
            chunk_sz = 16384
            n_tokens = tokens.size(0)
            n_chunks = (n_tokens + chunk_sz - 1) // chunk_sz

            optimizer.zero_grad(set_to_none=True)
            total_tok = 0
            loss_weighted_sum = 0.0

            # accumulate over chunks
            for ci in range(n_chunks):
                a = tokens[ci*chunk_sz: (ci+1)*chunk_sz]
                total_tok += a.size(0)
                with autocast():
                    recon, vals, idx = sae(a)
                    chunk_loss = ((recon - a)**2).mean()

                # bookkeeping for logging
                loss_weighted_sum += chunk_loss.detach().item() * a.size(0)

                # update usage ema per chunk
                if is_ddp:
                    base = sae.module
                    base.update_usage_ema(idx, vals)
                else:
                    sae.update_usage_ema(idx, vals)

                scaler.scale(chunk_loss).backward()

            # one optimizer step per image batch
            scaler.step(optimizer)
            scaler.update()

            # renorm after optimizer step
            if renorm_every and ((global_step % renorm_every) == 0):
                (sae.module if is_ddp else sae).renorm_decoder()

            # resample once per batch and broadcast to keep replicas in sync
            if is_ddp:
                base = sae.module
                res = 0
                if is_main:
                    res = base.resample_dead(tokens, dead_threshold, resample_after_steps, max_resamples_per_step)
                dist.barrier()
                broadcast_module_(base, src=0)
            else:
                res = sae.resample_dead(tokens, dead_threshold, resample_after_steps, max_resamples_per_step)

            # update running epoch stats correctly (token-weighted)
            avg_chunk_loss = loss_weighted_sum / max(1, total_tok)
            epoch_loss += avg_chunk_loss * total_tok
            epoch_n += total_tok
            global_step += 1
            pbar.set_postfix(loss=f"{(epoch_loss/max(1,epoch_n)):.5f}", k=top_k, res=res)

        # DDP reduce epoch stats
        if is_ddp:
            t = torch.tensor([epoch_loss, epoch_n], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            epoch_loss, epoch_n = t.tolist()

        epoch_mse = epoch_loss / max(1, epoch_n)
        if is_main:
            print(f"Epoch {epoch:03d} SAE MSE: {epoch_mse:.6f}")
            # Save checkpoint
            to_save = {
                "epoch": epoch,
                "sae_state": (sae.module if is_ddp else sae).state_dict(),
                "config": {
                    "latent_dim": latent_dim,
                    "target_density": target_density,
                    "top_k": top_k,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "dead_threshold": dead_threshold,
                    "resample_after_steps": resample_after_steps,
                    "max_resamples_per_step": max_resamples_per_step,
                },
                "feature_mean": fmean.cpu(),
                "feature_std": fstd.cpu(),
                "backbone": {
                    "checkpoint_path": checkpoint_path,
                    "num_classes": num_classes,
                },
                "epoch_mse": epoch_mse,
            }
            torch.save(to_save, save_dir / "last_sae.pt")
            if epoch_mse < best_loss:
                best_loss = epoch_mse
                torch.save(to_save, save_dir / "best_sae.pt")

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
