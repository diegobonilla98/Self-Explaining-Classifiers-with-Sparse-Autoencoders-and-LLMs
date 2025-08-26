import os
import io
import math
import json
import time
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageFilter
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import snapshot_download


# ------------------------------
# Config
# ------------------------------
@dataclass
class TrainConfig:
    data_root: str = r"D:\TuBerlin"  # Single path for all dataset files
    
    output_dir: str = "runs/tu-berlin"
    probe_size: int = 200
    num_workers: int = 4  # Will be adjusted based on GPU count

    image_size: int = 224
    batch_size: int = 64  # Per GPU batch size
    epochs: int = 180
    warmup_epochs: int = 5

    optimizer: str = "adamw"
    lr: float = 3e-4  # Base learning rate, will be scaled
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.999)

    label_smoothing: float = 0.1
    grad_clip_norm: float = 1.0
    amp: bool = True
    pretrained: bool = True
    seed: int = 42

    # Gradient accumulation (simulate larger effective batch size without OOM)
    grad_accum_steps: int = 8

    # Multi-GPU settings
    distributed: bool = False  # Will be set automatically
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    dist_backend: str = "nccl"
    dist_url: str = "env://"

    # validation frequency and probe logging
    val_every: int = 1
    log_probe_every: int = 5


# ------------------------------
# Utils
# ------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed(cfg: TrainConfig) -> TrainConfig:
    """Setup distributed training configuration based on available GPUs."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s)")
        
        if num_gpus > 1:
            # Multi-GPU setup
            cfg.distributed = True
            cfg.world_size = num_gpus
            
            # Scale learning rate linearly with number of GPUs
            cfg.lr = cfg.lr * num_gpus
            
            # Adjust num_workers based on GPU count (but respect Windows limitations)
            if os.name == 'nt':  # Windows
                cfg.num_workers = min(2, num_gpus)
            else:  # Linux/Unix
                cfg.num_workers = min(8, num_gpus * 2)
                
            print(f"Multi-GPU training enabled: {num_gpus} GPUs")
            print(f"Scaled learning rate: {cfg.lr:.2e}")
            print(f"Effective batch size: {cfg.batch_size * num_gpus}")
        else:
            # Single GPU
            print("Single GPU training")
            if os.name == 'nt':  # Windows
                cfg.num_workers = 0
    else:
        print("CPU training (no CUDA available)")
        cfg.num_workers = 0
    
    return cfg


def init_distributed_mode(cfg: TrainConfig, rank: int) -> None:
    """Initialize distributed training for a specific rank."""
    cfg.rank = rank
    cfg.local_rank = rank
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    # Initialize process group
    dist.init_process_group(
        backend=cfg.dist_backend,
        init_method=cfg.dist_url,
        world_size=cfg.world_size,
        rank=rank
    )
    
    # Set different random seed for each process
    set_seed(cfg.seed + rank)


def ensure_dataset_downloaded(data_root: str) -> None:
    """Download TU-Berlin dataset if it doesn't exist at the specified path."""
    if not os.path.exists(data_root) or not os.path.exists(os.path.join(data_root, "data")):
        print(f"Dataset not found at {data_root}. Downloading...")
        snapshot_download(
            repo_id="kmewhort/tu-berlin-png",
            repo_type="dataset",
            local_dir=data_root
        )
        print(f"Dataset downloaded to {data_root}")
    else:
        print(f"Dataset found at {data_root}")


def load_label_decoder(data_root: str) -> Dict[int, str]:
    """Parse README.md for class_label->names mapping produced by datasets.
    Matches helper used in download_tuberlin_dataset.py
    """
    readme_path = os.path.join(data_root, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.find("class_label:")
    if start == -1:
        raise ValueError("class_label section not found in README.md")
    start = content.find("names:", start)
    if start == -1:
        raise ValueError("names section not found in README.md")
    end = content.find("splits:", start)
    yaml_str = content[start:end]
    yaml_str = "names:\n" + "\n".join(line for line in yaml_str.splitlines()[1:])
    import yaml
    names = yaml.safe_load(yaml_str)["names"]
    return {int(k): v for k, v in names.items()}


# ------------------------------
# Data
# ------------------------------
class RandomStrokeJitter:
    """Randomly thicken or thin black strokes on white background (PIL-only)."""
    def __init__(self, p: float = 0.25, sizes: Tuple[int, int] = (3, 5)):
        self.p = p
        self.sizes = sizes

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        k = random.choice(self.sizes)
        if random.random() < 0.5:
            return img.filter(ImageFilter.MinFilter(size=k))
        else:
            return img.filter(ImageFilter.MaxFilter(size=k))


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            shear=5,
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomStrokeJitter(p=0.25, sizes=(3, 5)),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, val_tfms


class TuBerlinParquetDataset(Dataset):
    """Dataset for TU-Berlin parquet files downloaded from HuggingFace snapshot.

    Expects parquet files under cfg.parquet_dir with columns ["image", "label"],
    where image stores {"bytes": ...} dicts. This mirrors the sampling code in download_tuberlin_dataset.py
    """

    def __init__(self, parquet_paths: List[str], transform=None):
        self.parquet_paths = parquet_paths
        self.transform = transform
        # Concatenate parquet data lazily by reading each into a list of dataframes
        frames: List[pd.DataFrame] = []
        for p in parquet_paths:
            table = pq.read_table(p)
            frames.append(table.to_pandas())
        self.df = pd.concat(frames, ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_dict = row["image"]
        label = int(row["label"])  # ensure int
        img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
        img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale PIL
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TransformSubset(Dataset):
    """Dataset subset with transform applied. Moved to module level for Windows multiprocessing compatibility."""
    def __init__(self, base_df: pd.DataFrame, indices: List[int], transform):
        self.base_df = base_df
        self.indices = list(indices)
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, i):
        idx = self.indices[i]
        row = self.base_df.iloc[idx]
        img_dict = row["image"]
        label = int(row["label"])  # ensure int
        img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        img = self.transform(img)
        return img, label


# ------------------------------
# Model
# ------------------------------
class SketchResNet18(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()

        def set_block_stride(layer: nn.Sequential, stride: int) -> None:
            b0 = layer[0]
            if hasattr(b0, "conv1"):
                b0.conv1.stride = (stride, stride)
            if b0.downsample is not None and hasattr(b0.downsample[0], "stride"):
                b0.downsample[0].stride = (stride, stride)

        set_block_stride(m.layer1, 1)  # /1
        set_block_stride(m.layer2, 2)  # /2
        set_block_stride(m.layer3, 1)  # keep /2
        set_block_stride(m.layer4, 2)  # /8 → 28x28 for 224 input

        self.backbone = m
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(m.fc.in_features, num_classes)
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor, return_map: bool = False):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # Identity
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        A = self.backbone.layer4(x)  # (B, 512, 28, 28)
        z = self.pool(A).flatten(1)
        logits = self.fc(z)
        return (logits, A) if return_map else logits


# ------------------------------
# Training helpers
# ------------------------------
class CosineLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, epochs: int, warmup_epochs: int, steps_per_epoch: int):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = epochs * steps_per_epoch
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.warmup_steps > 0 and self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / max(1, self.warmup_steps)
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, desc: str = "Validation") -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    pbar = tqdm(loader, desc=desc, leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)
        
        # Update progress bar with current metrics
        current_acc = correct / max(1, total)
        pbar.set_postfix({'acc': f'{current_acc:.3f}'})
    
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def compute_A_stats(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, List[float]]:
    model.eval()
    sums = None
    sumsqs = None
    count = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        logits, A = model(images, return_map=True)
        B, C, H, W = A.shape
        a = A.permute(1, 0, 2, 3).contiguous().view(C, -1)
        if sums is None:
            sums = a.sum(dim=1)
            sumsqs = (a * a).sum(dim=1)
        else:
            sums += a.sum(dim=1)
            sumsqs += (a * a).sum(dim=1)
        count += a.shape[1]
    if count == 0:
        return {"mean": [], "std": []}
    mean = (sums / count).tolist()
    var = (sumsqs / count - (sums / count) ** 2).clamp_min(0)
    std = var.sqrt().tolist()
    return {"mean": mean, "std": std}


@torch.no_grad()
def log_probe_activations(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: str, step_tag: str, max_items: int = 50):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    saved = 0
    batch_idx = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits, A = model(images, return_map=True)
        # Save per-sample pooled CAM-like map for quick inspection
        # CAM approx: weight*feature, but we will save just feature maps and logits
        for i in range(images.size(0)):
            if saved >= max_items:
                return
            item = {
                "target": int(targets[i].item()),
                "logits": logits[i].detach().cpu().tolist(),
            }
            torch.save(A[i].detach().cpu(), os.path.join(out_dir, f"{step_tag}_A_{batch_idx}_{i}.pt"))
            with open(os.path.join(out_dir, f"{step_tag}_meta_{batch_idx}_{i}.json"), "w") as f:
                json.dump(item, f)
            saved += 1
        batch_idx += 1


# ------------------------------
# Main train
# ------------------------------

def find_parquet_files(data_root: str) -> List[str]:
    parquet_dir = os.path.join(data_root, "data")
    if not os.path.isdir(parquet_dir):
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")
    files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {parquet_dir}")
    return sorted(files)


def create_loaders(cfg: TrainConfig, label_decoder: Dict[int, str]) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    train_tfms, val_tfms = build_transforms(cfg.image_size)
    parquet_paths = find_parquet_files(cfg.data_root)
    full_ds = TuBerlinParquetDataset(parquet_paths, transform=None)

    # Stratified split is better, but keep simple: random split
    num_total = len(full_ds)
    num_val = max(1, int(0.1 * num_total))
    num_train = num_total - num_val
    generator = torch.Generator().manual_seed(cfg.seed)
    train_indices, val_indices = random_split(range(num_total), lengths=[num_train, num_val], generator=generator)

    train_ds = TransformSubset(full_ds.df, list(train_indices.indices), train_tfms)
    val_ds = TransformSubset(full_ds.df, list(val_indices.indices), val_tfms)

    # Probe set: fixed subset from validation set
    probe_size = min(cfg.probe_size, len(val_ds))
    probe_indices = list(range(probe_size))
    probe_ds = torch.utils.data.Subset(val_ds, probe_indices)

    # Infer num_classes from labels present
    num_classes = int(max(full_ds.df["label"]) + 1)

    # Create samplers for distributed training
    if cfg.distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=cfg.world_size, rank=cfg.rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=cfg.world_size, rank=cfg.rank, shuffle=False)
        probe_sampler = DistributedSampler(probe_ds, num_replicas=cfg.world_size, rank=cfg.rank, shuffle=False)
        
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler, 
                                num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, sampler=val_sampler,
                              num_workers=cfg.num_workers, pin_memory=True)
        probe_loader = DataLoader(probe_ds, batch_size=cfg.batch_size, sampler=probe_sampler,
                                num_workers=cfg.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                                num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, 
                              num_workers=cfg.num_workers, pin_memory=True)
        probe_loader = DataLoader(probe_ds, batch_size=cfg.batch_size, shuffle=False, 
                                num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, probe_loader, num_classes


def train_worker(rank: int, cfg: TrainConfig):
    """Training function for each GPU process."""
    if cfg.distributed:
        init_distributed_mode(cfg, rank)
    else:
        set_seed(cfg.seed)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Only rank 0 handles dataset download and logging
    if rank == 0:
        ensure_dataset_downloaded(cfg.data_root)
        os.makedirs(cfg.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, "tensorboard"))
        print(f"Using device: {device}")
    else:
        writer = None
    
    # Synchronize all processes after dataset download
    if cfg.distributed:
        dist.barrier()
    
    label_decoder = load_label_decoder(cfg.data_root)
    train_loader, val_loader, probe_loader, num_classes = create_loaders(cfg, label_decoder)
    
    if rank == 0:
        effective_batch_size = cfg.batch_size * cfg.world_size * max(1, cfg.grad_accum_steps)
        print(f"Training on {num_classes} classes with {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples")
        print(f"Per-GPU batch size: {cfg.batch_size}, Effective batch size: {effective_batch_size}")

    model = SketchResNet18(num_classes=num_classes, pretrained=cfg.pretrained).to(device)
    
    # Wrap model for distributed training
    if cfg.distributed:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=cfg.weight_decay)

    num_batches = len(train_loader)
    updates_per_epoch = math.ceil(num_batches / max(1, cfg.grad_accum_steps))
    scheduler = CosineLRScheduler(optimizer, base_lr=cfg.lr, epochs=cfg.epochs, warmup_epochs=cfg.warmup_epochs, steps_per_epoch=updates_per_epoch)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val_acc = 0.0
    start_time = time.time()
    global_step = 0

    # Main training loop with tqdm (only show progress on rank 0)
    if rank == 0:
        epoch_pbar = tqdm(range(cfg.epochs), desc="Training", unit="epoch")
    else:
        epoch_pbar = range(cfg.epochs)
    
    for epoch in epoch_pbar:
        # Set epoch for distributed sampler
        if cfg.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # Training loop with tqdm (only show progress on rank 0)
        if rank == 0:
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}", leave=False, unit="batch")
        else:
            train_pbar = train_loader

        # Zero gradients before starting accumulation
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, targets) in enumerate(train_pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            accumulate_steps = max(1, cfg.grad_accum_steps)
            is_update_step = ((batch_idx + 1) % accumulate_steps == 0) or ((batch_idx + 1) == num_batches)

            # Avoid DDP gradient sync on micro-steps
            use_no_sync = cfg.distributed and isinstance(model, DDP) and accumulate_steps > 1 and not is_update_step
            ctx = model.no_sync() if use_no_sync else torch.no_grad() if False else None

            if ctx is not None:
                with ctx:
                    with torch.cuda.amp.autocast(enabled=cfg.amp):
                        logits = model(images)
                        loss_unscaled = F.cross_entropy(logits, targets, label_smoothing=cfg.label_smoothing)
                        loss = loss_unscaled / accumulate_steps
                    scaler.scale(loss).backward()
            else:
                with torch.cuda.amp.autocast(enabled=cfg.amp):
                    logits = model(images)
                    loss_unscaled = F.cross_entropy(logits, targets, label_smoothing=cfg.label_smoothing)
                    loss = loss_unscaled / accumulate_steps
                scaler.scale(loss).backward()

            if is_update_step:
                if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                else:
                    grad_norm = 0.0
                scaler.step(optimizer)
                scaler.update()
                current_lr = scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                current_lr = optimizer.param_groups[0]["lr"]
                grad_norm = 0.0

            # Update metrics
            batch_loss = loss_unscaled.item()
            preds = logits.argmax(dim=1)
            batch_correct = (preds == targets).sum().item()
            batch_size = images.size(0)
            
            epoch_loss += batch_loss * batch_size
            epoch_correct += batch_correct
            epoch_total += batch_size
            
            # Update progress bar (only on rank 0)
            if rank == 0:
                current_acc = epoch_correct / max(1, epoch_total)
                train_pbar.set_postfix({
                    'loss': f'{batch_loss:.3f}',
                    'acc': f'{current_acc:.3f}',
                    'lr': f'{current_lr:.2e}'
                })
            
            # Log to TensorBoard every few batches (only on rank 0)
            if rank == 0 and batch_idx % 10 == 0:  # Log every 10 batches
                writer.add_scalar('Train/BatchLoss', batch_loss, global_step)
                writer.add_scalar('Train/BatchAcc', batch_correct / batch_size, global_step)
                writer.add_scalar('Train/LearningRate', current_lr, global_step)
                if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                    writer.add_scalar('Train/GradNorm', grad_norm, global_step)
            
            global_step += 1

        # Synchronize metrics across all processes for distributed training
        if cfg.distributed:
            # Convert to tensors for all_reduce
            metrics = torch.tensor([epoch_loss, epoch_correct, epoch_total], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            epoch_loss, epoch_correct, epoch_total = metrics.tolist()

        train_loss = epoch_loss / max(1, epoch_total)
        train_acc = epoch_correct / max(1, epoch_total)

        # Log epoch training metrics (only on rank 0)
        if rank == 0:
            writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            writer.add_scalar('Train/EpochAcc', train_acc, epoch)

        should_validate = ((epoch + 1) % cfg.val_every == 0) or (epoch == cfg.epochs - 1)
        if should_validate:
            val_desc = f"Val Epoch {epoch+1:03d}" if rank == 0 else "Validation"
            val_loss, val_acc = evaluate(model, val_loader, device, desc=val_desc)
            
            # Synchronize validation metrics across processes
            if cfg.distributed:
                val_metrics = torch.tensor([val_loss, val_acc], device=device)
                dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
                val_loss, val_acc = (val_metrics / cfg.world_size).tolist()
            
            # Only rank 0 handles logging and checkpointing
            if rank == 0:
                writer.add_scalar('Val/Loss', val_loss, epoch)
                writer.add_scalar('Val/Acc', val_acc, epoch)
                
                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc = val_acc

                    # Compute A(x) stats on validation loader
                    A_stats = compute_A_stats(model, val_loader, device)

                    # Only save the best checkpoint (save the underlying model for DDP)
                    model_state = model.module.state_dict() if cfg.distributed else model.state_dict()
                    ckpt = {
                        "epoch": epoch,
                        "model_state": model_state,
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "cfg": vars(cfg),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "A_stats": A_stats,
                        "num_classes": num_classes,
                        "global_step": global_step,
                    }
                    torch.save(ckpt, os.path.join(cfg.output_dir, "checkpoint_best.pt"))
                    
                    # Log A(x) channel statistics
                    if A_stats.get("mean") and A_stats.get("std"):
                        writer.add_histogram('Features/ChannelMeans', torch.tensor(A_stats["mean"]), epoch)
                        writer.add_histogram('Features/ChannelStds', torch.tensor(A_stats["std"]), epoch)

                # Probe logging: save a few last-conv maps and logits
                if (epoch + 1) % cfg.log_probe_every == 0:
                    probe_dir = os.path.join(cfg.output_dir, "probe")
                    log_probe_activations(model, probe_loader, device, probe_dir, step_tag=f"epoch{epoch:03d}", max_items=50)
            else:
                is_best = False

        # Update main progress bar (only on rank 0)
        if rank == 0:
            epoch_pbar.set_postfix({
                'train_acc': f'{train_acc:.3f}',
                'val_acc': f'{best_val_acc:.3f}',
                'best': '★' if is_best and should_validate else ''
            })

    # Clean up distributed training
    if cfg.distributed:
        dist.destroy_process_group()

    if rank == 0:
        writer.close()
        total_minutes = (time.time() - start_time) / 60.0
        print(f"\nFinished training in {total_minutes:.1f} min. Best val acc: {best_val_acc:.4f}")
        print(f"Best checkpoint saved to: {os.path.join(cfg.output_dir, 'checkpoint_best.pt')}")
        print(f"TensorBoard logs saved to: {os.path.join(cfg.output_dir, 'tensorboard')}")


def main():
    """Main function that sets up distributed training or single GPU training."""
    cfg = TrainConfig()
    cfg = setup_distributed(cfg)
    
    if cfg.distributed:
        # Multi-GPU training with spawn
        mp.spawn(train_worker, args=(cfg,), nprocs=cfg.world_size, join=True)
    else:
        # Single GPU or CPU training
        train_worker(0, cfg)


if __name__ == "__main__":
    main()
