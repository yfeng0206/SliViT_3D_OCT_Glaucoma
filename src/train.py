"""
SLIViT training script. Supports frozen FE (Phase 1) and full fine-tuning
(Phase 2) with per-component LRs, DDP, mixed precision, and gradient accumulation.

Usage:
    torchrun --nproc_per_node=4 train.py --data_dir /path/to/data --phase 1
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score

from dataset import FairVisionGlaucomaDataset
from model import SLIViT


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("slivit")


def setup_logging(output_dir):
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger.addHandler(logging.StreamHandler())
    logger.handlers[-1].setFormatter(fmt)
    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def log(msg):
    if is_main_process():
        logger.info(msg)


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    n_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images).squeeze(-1)
            loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        total_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    if dist.is_initialized():
        all_labels_cuda = all_labels.to(device)
        all_probs_cuda = all_probs.to(device)
        world = dist.get_world_size()
        g_labels = [torch.zeros_like(all_labels_cuda) for _ in range(world)]
        g_probs = [torch.zeros_like(all_probs_cuda) for _ in range(world)]
        dist.all_gather(g_labels, all_labels_cuda)
        dist.all_gather(g_probs, all_probs_cuda)
        all_labels = torch.cat(g_labels).cpu()
        all_probs = torch.cat(g_probs).cpu()

        loss_t = torch.tensor([total_loss, n_samples], device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        total_loss = loss_t[0].item()
        n_samples = int(loss_t[1].item())

    avg_loss = total_loss / max(n_samples, 1)
    y = all_labels.numpy()
    p = all_probs.numpy()
    auc = roc_auc_score(y, p) if len(np.unique(y)) >= 2 else 0.5
    return avg_loss, auc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, epoch, accum_steps=1):
    model.train()
    total_loss = 0.0
    n_samples = 0
    num_steps = len(dataloader)

    optimizer.zero_grad(set_to_none=True)
    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images).squeeze(-1)
            loss = criterion(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == num_steps:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps * labels.size(0)
        n_samples += labels.size(0)

        if is_main_process() and (step + 1) % 50 == 0:
            lrs = scheduler.get_last_lr()
            lr_str = "/".join(["%.2e" % lr for lr in lrs])
            log("  Step %d/%d | Loss %.4f | LR %s" % (step + 1, num_steps, loss.item() * accum_steps, lr_str))

    if dist.is_initialized():
        loss_t = torch.tensor([total_loss, n_samples], device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        total_loss = loss_t[0].item()
        n_samples = int(loss_t[1].item())

    return total_loss / max(n_samples, 1)


# ---------------------------------------------------------------------------
# Build optimizer with per-component LR
# ---------------------------------------------------------------------------

def build_optimizer(model, args):
    """Create AdamW with separate LR for feature_extractor, ViT, and head."""
    # Access the unwrapped model (inside DDP)
    m = model.module if hasattr(model, "module") else model

    fe_params = []
    vit_params = []
    head_params = []

    for name, param in m.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("convnext"):
            fe_params.append(param)
        elif name.startswith("head"):
            head_params.append(param)
        else:
            # token_proj, cls_token, pos_embed, vit.*
            vit_params.append(param)

    param_groups = []
    if fe_params:
        param_groups.append({"params": fe_params, "lr": args.lr_fe})
    if vit_params:
        param_groups.append({"params": vit_params, "lr": args.lr_vit})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_head})

    return torch.optim.AdamW(param_groups, weight_decay=0.01)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SLIViT Glaucoma Training")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_slices", type=int, default=32)
    parser.add_argument("--fe_checkpoint", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    # Phase and per-component learning rates
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="1=freeze FE, train ViT+head; 2=full fine-tune")
    parser.add_argument("--lr_fe", type=float, default=5e-6,
                        help="LR for feature extractor (Phase 2 only)")
    parser.add_argument("--lr_vit", type=float, default=2e-5,
                        help="LR for ViT encoder + token proj + pos embed")
    parser.add_argument("--lr_head", type=float, default=1e-4,
                        help="LR for classification head")
    # Kept for backward compat but ignored if lr_vit/lr_head are set
    parser.add_argument("--lr", type=float, default=None, help="(deprecated, use lr_vit/lr_head)")
    args = parser.parse_args()

    # ---- DDP setup -------------------------------------------------------
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(args.seed + dist.get_rank())

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    setup_logging(args.output_dir)

    phase_desc = "Phase 1: frozen FE, train ViT+head" if args.phase == 1 else "Phase 2: full fine-tune"
    log("=" * 60)
    log("SLIViT Glaucoma Training — %s" % phase_desc)
    log("=" * 60)
    log("Config: %s" % json.dumps(vars(args), indent=2))
    log("World size: %d | Device: %s | GPU: %s | PyTorch: %s"
        % (dist.get_world_size(), device, torch.cuda.get_device_name(device), torch.__version__))

    # ---- Datasets --------------------------------------------------------
    train_dir = os.path.join(args.data_dir, "Training")
    val_dir = os.path.join(args.data_dir, "Validation")

    train_dataset = FairVisionGlaucomaDataset(train_dir, num_slices=args.num_slices)
    val_dataset = FairVisionGlaucomaDataset(val_dir, num_slices=args.num_slices)

    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    log("Train: %d samples, %d batches/epoch" % (len(train_dataset), len(train_loader)))
    log("Val:   %d samples, %d batches/epoch" % (len(val_dataset), len(val_loader)))
    log("Effective batch size: %d" % (args.batch_size * dist.get_world_size()))

    # ---- Model -----------------------------------------------------------
    freeze_fe = (args.phase == 1)
    model = SLIViT(
        num_slices=args.num_slices,
        fe_checkpoint=args.fe_checkpoint,
        freeze_fe=freeze_fe,
    ).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=(args.phase == 2))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log("Parameters: %s total | %s trainable (%.1f%%)"
        % (format(total, ","), format(trainable, ","), 100 * trainable / total))

    # ---- Optimizer / scheduler / loss ------------------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, args)

    # Log param groups
    for i, pg in enumerate(optimizer.param_groups):
        n_params = sum(p.numel() for p in pg["params"])
        log("  Param group %d: %s params, lr=%.1e" % (i, format(n_params, ","), pg["lr"]))

    steps_per_epoch = len(train_loader) // args.accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_epochs=3, total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )
    scaler = GradScaler()

    log("Scheduler: cosine with 3 warmup epochs, %d steps/epoch" % steps_per_epoch)
    log("-" * 60)

    # ---- CSV logger ------------------------------------------------------
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    if is_main_process():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_auc", "lr", "time_s"])

    # ---- Training loop ---------------------------------------------------
    best_auc = 0.0
    total_train_time = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch,
            accum_steps=args.accum_steps,
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        total_train_time += elapsed
        lrs = scheduler.get_last_lr()
        lr_str = "/".join(["%.2e" % lr for lr in lrs])

        improved = val_auc > best_auc
        marker = " *" if improved else ""

        log("Epoch %d/%d (%4.0fs) | Train Loss: %.4f | Val Loss: %.4f | "
            "Val AUC: %.4f | LR: %s%s"
            % (epoch, args.epochs, elapsed, train_loss, val_loss, val_auc, lr_str, marker))

        if is_main_process():
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, val_loss, val_auc, lr_str, elapsed])

        if improved:
            best_auc = val_auc
            epochs_no_improve = 0
            if is_main_process():
                ckpt_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "val_loss": val_loss,
                    "args": vars(args),
                }, ckpt_path)
                log("  -> New best model saved (AUC=%.4f)" % val_auc)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                log("Early stopping: no improvement for %d epochs" % args.patience)
                break

    # ---- Test set evaluation ------------------------------------------------
    test_dir = os.path.join(args.data_dir, "Test")
    test_auc = None
    test_loss = None
    if os.path.exists(test_dir):
        log("-" * 60)
        log("Evaluating best model on TEST set...")

        # Load best checkpoint
        ckpt_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.module.load_state_dict(ckpt["model_state_dict"])
            log("  Loaded best model from epoch %d (val AUC=%.4f)" % (ckpt["epoch"], ckpt["val_auc"]))

        test_dataset = FairVisionGlaucomaDataset(test_dir, num_slices=args.num_slices)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, sampler=test_sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )
        log("  Test: %d samples" % len(test_dataset))

        test_loss, test_auc = evaluate(model, test_loader, criterion, device)
        log("  TEST Loss: %.4f | TEST AUC: %.4f" % (test_loss, test_auc))
    else:
        log("No Test directory found, skipping test evaluation.")

    # ---- Summary ---------------------------------------------------------
    log("=" * 60)
    log("Training complete")
    log("  Total time: %.0f s (%.1f min)" % (total_train_time, total_train_time / 60))
    log("  Best Val AUC: %.4f" % best_auc)
    if test_auc is not None:
        log("  Test AUC:     %.4f" % test_auc)
    log("=" * 60)

    if is_main_process():
        summary = {
            "best_val_auc": best_auc,
            "test_auc": test_auc,
            "test_loss": test_loss,
            "total_train_time_s": total_train_time,
            "epochs_run": epoch,
            "phase": args.phase,
            "num_slices": args.num_slices,
            "config": vars(args),
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
