"""
Evaluate saved checkpoints on the test set.
Downloads checkpoint from blob, loads model, runs test evaluation.

Usage:
    python eval_test.py --blob_prefix training-results/p2_slices32_... --num_slices 32 --phase 2
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import roc_auc_score

from dataset import FairVisionGlaucomaDataset
from model import SLIViT


def evaluate_test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    n_samples = 0

    with torch.no_grad():
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

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    avg_loss = total_loss / max(n_samples, 1)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) >= 2 else 0.5

    return avg_loss, auc, n_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--num_slices", type=int, default=32)
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save test results JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)
    if torch.cuda.is_available():
        print("GPU: %s" % torch.cuda.get_device_name(0))

    # Load model
    print("Loading model (phase=%d, slices=%d)..." % (args.phase, args.num_slices))
    freeze_fe = (args.phase == 1)
    model = SLIViT(num_slices=args.num_slices, freeze_fe=freeze_fe).to(device)

    # Load checkpoint
    print("Loading checkpoint: %s" % args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print("  Loaded from epoch %d, val_auc=%.4f" % (ckpt["epoch"], ckpt["val_auc"]))

    # Load test data
    test_dir = os.path.join(args.data_dir, "Test")
    print("Loading test data from: %s" % test_dir)
    test_dataset = FairVisionGlaucomaDataset(test_dir, num_slices=args.num_slices)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, shuffle=False,
    )
    print("  Test samples: %d" % len(test_dataset))

    # Also eval on val for comparison
    val_dir = os.path.join(args.data_dir, "Validation")
    val_dataset = FairVisionGlaucomaDataset(val_dir, num_slices=args.num_slices)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, shuffle=False,
    )

    criterion = nn.BCEWithLogitsLoss()

    # Evaluate
    print("\n=== Validation Set ===")
    val_loss, val_auc, val_n = evaluate_test(model, val_loader, criterion, device)
    print("  Val Loss: %.4f | Val AUC: %.4f | N=%d" % (val_loss, val_auc, val_n))

    print("\n=== Test Set ===")
    test_loss, test_auc, test_n = evaluate_test(model, test_loader, criterion, device)
    print("  Test Loss: %.4f | Test AUC: %.4f | N=%d" % (test_loss, test_auc, test_n))

    print("\n=== Summary ===")
    print("  Checkpoint epoch: %d" % ckpt["epoch"])
    print("  Val AUC (from training): %.4f" % ckpt["val_auc"])
    print("  Val AUC (re-evaluated):  %.4f" % val_auc)
    print("  Test AUC:                %.4f" % test_auc)

    # Save results
    results = {
        "checkpoint_epoch": ckpt["epoch"],
        "val_auc_training": ckpt["val_auc"],
        "val_auc_reeval": val_auc,
        "val_loss": val_loss,
        "test_auc": test_auc,
        "test_loss": test_loss,
        "num_slices": args.num_slices,
        "phase": args.phase,
        "test_samples": test_n,
        "val_samples": val_n,
    }

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print("  Results saved to: %s" % args.output_file)

    return results


if __name__ == "__main__":
    main()
