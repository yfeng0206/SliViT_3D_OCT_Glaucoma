# Phase 2: Full Fine-Tuning

All parameters trainable with per-component learning rates: ConvNeXt (lowest) < ViT (mid) < head (highest). The low ConvNeXt LR preserves Kermany-pretrained weights while allowing glaucoma-specific adaptation.

## Summary

Full fine-tuning pushes test AUC from ~0.83 (Phase 1) to 0.869 (Run 3). The best result came from our first Phase 2 attempt — higher LRs found a better optimum faster, and early stopping captured the peak. Controlled slice comparison (Runs 5 vs 6) shows 32 slices is sufficient.

## Runs

| Run | Slices | LR (fe) | LR (vit) | LR (head) | Dropout | Batch/GPU | Accum | Eff Batch | Val AUC | Test AUC | Best Epoch |
|-----|--------|---------|----------|-----------|---------|-----------|-------|-----------|---------|----------|------------|
| **3** | **32** | **5e-6** | **1e-5** | **5e-5** | **0.10** | **2** | **N/A** | **8** | **0.846** | **0.869** | **4** |
| 4 | 64 | 1e-6 | 5e-6 | 5e-5 | 0.15 | 1 | N/A | 4 | 0.841 | 0.868 | 6 |
| 5 | 64 | 1e-6 | 5e-6 | 5e-5 | 0.15 | 2 | 2 | 16 | 0.845 | 0.866 | 9 |
| 6 | 32 | 1e-6 | 5e-6 | 5e-5 | 0.15 | 2 | 2 | 16 | 0.840 | 0.864 | 7 |

## Training Config

- **Hardware:** 4x NVIDIA T4 (16GB each)
- **Timing:** 32 slices, bs=2/GPU: ~5 min/epoch, ~1 hour per run. 64 slices, bs=1/GPU: ~24 min/epoch, 4-5 hours per run.
- **Optimizer:** AdamW, weight decay 0.01
- **Scheduler:** Cosine LR with 3-epoch warmup
- **Loss:** BCEWithLogitsLoss
- **Early stopping:** on val AUC, patience=5
- **Stack:** PyTorch 1.13.1, CUDA 11.7, HuggingFace Transformers, 4-GPU DDP with fp16

## Detailed Run Analysis

### Run 3 — Best result (0.869 test AUC)

The first full fine-tuning attempt and the best one. Used higher LRs than subsequent runs (ConvNeXt 5e-6, ViT 1e-5) with lighter dropout (0.1). Peaked early at epoch 4, suggesting the aggressive LRs found a good optimum quickly before overfitting set in. Early stopping captured this peak.

The effective batch size of 8 (2/GPU x 4 GPUs, no accumulation) was smaller than later runs, which may have acted as implicit regularization through noisier gradients.

### Run 4 — More slices, confounded comparison (0.868 test AUC)

Increased to 64 slices for denser spatial coverage. Also lowered LRs (ConvNeXt 1e-6, ViT 5e-6) and increased dropout to 0.15. Had to drop batch size to 1/GPU due to VRAM constraints, giving an effective batch of only 4.

The result (0.868) is essentially tied with Run 3 (0.869), but the comparison is confounded: slice count, learning rates, dropout, and batch size all changed simultaneously. This motivated the controlled comparison in Runs 5-6.

### Runs 5 and 6 — Controlled slice comparison (0.866 vs 0.864)

Used gradient accumulation (2 steps with bs=2/GPU) to match an effective batch size of 16 for both 32 and 64 slices, with identical LRs and dropout. Results:
- Run 5 (64 slices): 0.866 test AUC, best epoch 9
- Run 6 (32 slices): 0.864 test AUC, best epoch 7

The 0.002 difference is negligible. **32 slices is sufficient**, and 64 slices costs ~5x more in compute (24 min/epoch vs 5 min/epoch) with no meaningful accuracy gain.

## Overfitting Analysis

Every Phase 2 run follows the same pattern: validation loss starts climbing after epoch 4-6 while training loss continues to decrease. By end of training:
- **Train loss:** ~0.04
- **Val loss:** >1.0

Interventions tried:
- **Dropout** (0.10 in Run 3, 0.15 in Runs 4-6): marginal help
- **Lower learning rates** (Runs 4-6 vs Run 3): delayed but did not prevent overfitting
- **Per-component LRs** (ConvNeXt < ViT < head): slowed ConvNeXt drift but overall pattern unchanged

### Root cause

The model has 50M trainable parameters for only 6,000 training images. Of the 50M, approximately 33M are in projection layers forced by the mismatch between `vit_dim=256` and 20 attention heads (256 doesn't divide by 20, so each transformer block projects 256 to 1280 and back). Switching to 16 heads would make `dim_head=16` and eliminate these projections, cutting trainable params to ~15M — a more appropriate ratio for 6K training samples.

![Phase 2 Training](../../../results/phase2_training.png)
