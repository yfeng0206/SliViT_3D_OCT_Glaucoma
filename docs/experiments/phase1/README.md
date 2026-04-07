# Phase 1: Frozen Feature Extractor

ConvNeXt-Tiny frozen (ImageNet + Kermany OCT pretrained). Only the ViT integrator and classification head are trained.

## Summary

The frozen ConvNeXt features cap performance at ~0.83 val AUC. Splitting the learning rate between ViT and head (Run 2) gives no meaningful improvement over a single LR (Run 1), confirming that the bottleneck is the feature extractor, not the learning rate schedule.

## Runs

| Run | Slices | LR (vit) | LR (head) | Dropout | Batch/GPU | Eff Batch | Val AUC | Test AUC | Best Epoch |
|-----|--------|----------|-----------|---------|-----------|-----------|---------|----------|------------|
| 1 | 32 | 5e-5 | 5e-5 | 0.0 | — | 16 | 0.831 | N/A | 6 |
| 2 | 32 | 2e-5 | 1e-4 | 0.0 | — | 16 | 0.832 | N/A | 6 |

Test AUC was not evaluated for Phase 1 runs — these served as baselines to motivate full fine-tuning.

## Training Config

- **Hardware:** 4x NVIDIA T4 (16GB each)
- **Optimizer:** AdamW, weight decay 0.01
- **Scheduler:** Cosine LR with 3-epoch warmup
- **Loss:** BCEWithLogitsLoss
- **Early stopping:** on val AUC, patience=5
- **ConvNeXt weights:** frozen, from [SLIViT's checkpoint](https://drive.google.com/drive/folders/1f8P3g8ofBTWMFiuNS8vc01s98HyS7oRT)

## Analysis

### Why frozen features cap at ~0.83

The ConvNeXt was pretrained on the Kermany OCT dataset (84K images: CNV, DME, drusen, normal). While this teaches general retinal OCT features (layer boundaries, thickness patterns), glaucoma has its own specific patterns — RNFL thinning, optic nerve head changes — that the frozen extractor cannot learn. The ViT can only combine features that the ConvNeXt provides; it cannot invent new ones.

### LR splitting has no effect

Run 2 gave the head a 5x higher LR (1e-4) and the ViT a lower LR (2e-5), but the result was indistinguishable from the uniform LR in Run 1 (0.832 vs 0.831). This is expected: when the input features are fixed, there is limited room for a better ViT configuration to help.

![Phase 1 Training](../../../results/phase1_training.png)
