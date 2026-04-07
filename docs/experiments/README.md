# SLIViT Experiments

All experiment runs for SLIViT glaucoma classification on FairVision OCT data.

## Summary

| Run | Phase | Slices | LR config | Dropout | Eff Batch | Val AUC | Test AUC | Best Epoch |
|-----|-------|--------|-----------|---------|-----------|---------|----------|------------|
| **3** | **2** | **32** | **5e-6 / 1e-5 / 5e-5** | **0.10** | **8** | **0.846** | **0.869** | **4** |
| 4 | 2 | 64 | 1e-6 / 5e-6 / 5e-5 | 0.15 | 4 | 0.841 | 0.868 | 6 |
| 5 | 2 | 64 | 1e-6 / 5e-6 / 5e-5 | 0.15 | 16 | 0.845 | 0.866 | 9 |
| 6 | 2 | 32 | 1e-6 / 5e-6 / 5e-5 | 0.15 | 16 | 0.840 | 0.864 | 7 |
| 2 | 1 | 32 | 2e-5 / 1e-4 (vit/head) | 0.0 | 16 | 0.832 | N/A | 6 |
| 1 | 1 | 32 | 5e-5 / 5e-5 (vit/head) | 0.0 | 16 | 0.831 | N/A | 6 |

## Phases

### [Phase 1: Frozen Feature Extractor](phase1/)

ConvNeXt frozen (Kermany-pretrained), train ViT + classification head only. 2 runs exploring learning rate schedules. Capped at ~0.83 val AUC, confirming that frozen features are the bottleneck.

### [Phase 2: Full Fine-Tuning](phase2/)

All parameters trainable with per-component learning rates (ConvNeXt < ViT < head). 4 runs exploring slice count, learning rates, dropout, and batch size. Best result: 0.869 test AUC (Run 3).
