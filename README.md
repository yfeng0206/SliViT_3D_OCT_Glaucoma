# SLIViT for Glaucoma Classification on FairVision OCT Data

This repo documents our work reproducing the [SLIViT](https://github.com/cozygene/SLIViT) architecture and applying it to binary glaucoma classification on the [Harvard FairVision](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision) OCT dataset. We walk through the setup, the experiments we ran, what worked, what didn't, and what we learned along the way.

## Background

### How SLIViT works

SLIViT was designed to classify 3D medical volumes without needing a full 3D CNN. Instead, it slices the volume into 2D images and processes them in two stages:

**Stage 1 — ConvNeXt-Tiny (the feature extractor).** This is a standard 2D CNN. It looks at each OCT slice independently and pulls out spatial features — things like retinal layer boundaries, thickness patterns, and so on. The clever part is that it doesn't start from scratch. It's first pretrained on ImageNet (general image understanding), then further pretrained on the [Kermany OCT dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2) — 84,000 retinal OCT images labeled as CNV, DME, drusen, or normal. By the time it gets to our glaucoma task, it already understands what retinal layers look like in OCT.

**Stage 2 — ViT (the integrator).** This is a Vision Transformer that takes the per-slice features from the ConvNeXt and figures out how they relate across the volume. Think of it as: the ConvNeXt tells you what each slice looks like, and the ViT decides what the whole volume means. This part is randomly initialized — it has to learn from our training data.

**Stage 3 — Classification head.** Just `LayerNorm + Linear(256, 1)`. Outputs a single number: higher means more likely glaucoma.

### Why start with a frozen feature extractor?

The ConvNeXt already knows OCT pretty well from the Kermany pretraining. So we figured: why not freeze it and only train the ViT and head? This is cheaper, faster, and you don't risk messing up features that already work. Both tasks involve retinal OCT, so features like RNFL thickness and layer boundaries should carry over.

This turned out to cap around 0.83 AUC. The Kermany pretraining covers CNV, DME, drusen, and normal retinas — but glaucoma has its own patterns (RNFL thinning, optic nerve changes) that the frozen features can't quite capture. The ViT alone can't compensate for features that aren't there. When we unfroze the ConvNeXt and let it fine-tune (Phase 2), test AUC jumped to 0.87.

### The dataset

We used the Glaucoma subset of [Harvard FairVision](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision):
- 10,000 subjects total
- Each has a 200x200x200 OCT B-scan volume stored as `.npz`
- Binary labels: glaucoma (1) or not (0)
- Pre-split: 6,000 train / 1,000 validation / 3,000 test
- Also includes demographic info (race, gender, ethnicity) for fairness research, though we didn't use it here

The data is about 63GB compressed and available on [HuggingFace](https://huggingface.co/datasets/ming0100/Harvard_FairVision) (`dataset-004.zip`).

## Architecture details

```
OCT Volume (200x200x200)
  -> Sample N slices uniformly (we tested 32, 64, 100)
  -> Resize each to 256x256, convert grayscale to 3-channel
  -> Tile vertically into one tall image: 3 x (Nx256) x 256
  -> ConvNeXt-Tiny (ImageNet -> Kermany OCT pretrained)
  -> N feature maps of 768x64 each
  -> Linear projection: 49152-d -> 256-d per token
  -> ViT encoder (5 layers, 20 heads, dim_head=64, mlp_dim=512)
  -> CLS token -> LayerNorm -> Linear(256, 1) -> logit
```

Total parameters: 77.8M. Of those, 27.8M are in the ConvNeXt (frozen in Phase 1) and 50M in the ViT + projections + head.

One thing worth noting: since `vit_dim=256` and `heads=20`, and 256 doesn't divide evenly by 20, we need projection layers in each transformer block to go from 256 to 1280 (=20x64) and back. These projections account for ~33M of the 50M trainable params. Switching to 16 heads would eliminate them entirely and drop trainable params to around 15M — something worth trying.

## Results

### Phase 1: Frozen feature extractor, train ViT + head only

| Run | Slices | LR (vit / head) | Dropout | Eff Batch | Val AUC | Test AUC | Best Epoch |
|-----|--------|-----------------|---------|-----------|---------|----------|------------|
| 1 | 32 | 5e-5 / 5e-5 | 0.0 | 16 | 0.831 | — | 6 |
| 2 | 32 | 2e-5 / 1e-4 | 0.0 | 16 | 0.832 | — | 6 |

### Phase 2: Full fine-tuning, all parameters trainable

| Run | Slices | LR (fe / vit / head) | Dropout | Batch/GPU | Accum | Eff Batch | Val AUC | Test AUC | Best Epoch |
|-----|--------|----------------------|---------|-----------|-------|-----------|---------|----------|------------|
| 3 | 32 | 5e-6 / 1e-5 / 5e-5 | 0.10 | 2 | — | 8 | 0.846 | **0.869** | 4 |
| 4 | 64 | 1e-6 / 5e-6 / 5e-5 | 0.15 | 1 | — | 4 | 0.841 | 0.868 | 6 |
| 5 | 64 | 1e-6 / 5e-6 / 5e-5 | 0.15 | 2 | 2 | 16 | 0.845 | 0.866 | 9 |
| 6 | 32 | 1e-6 / 5e-6 / 5e-5 | 0.15 | 2 | 2 | 16 | 0.840 | 0.864 | 7 |

## What we tried and why

### Phase 1 experiments (Runs 1-2)

We started with the SLIViT paper's defaults and froze the ConvNeXt. Run 1 used a single learning rate (5e-5) for everything. It hit 0.831 val AUC and overfitted hard by epoch 7.

For Run 2, we split the learning rate — giving the head a higher rate (1e-4) since it's randomly initialized and needs to learn faster, and a lower rate for the ViT (2e-5). Made almost no difference: 0.832. The bottleneck wasn't the learning rate, it was the frozen features.

### Moving to full fine-tuning (Run 3)

The SLIViT paper actually trains everything end-to-end. We should have started here, but the frozen approach was a reasonable sanity check. For Run 3 we unfroze the ConvNeXt with a deliberately low LR (5e-6) — the idea being that the Kermany-pretrained weights are a good starting point and we don't want to blow them up. The ViT got 1e-5 and the head got 5e-5.

We also added dropout (0.1) since we were now training 78M parameters on just 6K images. The result was our best yet: 0.846 val / **0.869 test AUC**. The test set actually scored higher than validation, which makes sense — the val set is only 1,000 samples so it has more variance.

### Trying more slices and lowering LR further (Run 4)

32 slices samples every ~6th slice from the 200-slice volume. We tried 64 to give the model denser spatial coverage. But we also noticed that Run 3 overfitted by epoch 4 — the ViT with 50M params was learning too fast. So we made two changes at once:

- Dropped the ConvNeXt LR from 5e-6 to 1e-6. The pretrained backbone was changing too quickly and pulling the model toward memorization. At 1e-6 it barely moves — mostly preserving the Kermany features while allowing subtle adaptation.
- Dropped the ViT LR from 1e-5 to 5e-6 for the same reason. The ViT has the most parameters and was the main driver of overfitting. Slowing it down gave us 2 extra epochs before the val loss started climbing (best epoch moved from 4 to 6).
- Bumped dropout from 0.1 to 0.15 as additional regularization.

We also had to drop batch size from 2 to 1 per GPU because 64-slice images are twice as tall (16384 vs 8192 pixels) and take more VRAM. This cut our effective batch from 8 to 4.

Result: 0.868 test AUC — essentially the same as 32 slices. The lower LRs did help with overfitting (6 useful epochs instead of 4), but the comparison between 32 and 64 slices isn't clean because we changed too many things at once: slice count, LR, dropout, and batch size all differ.

### Fixing the batch size confound (Runs 5-6)

Looking at the GPU memory monitor, we noticed each T4 was only using about 8GB out of 16GB with 64 slices at bs=1. So we could go to bs=2 per GPU without hitting the memory limit.

But we also wanted to push the effective batch to 16 (up from 8 in Run 3 and 4 in Run 4) for smoother gradients. Rather than trying bs=4 and risking OOM, we added gradient accumulation — the model does 2 forward passes, accumulates the gradients, then does one weight update. Mathematically identical to a batch twice the size, but uses the same memory as the smaller batch.

Runs 5 and 6 both use bs=2 per GPU with 2 accumulation steps, giving an effective batch of 16 (2 x 4 GPUs x 2 accum). Same LR, same dropout, same everything — the only difference is 32 vs 64 slices.

Results: 64 slices got 0.866 test AUC, 32 slices got 0.864. A 0.002 difference — basically noise. **32 slices is enough.** Doubling the spatial coverage doesn't help because the ConvNeXt + ViT architecture already captures the relevant retinal structures from 32 evenly-spaced slices. The extra slices just add redundant information and cost 2x more compute.

### The best run was actually our first Phase 2 attempt

Run 3 (test AUC 0.869) used slightly higher learning rates (5e-6/1e-5 vs 1e-6/5e-6) and less dropout (0.1 vs 0.15) compared to the later runs. The lower LRs in Runs 4-6 gave more stable training — the model trained for more epochs before overfitting (best epoch 7-9 vs 4). But the peak performance was actually slightly worse. It seems like the higher LR lets the model reach a better optimum faster, even if it can't hold it for long. With early stopping catching the peak, that's fine.

### The overfitting problem

This has been the main issue throughout. Every single run follows the same trajectory: training loss drops nicely, validation loss starts climbing after epoch 4-6, and the model starts memorizing. Best checkpoints always come from very early epochs.

The numbers tell the story — by the end of training, train loss is around 0.04 while val loss is above 1.0. The model gets very confident on training data but those confident predictions don't generalize.

We tried several things:
- **Dropout** (0.0 -> 0.10 -> 0.15): helped a little, maybe 1-2 extra epochs before overfitting
- **Lower learning rates**: slowed things down but didn't change the outcome
- **Per-component LRs**: useful for controlling how fast each part adapts, but didn't solve overfitting

The root cause is probably the parameter count. 50M trainable parameters learning from 6,000 images is asking for trouble. The projection layers alone (needed because 256 doesn't divide by 20 heads) eat up 33M params. Reducing to 16 heads would cut that dramatically and is probably the single most impactful change we haven't tried yet.

## Training setup

**Hardware:** 4x NVIDIA T4 (16GB each)

**How long things take:**
- 32 slices at bs=2/GPU: about 5 minutes per epoch. A full run with early stopping takes around an hour.
- 64 slices at bs=1/GPU: about 24 minutes per epoch. Full runs take 4-5 hours.
- Downloading the dataset to the compute node adds ~10 minutes at the start of each job.

**Stack:** PyTorch 1.13.1, CUDA 11.7, HuggingFace Transformers for the ConvNeXt backbone. Training uses 4-GPU DDP with mixed precision (fp16).

**Training config:**
- AdamW optimizer, weight decay 0.01
- Cosine LR schedule with 3-epoch warmup
- BCEWithLogitsLoss
- Early stopping on val AUC, patience=5
- ConvNeXt initialized from [SLIViT's published checkpoint](https://drive.google.com/drive/folders/1f8P3g8ofBTWMFiuNS8vc01s98HyS7oRT)

## Project structure

```
src/
  model.py           SLIViT model (ConvNeXt + ViT + head)
  dataset.py         Loads FairVision .npz files, samples and tiles slices
  train.py           Training loop with DDP, gradient accumulation, test eval
  eval_test.py       Standalone test set evaluation from a saved checkpoint
  setup_data.py      Downloads data from cloud storage to compute node
  upload_results.py  Pushes results back to cloud storage after training
  run.sh             Entry point for training jobs
  run_eval.sh        Entry point for evaluation jobs

configs/
  phase1_32.yml      Phase 1, 32 slices
  phase1_64.yml      Phase 1, 64 slices
  phase2_32.yml      Phase 2, 32 slices
  phase2_64.yml      Phase 2, 64 slices
  phase2_128.yml     Phase 2, 100 slices
  environment.yml    Conda environment

scripts/
  download_hf.py     Downloads the dataset from HuggingFace
```

## What's next

Things we haven't tried yet that could push past the 0.87 ceiling:

- **16 attention heads instead of 20** — eliminates the projection layers, drops trainable params from 50M to ~15M. This is probably the single most impactful change for overfitting.
- **Data augmentation** — we used zero augmentation in all runs. Random flips, intensity jitter, and random slice offsets would effectively multiply the training data.
- **Label smoothing** — soft targets (0.05/0.95 instead of 0/1) to prevent overconfidence.
- **Fairness analysis** — the dataset includes demographic metadata. Evaluating per-group AUC would be valuable for clinical relevance.

## References

- Avram et al., "SLIViT: a general AI framework for clinical-feature diagnosis from limited 3D biomedical-imaging data" ([paper](https://pubmed.ncbi.nlm.nih.gov/38045283/), [code](https://github.com/cozygene/SLIViT))
- Luo et al., "Harvard Ophthalmology AI-Lab FairVision Dataset" ([paper](https://arxiv.org/abs/2310.02492), [code](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision))
- Liu et al., "A ConvNet for the 2020s" ([paper](https://arxiv.org/abs/2201.03545))

## Sample

32 uniformly sampled B-scans from one OCT volume (non-glaucoma):

![OCT Slices](sample_slices.png)
