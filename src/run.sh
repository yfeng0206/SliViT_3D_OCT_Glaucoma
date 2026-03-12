#!/bin/bash

PHASE=${PHASE:-1}
NUM_SLICES=${NUM_SLICES:-32}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-4}
LR_FE=${LR_FE:-5e-6}
LR_VIT=${LR_VIT:-2e-5}
LR_HEAD=${LR_HEAD:-1e-4}
PATIENCE=${PATIENCE:-5}
ACCUM_STEPS=${ACCUM_STEPS:-1}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="p${PHASE}_slices${NUM_SLICES}_lrv${LR_VIT}_lrh${LR_HEAD}_bs${BATCH_SIZE}_ep${EPOCHS}"
OUTPUT_DIR="/tmp/fairvision_outputs/${RUN_TAG}_${TIMESTAMP}"
BLOB_PREFIX="training-results/${RUN_TAG}_${TIMESTAMP}"
DATA_DIR="/tmp/fairvision_data"
mkdir -p $OUTPUT_DIR

echo "=== Disk space ==="
df -h /tmp 2>/dev/null || df -h
rm -rf /home/azureuser/fairvision_data /home/azureuser/fairvision_outputs 2>/dev/null || true

echo "=== Environment Info ==="
python --version
pip show torch 2>/dev/null | grep -E "^Name|^Version"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo "=== Installing dependencies ==="
pip install transformers scikit-learn pillow azure-storage-blob azure-identity 2>&1 | tail -5

echo "=== Downloading data ==="
python setup_data.py --output_dir $DATA_DIR

echo "=== Starting training ==="
echo "Phase: $PHASE"
echo "Config: slices=$NUM_SLICES, epochs=$EPOCHS, bs=$BATCH_SIZE, patience=$PATIENCE"
echo "LR: fe=$LR_FE, vit=$LR_VIT, head=$LR_HEAD"
echo "Output: $OUTPUT_DIR"
echo "Blob:   $BLOB_PREFIX"

TRAIN_EXIT=0
torchrun \
  --nproc_per_node=4 \
  train.py \
  --data_dir $DATA_DIR/data \
  --output_dir $OUTPUT_DIR \
  --fe_checkpoint $DATA_DIR/feature_extractor.pth \
  --phase $PHASE \
  --batch_size $BATCH_SIZE \
  --lr_fe $LR_FE \
  --lr_vit $LR_VIT \
  --lr_head $LR_HEAD \
  --epochs $EPOCHS \
  --num_workers 2 \
  --seed 42 \
  --num_slices $NUM_SLICES \
  --patience $PATIENCE \
  --accum_steps $ACCUM_STEPS 2>&1 | tee $OUTPUT_DIR/torchrun_stdout.log || TRAIN_EXIT=$?

echo "=== Training exit code: $TRAIN_EXIT ==="
echo "=== Output files ==="
ls -la $OUTPUT_DIR/

echo "=== Uploading results to blob storage ==="
python upload_results.py --output_dir "$OUTPUT_DIR" --blob_prefix "$BLOB_PREFIX"

echo "=== All done (train exit=$TRAIN_EXIT) ==="
exit $TRAIN_EXIT
