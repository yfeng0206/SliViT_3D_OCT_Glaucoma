#!/bin/bash

EVAL_BLOB_PREFIX=${EVAL_BLOB_PREFIX:-"training-results/p2_slices32_lrv1e-5_lrh5e-5_bs2_ep25_20260311_040713"}
NUM_SLICES=${NUM_SLICES:-32}
PHASE=${PHASE:-2}
DATA_DIR="/tmp/fairvision_data"
EVAL_DIR="/tmp/fairvision_eval"
mkdir -p $EVAL_DIR

echo "=== Installing dependencies ==="
pip install transformers scikit-learn pillow azure-storage-blob azure-identity 2>&1 | tail -5

echo "=== Downloading data ==="
python setup_data.py --output_dir $DATA_DIR

echo "=== Downloading checkpoint ==="
python -c "
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobClient
import os

cred = ManagedIdentityCredential()
blob_name = '${EVAL_BLOB_PREFIX}/best_model.pt'
print('Downloading: %s' % blob_name)
blob = BlobClient(
    account_url='https://YOUR_STORAGE_ACCOUNT.blob.core.windows.net',
    container_name='YOUR_CONTAINER_NAME',
    blob_name=blob_name,
    credential=cred,
)
with open('$EVAL_DIR/best_model.pt', 'wb') as f:
    f.write(blob.download_blob().readall())
print('Downloaded: %d bytes' % os.path.getsize('$EVAL_DIR/best_model.pt'))
"

echo "=== Running test evaluation ==="
python eval_test.py \
  --data_dir $DATA_DIR/data \
  --checkpoint $EVAL_DIR/best_model.pt \
  --num_slices $NUM_SLICES \
  --phase $PHASE \
  --batch_size 4 \
  --output_file $EVAL_DIR/test_results.json

echo "=== Uploading results ==="
python upload_results.py --output_dir $EVAL_DIR --blob_prefix "eval-results/p${PHASE}_slices${NUM_SLICES}"

echo "=== Done ==="
