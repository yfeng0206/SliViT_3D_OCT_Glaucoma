"""
Download data from Azure Blob Storage to local disk on the compute instance.
Uses DefaultAzureCredential (managed identity on compute).
"""
import argparse
import os


def download_blobs(account, container_name, prefix, local_dir, only_npz=True):
    """Download blobs using azure-storage-blob SDK."""
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import ContainerClient

    credential = DefaultAzureCredential()
    container = ContainerClient(
        account_url=f"https://{account}.blob.core.windows.net",
        container_name=container_name,
        credential=credential,
    )

    blobs = list(container.list_blobs(name_starts_with=prefix))
    print(f"  Found {len(blobs)} blobs with prefix '{prefix}'")

    downloaded = 0
    for blob in blobs:
        if only_npz and not blob.name.endswith(".npz"):
            continue
        # blob.name = "fhl-test-data/Training/data_00001.npz"
        rel_path = blob.name[len(prefix):].lstrip("/")
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(container.download_blob(blob).readall())
        downloaded += 1
        if downloaded % 500 == 0:
            print(f"  Downloaded {downloaded} files...")

    print(f"  Downloaded {downloaded} files total")


def download_single_blob(account, container_name, blob_name, local_path):
    """Download a single blob."""
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobClient

    credential = DefaultAzureCredential()
    blob = BlobClient(
        account_url=f"https://{account}.blob.core.windows.net",
        container_name=container_name,
        blob_name=blob_name,
        credential=credential,
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(blob.download_blob().readall())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", default="YOUR_STORAGE_ACCOUNT")
    parser.add_argument("--container", default="YOUR_CONTAINER_NAME")
    parser.add_argument("--data_prefix", default="fhl-test-data")
    parser.add_argument("--checkpoint_blob", default="checkpoints/feature_extractor.pth")
    parser.add_argument("--output_dir", default="/tmp/fairvision")
    args = parser.parse_args()

    data_dir = os.path.join(args.output_dir, "data")
    ckpt_path = os.path.join(args.output_dir, "feature_extractor.pth")

    os.makedirs(data_dir, exist_ok=True)

    # Download dataset
    if not os.path.exists(os.path.join(data_dir, "Training")):
        print("Downloading dataset from blob storage...")
        download_blobs(args.account, args.container, args.data_prefix, data_dir)
    else:
        print("Data already exists, skipping download.")

    # Download checkpoint
    if not os.path.exists(ckpt_path):
        print("Downloading checkpoint...")
        download_single_blob(args.account, args.container, args.checkpoint_blob, ckpt_path)
        print(f"  Saved to {ckpt_path}")

    # Print summary
    for split in ["Training", "Validation", "Test"]:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            n = len([f for f in os.listdir(split_dir) if f.endswith(".npz")])
            print(f"  {split}: {n} files")

    print(f"Checkpoint: {ckpt_path}")
    print(f"Data dir: {data_dir}")


if __name__ == "__main__":
    main()
