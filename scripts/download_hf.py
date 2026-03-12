"""
Download FairVision Glaucoma (dataset-004.zip) from HuggingFace,
extract only .npz files, then upload to AML datastore.

Usage:
    python scripts/download_hf.py --local_dir ./data/glaucoma
    python scripts/download_hf.py --local_dir ./data/glaucoma --upload
"""

import argparse
import os
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default="./data/glaucoma")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--keep_zip", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    zip_path = os.path.join(args.local_dir, "dataset-004.zip")

    # Step 1: Download from HuggingFace
    if not os.path.exists(zip_path):
        print("=== Downloading dataset-004.zip (Glaucoma, ~63GB) from HuggingFace ===")
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="ming0100/Harvard_FairVision",
            filename="dataset-004.zip",
            repo_type="dataset",
            local_dir=args.local_dir,
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may put it in a subfolder; move if needed
        if downloaded != zip_path and os.path.exists(downloaded):
            os.rename(downloaded, zip_path)
        print(f"Downloaded to {zip_path}")
    else:
        print(f"Zip already exists: {zip_path}")

    # Step 2: Extract only .npz files
    extract_dir = os.path.join(args.local_dir, "extracted")
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        print("=== Extracting .npz files only ===")
        with zipfile.ZipFile(zip_path, "r") as zf:
            npz_files = [
                f for f in zf.namelist()
                if f.endswith(".npz") and not f.startswith("__MACOSX")
            ]
            print(f"  Found {len(npz_files)} .npz files")
            for i, name in enumerate(npz_files):
                zf.extract(name, extract_dir)
                if (i + 1) % 500 == 0:
                    print(f"  Extracted {i + 1}/{len(npz_files)}")
        print("  Extraction complete")
    else:
        print(f"Already extracted: {extract_dir}")

    # Step 3: Show structure
    for split in ["Training", "Validation", "Test"]:
        split_dir = os.path.join(extract_dir, split)
        if os.path.exists(split_dir):
            n = sum(1 for f in os.listdir(split_dir) if f.endswith(".npz"))
            print(f"  {split}: {n} .npz files")
        else:
            print(f"  {split}: directory not found")

    # Step 4: Download metadata CSV
    csv_path = os.path.join(args.local_dir, "data_summary_glaucoma.csv")
    if not os.path.exists(csv_path):
        print("=== Downloading metadata CSV ===")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="ming0100/Harvard_FairVision",
            filename="Harvard FairVision (Harvard-FairVision)-20251128T054551Z-1-002.zip",
            repo_type="dataset",
            local_dir=args.local_dir,
            local_dir_use_symlinks=False,
        )
        # Extract CSV from the small metadata zip
        meta_zip = os.path.join(
            args.local_dir,
            "Harvard FairVision (Harvard-FairVision)-20251128T054551Z-1-002.zip",
        )
        if os.path.exists(meta_zip):
            with zipfile.ZipFile(meta_zip, "r") as zf:
                for name in zf.namelist():
                    if "glaucoma" in name.lower() and name.endswith(".csv"):
                        zf.extract(name, args.local_dir)
                        print(f"  Extracted: {name}")

    # Step 5: Upload to AML datastore
    if args.upload:
        print("=== Uploading to AML datastore ===")
        cmd = [
            "az", "ml", "data", "create",
            "--name", "fairvision-glaucoma",
            "--version", "1",
            "--type", "uri_folder",
            "--path", extract_dir,
            "--description",
            "FairVision Glaucoma OCT dataset - 10k subjects, .npz files only",
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  Upload complete!")
            print(result.stdout[:500])
        else:
            print(f"  Error: {result.stderr[:500]}")

    # Cleanup
    if not args.keep_zip and os.path.exists(zip_path):
        print(f"Removing {zip_path}...")
        os.remove(zip_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
