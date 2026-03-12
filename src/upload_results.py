"""Upload training results to Azure Blob Storage."""
import argparse
import glob
import os
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--blob_prefix", required=True)
    args = parser.parse_args()

    print("Uploading from: %s" % args.output_dir)
    print("Uploading to:   %s/" % args.blob_prefix)

    files = glob.glob(os.path.join(args.output_dir, "*"))
    files = [f for f in files if os.path.isfile(f)]
    print("Found %d files" % len(files))

    if not files:
        print("Nothing to upload!")
        return

    try:
        from azure.identity import ManagedIdentityCredential
        from azure.storage.blob import ContainerClient

        cred = ManagedIdentityCredential()
        container = ContainerClient(
            account_url="https://YOUR_STORAGE_ACCOUNT.blob.core.windows.net",
            container_name="YOUR_CONTAINER_NAME",
            credential=cred,
        )

        for fpath in files:
            fname = os.path.basename(fpath)
            blob_name = "%s/%s" % (args.blob_prefix, fname)
            size = os.path.getsize(fpath)
            print("  %s (%s bytes) -> %s" % (fname, format(size, ","), blob_name))
            with open(fpath, "rb") as f:
                container.upload_blob(blob_name, f, overwrite=True)

        print("Upload complete!")
    except Exception as e:
        print("Upload FAILED: %s" % e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
