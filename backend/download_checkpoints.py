"""
Model Checkpoint Downloader & Integrity Verifier for Production Deployment.

Ensures that authentic PyTorch binary checkpoints are present at runtime/build-time
rather than Git LFS pointer text files ("version https://git-lfs.github.com/spec/v1").
If Git LFS pointer files are detected or checkpoints are missing, downloads genuine
binary checkpoints from GitHub Release assets and verifies their PyTorch structure.
"""

import os
import sys
import argparse
from pathlib import Path
import urllib.request
import urllib.error

# Default GitHub Release asset URLs for AI-Brain-Tumor-Analysis (v1.0.0)
DEFAULT_RELEASE_TAG = "v1.0.0"
DEFAULT_REPO = "bhav-vaidya-23/AI-Brain-Tumor-Analysis"
DEFAULT_CLASSIFIER_URL = (
    f"https://github.com/{DEFAULT_REPO}/releases/download/{DEFAULT_RELEASE_TAG}/best_classifier.pth"
)
DEFAULT_UNET_URL = (
    f"https://github.com/{DEFAULT_REPO}/releases/download/{DEFAULT_RELEASE_TAG}/best_unet.pth"
)

# Minimum acceptable binary file sizes (bytes)
MIN_CLASSIFIER_BYTES = 50 * 1024 * 1024    # ~106.8 MB expected
MIN_UNET_BYTES = 200 * 1024 * 1024         # ~355.3 MB expected


def is_lfs_pointer(file_path: Path) -> bool:
    """Returns True if the file is a Git LFS pointer text stub."""
    if not file_path.exists():
        return False
    try:
        if file_path.stat().st_size > 1_000_000:  # LFS pointers are tiny (<1 KB)
            return False
        with open(file_path, "rb") as f:
            header = f.read(50)
            return header.startswith(b"version https://git-lfs") or header.startswith(b"version https://")
    except Exception:
        return False


def verify_checkpoint(file_path: Path, min_bytes: int, model_name: str) -> tuple[bool, str]:
    """
    Validates that a file is a genuine, uncorrupted PyTorch checkpoint.
    Returns (is_valid, message).
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    file_size = file_path.stat().st_size
    if is_lfs_pointer(file_path):
        return False, (
            f"{file_path.name} is a Git LFS pointer text stub "
            f"(size: {file_size} bytes), NOT a binary model checkpoint."
        )

    if file_size < min_bytes:
        return False, (
            f"{file_path.name} is too small ({file_size / (1024*1024):.2f} MB; "
            f"expected >= {min_bytes / (1024*1024):.1f} MB). Likely incomplete or corrupted."
        )

    # Check archive magic bytes (PyTorch checkpoints are zip archives or pickled streams)
    with open(file_path, "rb") as f:
        header = f.read(16)
        # Zip archive magic: PK\x03\x04
        # Pickle magic: \x80\x02, \x80\x03, \x80\x04
        is_zip = header.startswith(b"PK\x03\x04")
        is_pickle = header.startswith(b"\x80")
        if not (is_zip or is_pickle):
            return False, (
                f"{file_path.name} does not match PyTorch checkpoint signature "
                f"(first bytes: {header[:8]!r})."
            )

    # Verify PyTorch state_dict structure
    try:
        import torch
        ckpt = torch.load(file_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            return False, f"{file_path.name} loaded content is type {type(ckpt)}, expected dict."
        if "model_state_dict" not in ckpt:
            return False, f"{file_path.name} missing 'model_state_dict' key (keys: {list(ckpt.keys())})."
    except Exception as e:
        return False, f"Failed to load {file_path.name} with torch.load: {e}"

    return True, f"Verified valid PyTorch checkpoint ({file_size / (1024*1024):.2f} MB)"


def download_asset(url: str, dest_path: Path):
    """Downloads a file from a URL with progress logging and atomic rename."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    print(f"[Checkpoints] Initiating download: {url}", flush=True)
    print(f"[Checkpoints] Target destination: {dest_path}", flush=True)

    headers = {
        "User-Agent": "BrainTumorAI-Deployment/1.0",
        "Accept": "application/octet-stream"
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
            total_size = response.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size else None
            downloaded = 0
            chunk_size = 4 * 1024 * 1024  # 4 MB chunks

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = (downloaded / total_bytes) * 100
                    print(
                        f"  Downloaded {downloaded / (1024*1024):.1f} / "
                        f"{total_bytes / (1024*1024):.1f} MB ({pct:.1f}%)...",
                        flush=True
                    )
                else:
                    print(f"  Downloaded {downloaded / (1024*1024):.1f} MB...", flush=True)

        # Replace target file atomically
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
        print(f"[Checkpoints] Successfully saved to {dest_path}", flush=True)

    except urllib.error.HTTPError as err:
        if temp_path.exists():
            temp_path.unlink()
        print(f"\n[Checkpoints ERROR] HTTP {err.code}: {err.reason} downloading from:", file=sys.stderr)
        print(f"  {url}\n", file=sys.stderr)
        print("POSSIBLE CAUSE & REMEDY:", file=sys.stderr)
        print(f"  1. The GitHub Release asset for '{dest_path.name}' may not be published yet.", file=sys.stderr)
        print(f"  2. Create a GitHub Release with tag '{DEFAULT_RELEASE_TAG}' on repository '{DEFAULT_REPO}'.", file=sys.stderr)
        print(f"  3. Upload 'outputs/{dest_path.name}' as a Release binary asset.", file=sys.stderr)
        print("  4. See docs/DEPLOYMENT.md for step-by-step guidance.\n", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        print(f"\n[Checkpoints ERROR] Download failed for {url}: {exc}", file=sys.stderr)
        raise SystemExit(1)


def ensure_checkpoint(file_path: Path, url: str, min_bytes: int, model_name: str, force: bool = False):
    """Verifies existing checkpoint, downloading if missing or Git LFS pointer text."""
    if file_path.exists() and not force:
        valid, msg = verify_checkpoint(file_path, min_bytes, model_name)
        if valid:
            print(f"[Checkpoints] Existing {model_name} is valid: {msg}", flush=True)
            return
        else:
            print(f"[Checkpoints] Existing file invalid ({msg}). Downloading authentic binary...", flush=True)
    else:
        print(f"[Checkpoints] File not found at {file_path}. Downloading from release...", flush=True)

    download_asset(url, file_path)

    # Post-download verification
    valid, msg = verify_checkpoint(file_path, min_bytes, model_name)
    if not valid:
        print(f"[Checkpoints ERROR] Post-download verification failed: {msg}", file=sys.stderr)
        if file_path.exists():
            file_path.unlink()
        raise SystemExit(1)
    print(f"[Checkpoints SUCCESS] {model_name} ready: {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Ensure real binary checkpoints for production deployment.")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("/app/outputs"),
        help="Target directory where checkpoints must reside (default: /app/outputs)"
    )
    parser.add_argument(
        "--classifier-url",
        type=str,
        default=os.getenv("CLASSIFIER_URL", DEFAULT_CLASSIFIER_URL),
        help="Download URL for best_classifier.pth"
    )
    parser.add_argument(
        "--unet-url",
        type=str,
        default=os.getenv("UNET_URL", DEFAULT_UNET_URL),
        help="Download URL for best_unet.pth"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify checkpoints in destination directory without downloading."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if a valid checkpoint file exists."
    )
    args = parser.parse_args()

    dest_dir = args.dest_dir.resolve()
    classifier_path = dest_dir / "best_classifier.pth"
    unet_path = dest_dir / "best_unet.pth"

    print("======================================================================", flush=True)
    print(f"[Checkpoints] Destination Directory: {dest_dir}", flush=True)
    print(f"[Checkpoints] Classifier URL:        {args.classifier_url}", flush=True)
    print(f"[Checkpoints] U-Net URL:             {args.unet_url}", flush=True)
    print("======================================================================", flush=True)

    if args.verify_only:
        c_valid, c_msg = verify_checkpoint(classifier_path, MIN_CLASSIFIER_BYTES, "Classifier")
        u_valid, u_msg = verify_checkpoint(unet_path, MIN_UNET_BYTES, "U-Net")
        print(f"Classifier: {'[OK]' if c_valid else '[FAIL]'} {c_msg}")
        print(f"U-Net:      {'[OK]' if u_valid else '[FAIL]'} {u_msg}")
        if not (c_valid and u_valid):
            sys.exit(1)
        print("[Checkpoints] All checkpoints verified successfully.")
        return

    # Process Classifier
    ensure_checkpoint(
        file_path=classifier_path,
        url=args.classifier_url,
        min_bytes=MIN_CLASSIFIER_BYTES,
        model_name="ResNet-18 Classifier",
        force=args.force
    )

    # Process U-Net
    ensure_checkpoint(
        file_path=unet_path,
        url=args.unet_url,
        min_bytes=MIN_UNET_BYTES,
        model_name="U-Net Segmentation",
        force=args.force
    )

    print("======================================================================", flush=True)
    print("[Checkpoints] Both checkpoints verified and ready for production inference.", flush=True)
    print("======================================================================", flush=True)


if __name__ == "__main__":
    main()
