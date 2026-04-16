"""
Model downloader — runs at Docker build time (and optionally at runtime).
Downloads weights from public URLs or Cloudflare R2.

Usage:
    python3 download_models.py              # download all
    python3 download_models.py --check      # print status only
    python3 download_models.py yolov8m.pt   # download specific model
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# ─── Model registry ────────────────────────────────────────────────────────
# fmt: off
MODELS: dict[str, dict] = {
    # Public Ultralytics weights
    "yolov8m.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt",
        "md5": None,   # skip hash check — official release
    },
    "yolov8m-pose.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-pose.pt",
        "md5": None,
    },
    "yolov8n.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt",
        "md5": None,
    },
    "yolov8n-pose.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt",
        "md5": None,
    },
    "yolov8s.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt",
        "md5": None,
    },

    # ── Custom models — upload to Cloudflare R2 and set URL ──────────────
    # Uncomment and fill in after uploading to R2:
    # "jersey_classifier_aigis.pt": {
    #     "url": "https://r2.iceiq.app/models/jersey_classifier_aigis.pt",
    #     "md5": None,
    # },
    # "jersey_classifier_hockey_machine.pt": {
    #     "url": "https://r2.iceiq.app/models/jersey_classifier_hockey_machine.pt",
    #     "md5": None,
    # },
    # "team_classifier_cnn.pt": {
    #     "url": "https://r2.iceiq.app/models/team_classifier_cnn.pt",
    #     "md5": None,
    # },
}
# fmt: on

# Root of this script = /app inside Docker, or repo root locally
REPO_ROOT = Path(__file__).parent
MODELS_DIR = REPO_ROOT / "models"


# ─── Helpers ────────────────────────────────────────────────────────────────

def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def _progress(count: int, block: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(count * block / total * 100, 100)
    bar = "#" * int(pct / 5)
    print(f"\r  [{bar:<20}] {pct:5.1f}%", end="", flush=True)


def download(name: str, url: str, dest: Path, expected_md5: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    print(f"  Downloading {name}")
    print(f"  → {url}")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        print()  # newline after progress bar
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {name}: {e}") from e

    if expected_md5:
        actual = md5sum(tmp)
        if actual != expected_md5:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"MD5 mismatch for {name}: expected {expected_md5}, got {actual}"
            )

    tmp.rename(dest)
    size_mb = dest.stat().st_size / 1_048_576
    print(f"  ✓ {name} ({size_mb:.1f} MB)")


def needs_download(dest: Path, expected_md5: str | None) -> bool:
    if not dest.exists():
        return True
    if expected_md5 and md5sum(dest) != expected_md5:
        print(f"  ⚠ {dest.name}: MD5 mismatch — re-downloading")
        return True
    return False


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    check_only = "--check" in sys.argv
    targets = [a for a in sys.argv[1:] if not a.startswith("--")]

    registry = {k: v for k, v in MODELS.items() if not targets or k in targets}

    if not registry:
        print(f"No models matched: {targets}")
        sys.exit(1)

    print(f"\n=== IceIQ Model Downloader ({'check' if check_only else 'download'}) ===")
    print(f"Models dir: {MODELS_DIR}\n")

    errors: list[str] = []
    for name, spec in registry.items():
        dest = MODELS_DIR / name
        if needs_download(dest, spec.get("md5")):
            if check_only:
                print(f"  MISSING  {name}")
            else:
                try:
                    download(name, spec["url"], dest, spec.get("md5"))
                except RuntimeError as e:
                    print(f"  ERROR: {e}")
                    errors.append(name)
        else:
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  OK       {name} ({size_mb:.1f} MB)")

    print()
    if errors:
        print(f"Failed: {', '.join(errors)}")
        sys.exit(1)
    print("All models ready.")


if __name__ == "__main__":
    main()
