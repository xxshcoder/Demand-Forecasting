#!/usr/bin/env bash
# One attempt at downloading the real Kaggle dataset. On any failure the
# caller must use scripts/make_synthetic_data.py and record that decision in
# docs/DECISIONS.md. Never put Kaggle credentials in this repository.
set -euo pipefail

RAW_DIR="${1:-data/raw}"
COMPETITION="demand-forecasting-kernels-only"

mkdir -p "$RAW_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install with: python -m pip install kaggle" >&2
  echo "Place kaggle.json in ~/.kaggle/ (chmod 600) or set KAGGLE_USERNAME/KAGGLE_KEY." >&2
  exit 1
fi

if ! kaggle competitions download -c "$COMPETITION" -p "$RAW_DIR"; then
  echo "Download failed. Check credentials and competition rules at" >&2
  echo "  https://www.kaggle.com/competitions/$COMPETITION" >&2
  echo "Falling back to synthetic data is documented in docs/DECISIONS.md." >&2
  exit 1
fi

RAW_DIR="$RAW_DIR" python - <<'PY'
import os
import zipfile
from pathlib import Path

raw = Path(os.environ["RAW_DIR"])
for archive in raw.glob("*.zip"):
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if member.endswith("train.csv"):
                zf.extract(member, raw)
                target = raw / member
                target.rename(raw / "train.csv")
                print(f"Extracted {target} -> {raw / 'train.csv'}")
PY

if [[ ! -f "$RAW_DIR/train.csv" ]]; then
  echo "Download finished but $RAW_DIR/train.csv is missing." >&2
  exit 1
fi
echo "Real dataset ready at $RAW_DIR/train.csv"
