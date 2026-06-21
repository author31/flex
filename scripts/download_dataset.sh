#!/usr/bin/env bash
#
# Download the "High-Resolution Anime Face Dataset (512x512)" from Kaggle.
#
# Requires a Kaggle account. The public dataset endpoint works without auth for
# this dataset, but if you hit a 403 export KAGGLE_USERNAME / KAGGLE_KEY (from
# https://www.kaggle.com/settings -> "Create New Token") and the curl below will
# pick them up via --user.
#
# Usage:
#   ./download_dataset.sh [OUT_DIR]
#
#   OUT_DIR  destination directory (default: ./data/anime-faces)
#
set -euo pipefail

DATASET="subinium/highresolution-anime-face-dataset-512x512"
OUT_DIR="${1:-./data/anime-faces}"
ZIP_PATH="${OUT_DIR}/dataset.zip"

mkdir -p "${OUT_DIR}"

echo "==> Downloading ${DATASET}"
AUTH=()
if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
  AUTH=(--user "${KAGGLE_USERNAME}:${KAGGLE_KEY}")
  echo "    using KAGGLE_USERNAME credentials"
fi

curl -L "${AUTH[@]}" -o "${ZIP_PATH}" \
  "https://www.kaggle.com/api/v1/datasets/download/${DATASET}"

echo "==> Unzipping into ${OUT_DIR}"
unzip -q -o "${ZIP_PATH}" -d "${OUT_DIR}"
rm -f "${ZIP_PATH}"

echo "==> Done. Images under: ${OUT_DIR}"
find "${OUT_DIR}" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l \
  | xargs echo "    image count:"
