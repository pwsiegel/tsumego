#!/bin/bash
# Vertex AI Custom Job entrypoint.
#
# Usage (passed as containerSpec.args in the job YAML):
#     entrypoint.sh boards [extra train args...]
#     entrypoint.sh stones [extra train args...]
#
# Layout: GOAPP_DATA_DIR points at local container disk (so derived YOLO
# datasets and ultralytics run artifacts stay off the slow FUSE mount).
# Only the input synth_pages dir is symlinked into the bucket, and the
# final best.pt is copied directly into the bucket via --model-out.

set -euo pipefail

MODEL_TYPE="${1:-}"
shift || true

BUCKET_DATA="/gcs/tsumego-pwsiegel-data/data"
LOCAL_DATA="/tmp/goapp-data"
LOCAL_MODELS="/tmp/goapp-models"

mkdir -p "$LOCAL_DATA" "$LOCAL_MODELS"
ln -sfn "$BUCKET_DATA/synth_pages" "$LOCAL_DATA/synth_pages"

export GOAPP_DATA_DIR="$LOCAL_DATA"
export GOAPP_MODELS_DIR="$LOCAL_MODELS"

case "$MODEL_TYPE" in
  boards)
    exec python -m goapp.ml.board_detect.train \
        --model-out "$BUCKET_DATA/models/board_detector.pt" "$@"
    ;;
  stones)
    exec python -m goapp.ml.stone_detect.train \
        --model-out "$BUCKET_DATA/models/stone_detector.pt" "$@"
    ;;
  *)
    echo "first arg must be 'boards' or 'stones'; got: '$MODEL_TYPE'" >&2
    exit 1
    ;;
esac
