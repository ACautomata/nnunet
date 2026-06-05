#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/train_brats2023_all_modalities.sh --dataroot PATH --runtime-dir PATH [options]

Required path arguments:
  --dataroot PATH      BraTS2023 training-data root.
  --runtime-dir PATH   Directory where all nnUNet outputs will be written.

Options:
  --modalities "LIST"       Quoted space-separated modalities (default: "t1c t1n t2f t2w").
  --train-config NAME      nnUNet config to train (default: 3d_fullres).
  --fold N                Fold to train (default: 0).
  --gpu-id N              GPU id passed to nnUNet (default: 0).
  --trainer-class-name N   nnUNet trainer class (default: nnUNetTrainer).
  --dataset-id N          nnUNet dataset id inside each modality work dir (default: 1001).
  -h, --help              Show this help.
EOF
}

DATAROOT=""
RUNTIME_DIR=""
MODALITIES=(t1c t1n t2f t2w)
TRAIN_CONFIG="3d_fullres"
FOLD="0"
GPU_ID="0"
TRAINER_CLASS_NAME="nnUNetTrainer"
DATASET_ID="1001"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataroot)
            DATAROOT="${2:-}"
            shift 2
            ;;
        --runtime-dir)
            RUNTIME_DIR="${2:-}"
            shift 2
            ;;
        --modalities)
            read -r -a MODALITIES <<< "${2:-}"
            shift 2
            ;;
        --train-config)
            TRAIN_CONFIG="${2:-}"
            shift 2
            ;;
        --fold)
            FOLD="${2:-}"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="${2:-}"
            shift 2
            ;;
        --trainer-class-name)
            TRAINER_CLASS_NAME="${2:-}"
            shift 2
            ;;
        --dataset-id)
            DATASET_ID="${2:-}"
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$DATAROOT" || -z "$RUNTIME_DIR" ]]; then
    echo "Both --dataroot and --runtime-dir are required." >&2
    usage >&2
    exit 2
fi

if [[ ! -d "$DATAROOT" ]]; then
    echo "BraTS2023 data root not found: $DATAROOT" >&2
    exit 1
fi

mkdir -p "$RUNTIME_DIR"

for modality in "${MODALITIES[@]}"; do
    case "$modality" in
        t1c | t1n | t2f | t2w) ;;
        *)
            echo "Unknown modality '$modality'. Expected one of: t1c t1n t2f t2w" >&2
            exit 1
            ;;
    esac

    work_dir="$RUNTIME_DIR/$modality"
    mkdir -p "$work_dir"

    echo "=== Training BraTS2023 modality: $modality ==="
    echo "    work dir: $work_dir"

    python -m nnunet.train \
        mode=all \
        dataroot="$DATAROOT" \
        nnunet_raw="$work_dir/nnunet_raw" \
        nnunet_preprocessed="$work_dir/nnunet_preprocessed" \
        nnunet_results="$work_dir/nnunet_results" \
        modality_file="$modality" \
        dataset_name_or_id="$DATASET_ID" \
        train_config="$TRAIN_CONFIG" \
        fold="$FOLD" \
        gpu_id="$GPU_ID" \
        trainer_class_name="$TRAINER_CLASS_NAME" \
        hydra.run.dir="$work_dir/hydra"
done
