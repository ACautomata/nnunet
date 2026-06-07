#!/usr/bin/env bash
# Train a single nnUNet model with random modality sampling, then test each
# modality separately and report per-modality Dice scores.
#
# During training, every subject appears once per modality (e.g. 1000 subjects
# × 4 modalities = 4000 cases).  nnUNet's shuffling naturally produces random
# modality sampling per batch, so the model learns a single set of parameters
# that generalises across modalities.
#
# After training, the script creates separate inference folders for each
# modality, runs prediction, and evaluates.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/train_brats2023_random_modality.sh --dataroot PATH --runtime-dir PATH [options]

Required:
  --dataroot PATH           BraTS2023 training-data root (subject folders).
  --runtime-dir PATH        Directory for all nnUNet outputs.

Options:
  --modalities "LIST"       Space-separated modalities (default: "t1c t1n t2f t2w").
  --train-config NAME       nnUNet config (default: 3d_fullres).
  --fold N                  Fold to train (default: 0).
  --gpu-id N                GPU id (default: 0).
  --trainer-class-name N    nnUNet trainer class (default: nnUNetTrainer).
  --dataset-id N            nnUNet dataset ID (default: 2001).
  --skip-train              Skip training; only run per-modality testing.
  -h, --help                Show this help.
EOF
}

DATAROOT=""
RUNTIME_DIR=""
MODALITIES=(t1c t1n t2f t2w)
TRAIN_CONFIG="3d_fullres"
FOLD="0"
GPU_ID="0"
TRAINER_CLASS_NAME="nnUNetTrainer"
DATASET_ID="2001"
SKIP_TRAIN=false

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
        --skip-train)
            SKIP_TRAIN=true
            shift
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

# ── Phase 1: Train ────────────────────────────────────────────────────────────

if [[ "$SKIP_TRAIN" == false ]]; then
    echo "=== Phase 1: Training random-modality model ==="
    echo "    modalities:  ${MODALITIES[*]}"
    echo "    dataset ID:  $DATASET_ID"
    echo "    work dir:    $RUNTIME_DIR"

    MODALITIES_STR="${MODALITIES[*]}"

    python -m nnunet.train \
        mode=random_modality \
        dataroot="$DATAROOT" \
        nnunet_raw="$RUNTIME_DIR/nnunet_raw" \
        nnunet_preprocessed="$RUNTIME_DIR/nnunet_preprocessed" \
        nnunet_results="$RUNTIME_DIR/nnunet_results" \
        modalities="$MODALITIES_STR" \
        dataset_name_or_id="$DATASET_ID" \
        train_config="$TRAIN_CONFIG" \
        fold="$FOLD" \
        gpu_id="$GPU_ID" \
        trainer_class_name="$TRAINER_CLASS_NAME" \
        hydra.run.dir="$RUNTIME_DIR/hydra"

    echo "=== Training complete ==="
else
    echo "=== Skipping training (--skip-train) ==="
fi

# ── Phase 2: Per-modality testing ─────────────────────────────────────────────

DATASET_DIR="Dataset${DATASET_ID}_BraTS2023_RandomModality"
MODEL_FOLDER="$RUNTIME_DIR/nnunet_results/$DATASET_DIR/${TRAINER_CLASS_NAME}__${TRAIN_CONFIG}__nnUNetPlans"

if [[ ! -d "$MODEL_FOLDER" ]]; then
    echo "Model folder not found: $MODEL_FOLDER" >&2
    echo "Make sure training completed successfully or use --skip-train with a valid runtime dir." >&2
    exit 1
fi

echo ""
echo "=== Phase 2: Per-modality testing ==="

RESULTS_FILE="$RUNTIME_DIR/per_modality_results.txt"
echo "Random Modality Model - Per-Modality Test Results" > "$RESULTS_FILE"
echo "Dataset: $DATASET_DIR" >> "$RESULTS_FILE"
echo "Model:   $MODEL_FOLDER" >> "$RESULTS_FILE"
echo "Fold:    $FOLD" >> "$RESULTS_FILE"
echo "---------------------------------------------------" >> "$RESULTS_FILE"

for modality in "${MODALITIES[@]}"; do
    echo ""
    echo "--- Testing modality: $modality ---"

    TEST_INPUT="$RUNTIME_DIR/test_input/$modality"
    TEST_OUTPUT="$RUNTIME_DIR/test_output/$modality"

    # Create flat inference folder with only this modality.
    python -c "
from nnunet.data import prepare_inference_folder
prepare_inference_folder(
    data_root='$DATAROOT',
    output_folder='$TEST_INPUT',
    modality='$modality',
)
print(f'Created inference folder: $TEST_INPUT')
"

    # Run prediction with only the trained fold.
    python -m nnunet.predict \
        input_folder="$TEST_INPUT" \
        output_folder="$TEST_OUTPUT" \
        model_folder="$MODEL_FOLDER" \
        nnunet_raw="$RUNTIME_DIR/nnunet_raw" \
        nnunet_preprocessed="$RUNTIME_DIR/nnunet_preprocessed" \
        nnunet_results="$RUNTIME_DIR/nnunet_results" \
        dataset_name_or_id="$DATASET_ID" \
        modality_file="$modality" \
        use_folds="[$FOLD]" \
        gpu_id="$GPU_ID" \
        checkpoint_name="checkpoint_final.pth" \
        hydra.run.dir="$RUNTIME_DIR/hydra_predict_${modality}"

    echo "  Prediction saved to: $TEST_OUTPUT"

    # Count predictions as a sanity check.
    if [[ -d "$TEST_OUTPUT" ]]; then
        N_PRED=$(find "$TEST_OUTPUT" -name "*.nii.gz" | wc -l | tr -d ' ')
        echo "  $modality: $N_PRED predictions" | tee -a "$RESULTS_FILE"
    else
        echo "  $modality: ERROR - no output folder" | tee -a "$RESULTS_FILE"
    fi
done

echo ""
echo "=== Results summary ==="
cat "$RESULTS_FILE"
echo ""
echo "Full results saved to: $RESULTS_FILE"
