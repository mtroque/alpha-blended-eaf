#!/bin/bash
# colab_run.sh
# ----------------------------------------------------------------------------
# Single entry-point script for running alpha-blended-eaf experiments on Colab.
#
# Usage (from a Colab cell, after mounting Drive):
#   !bash /content/alpha-blended-eaf/colab_run.sh
#
# What it does:
#   1. Sets up paths (assumes Drive is mounted at /content/drive)
#   2. Clones or updates pinned Ultralytics v8.4.50
#   3. Applies the surgical patch (adds AlphaBlendedEAFReLU to Ultralytics)
#   4. Installs dependencies
#   5. Downloads RDD2022 Japan, India, and China datasets
#   6. Converts VOC XML annotations to YOLO label format
#   7. Creates dataset YAMLs for full-train and unseen evaluation runs
#   8. Generates stratified 10-fold splits for Japan
#   9. Verifies the activation patch works end-to-end
#
# What it does NOT do:
#   - Run the sweep (runner/run_sweep.py is built in a later phase)
#   - Aggregate results (runner/aggregate_results.py is built in a later phase)
#
# Re-running is safe and cheap: each step has a "skip if already done" guard.
# ----------------------------------------------------------------------------

set -euo pipefail

# --- Configuration ---
ULTRALYTICS_TAG="v8.4.50"
REPO_NAME="alpha-blended-eaf"

JAPAN_DATASET_NAME="RDD2022_Japan"
INDIA_DATASET_NAME="RDD2022_India"
CHINA_DATASET_NAME="RDD2022_China_MotorBike"

JAPAN_DATASET_URL="https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Japan.zip"
INDIA_DATASET_URL="https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip"
CHINA_DATASET_URL="https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_China_MotorBike.zip"

# --- Paths ---
# Adjust DRIVE_ROOT if your Drive layout differs.
DRIVE_ROOT="/content/drive/MyDrive/Dissertation_Results"
DRIVE_CACHE="${DRIVE_ROOT}/cache"
DRIVE_RESULTS="${DRIVE_ROOT}/Results"

# Repo location: by default we assume this script lives at $REPO_DIR/colab_run.sh.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORK_DIR="/content/src"
ULTRA_DIR="${WORK_DIR}/ultralytics"

JAPAN_DATASET_ZIP="${DRIVE_CACHE}/${JAPAN_DATASET_NAME}.zip"
INDIA_DATASET_ZIP="${DRIVE_CACHE}/${INDIA_DATASET_NAME}.zip"
CHINA_DATASET_ZIP="${DRIVE_CACHE}/${CHINA_DATASET_NAME}.zip"

JAPAN_DATASET_DIR="${WORK_DIR}/Japan"
INDIA_DATASET_DIR="${WORK_DIR}/India"
CHINA_DATASET_DIR="${WORK_DIR}/China_MotorBike"

KFOLD_DIR="${WORK_DIR}/Japan_kfold"

# --- Sanity: is Drive mounted? ---
if [ ! -d "/content/drive/MyDrive" ]; then
    echo "ERROR: Google Drive is not mounted at /content/drive."
    echo "Mount it from your Colab notebook before running this script:"
    echo ""
    echo "    from google.colab import drive"
    echo "    drive.mount('/content/drive')"
    echo ""
    exit 1
fi

# Ensure Drive directories exist.
mkdir -p "$DRIVE_CACHE" "$DRIVE_RESULTS"

echo "============================================================"
echo "alpha-blended-eaf: Colab pipeline setup"
echo "============================================================"
echo "  Repo dir:      $REPO_DIR"
echo "  Work dir:      $WORK_DIR"
echo "  Ultralytics:   $ULTRALYTICS_TAG"
echo "  Drive cache:   $DRIVE_CACHE"
echo "  Drive results: $DRIVE_RESULTS"
echo "============================================================"
echo ""

mkdir -p "$WORK_DIR"

# ============================================================================
# Step 1: clone or update pinned Ultralytics
# ============================================================================
echo "[1/8] Setting up Ultralytics @ $ULTRALYTICS_TAG..."
if [ ! -d "$ULTRA_DIR/.git" ]; then
    git clone --branch "$ULTRALYTICS_TAG" --depth 1 \
        https://github.com/ultralytics/ultralytics "$ULTRA_DIR"
    echo "      Cloned fresh."
else
    cd "$ULTRA_DIR"
    CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "unknown")
    if [ "$CURRENT_TAG" != "$ULTRALYTICS_TAG" ]; then
        echo "      Existing checkout is at $CURRENT_TAG, switching to $ULTRALYTICS_TAG..."
        git fetch --tags --depth 1 origin "refs/tags/${ULTRALYTICS_TAG}:refs/tags/${ULTRALYTICS_TAG}"
        git checkout "tags/${ULTRALYTICS_TAG}"
    else
        echo "      Already at $ULTRALYTICS_TAG."
    fi
    cd - >/dev/null
fi

# ============================================================================
# Step 2: apply the activation patch
# ============================================================================
echo ""
echo "[2/8] Applying activation patch..."
bash "$REPO_DIR/patches/apply_patch.sh" "$REPO_DIR" "$ULTRA_DIR"

# ============================================================================
# Step 3: install dependencies
# ============================================================================
echo ""
echo "[3/8] Installing Ultralytics (editable) and supporting libraries..."
pip install -q -e "$ULTRA_DIR"
pip install -q iterative-stratification scikit-learn pyyaml
echo "      Done."

# ============================================================================
# Step 4: download RDD2022 Japan, India, and China datasets
# ============================================================================
echo ""
echo "[4/8] Acquiring RDD2022 datasets..."
download_if_missing () {
    local ZIP_PATH=$1
    local URL=$2
    local LABEL=$3

    if [ -f "$ZIP_PATH" ]; then
        SIZE_MB=$(du -m "$ZIP_PATH" | cut -f1)
        echo "      Found cached ${LABEL} zip on Drive (${SIZE_MB} MB)"
    else
        echo "      Downloading ${LABEL} dataset..."
        wget -q --show-progress -O "$ZIP_PATH" "$URL"
    fi
}

download_if_missing "$JAPAN_DATASET_ZIP" "$JAPAN_DATASET_URL" "Japan"
download_if_missing "$INDIA_DATASET_ZIP" "$INDIA_DATASET_URL" "India"
download_if_missing "$CHINA_DATASET_ZIP" "$CHINA_DATASET_URL" "China"

unzip_dataset () {
    local ZIP_PATH=$1
    local TARGET_DIR=$2
    local EXTRACT_ROOT=$3
    local INNER_DIR=$4

    if [ ! -d "$TARGET_DIR" ]; then
        echo "      Unzipping ${INNER_DIR}..."
        unzip -oq "$ZIP_PATH" -d "$WORK_DIR"

        if [ -d "${WORK_DIR}/${EXTRACT_ROOT}/${INNER_DIR}" ] && [ ! -d "$TARGET_DIR" ]; then
            mv "${WORK_DIR}/${EXTRACT_ROOT}/${INNER_DIR}" "$TARGET_DIR"
            rmdir "${WORK_DIR}/${EXTRACT_ROOT}" 2>/dev/null || true
        fi
    else
        echo "      Already unzipped at $TARGET_DIR (skipping)."
    fi
}

unzip_dataset \
    "$JAPAN_DATASET_ZIP" \
    "$JAPAN_DATASET_DIR" \
    "RDD2022_Japan" \
    "Japan"

unzip_dataset \
    "$INDIA_DATASET_ZIP" \
    "$INDIA_DATASET_DIR" \
    "RDD2022_India" \
    "India"

unzip_dataset \
    "$CHINA_DATASET_ZIP" \
    "$CHINA_DATASET_DIR" \
    "RDD2022_China_MotorBike" \
    "China_MotorBike"

# ============================================================================
# Step 5: convert XML to YOLO label format
# ============================================================================
echo ""
echo "[5/8] Converting VOC XML annotations to YOLO format..."

convert_xml_to_yolo () {
    local SPLIT_DIR=$1
    local LABEL=$2

    local LABELS_DIR="${SPLIT_DIR}/labels"
    local XML_DIR="${SPLIT_DIR}/annotations/xmls"

    mkdir -p "$LABELS_DIR"

    local N_LABELS
    local N_XMLS

    N_LABELS=$(find "$LABELS_DIR" -name "*.txt" 2>/dev/null | wc -l || echo 0)
    N_XMLS=$(find "$XML_DIR" -name "*.xml" 2>/dev/null | wc -l || echo 0)

    if [ "$N_LABELS" -gt 0 ] && [ "$N_LABELS" -ge "$N_XMLS" ]; then
        echo "      ${LABEL}: found $N_LABELS YOLO labels (>= $N_XMLS XMLs). Skipping conversion."
    else
        echo "      ${LABEL}: converting $N_XMLS XML files..."
        python "$REPO_DIR/data_prep/xml_parser.py" \
            --xml-dir "$XML_DIR" \
            --labels-out-dir "$LABELS_DIR"
    fi
}

convert_xml_to_yolo "${JAPAN_DATASET_DIR}/train" "Japan train"
convert_xml_to_yolo "${INDIA_DATASET_DIR}/train" "India train"
convert_xml_to_yolo "${CHINA_DATASET_DIR}/train" "China_MotorBike train"

# ============================================================================
# Step 6: create dataset YAMLs for final unseen evaluation
# ============================================================================
echo ""
echo "[6/8] Creating dataset YAMLs for final training and evaluation..."

create_eval_dataset_yaml () {
    local DATASET_DIR=$1
    local OUT_YAML=$2
    local LABEL=$3

    cat > "$OUT_YAML" <<EOF
path: ${DATASET_DIR}
train: train/images
val: train/images

nc: 9

names:
  0: 'D00'
  1: 'D01'
  2: 'D10'
  3: 'D11'
  4: 'D20'
  5: 'D40'
  6: 'D43'
  7: 'D44'
  8: 'D50'
EOF

    echo "      ${LABEL}: wrote ${OUT_YAML}"
}

create_eval_dataset_yaml \
    "$INDIA_DATASET_DIR" \
    "${INDIA_DATASET_DIR}/dataset_test.yaml" \
    "India train"

create_eval_dataset_yaml \
    "$CHINA_DATASET_DIR" \
    "${CHINA_DATASET_DIR}/dataset_test.yaml" \
    "China_MotorBike train"

create_full_train_dataset_yaml () {
    local DATASET_DIR=$1
    local OUT_YAML=$2
    local LABEL=$3

    cat > "$OUT_YAML" <<EOF
path: ${DATASET_DIR}
train: train/images
val: train/images

nc: 9

names:
  0: 'D00'
  1: 'D01'
  2: 'D10'
  3: 'D11'
  4: 'D20'
  5: 'D40'
  6: 'D43'
  7: 'D44'
  8: 'D50'
EOF

    echo "      ${LABEL}: wrote ${OUT_YAML}"
}

create_full_train_dataset_yaml \
    "$JAPAN_DATASET_DIR" \
    "${JAPAN_DATASET_DIR}/dataset_full.yaml" \
    "Japan full train"

# ============================================================================
# Step 7: generate 10-fold stratified splits for Japan only
# ============================================================================
echo ""
echo "[7/8] Generating stratified 10-fold splits..."
if [ -f "${KFOLD_DIR}/dataset_fold9.yaml" ]; then
    echo "      Found existing 10-fold structure at $KFOLD_DIR. Skipping split."
else
    python "${REPO_DIR}/data_prep/kfold_split.py" \
        --images-dir "${JAPAN_DATASET_DIR}/train/images" \
        --labels-dir "${JAPAN_DATASET_DIR}/train/labels" \
        --output-root "$KFOLD_DIR" \
        --n-folds 10 \
        --seed 42 \
        --class-names-yaml "${REPO_DIR}/configs/dataset.yaml"
fi

# ============================================================================
# Step 8: verify the activation patch end-to-end
# ============================================================================
echo ""
echo "[8/8] Verifying activation patch..."
cd "${REPO_DIR}"
python "${REPO_DIR}/runner/verify_patch.py" \
    --model-yaml "${REPO_DIR}/configs/model_eaf.yaml" \
    --expected-epsilon 0.1 \
    --expected-r 0.3
cd - >/dev/null

echo ""
echo "============================================================"
echo "Setup complete. Ready to launch the sweep."
echo "============================================================"
echo ""
echo "Next: Stage 1 sweep (4 configs x 10 folds = 40 training runs)."
echo ""
echo "Dry run (1 config x 1 fold x 2 epochs, ~10 min) to validate the full pipeline:"
echo ""
echo "    python ${REPO_DIR}/runner/run_sweep.py \\"
echo "        --sweep-config  ${REPO_DIR}/configs/sweep_configs.yaml \\"
echo "        --kfold-dir     ${KFOLD_DIR} \\"
echo "        --results-dir   \"${DRIVE_RESULTS}\" \\"
echo "        --repo-root     ${REPO_DIR} \\"
echo "        --dry-run"
echo ""
echo "Full Stage 1 sweep (drop --dry-run when dry run passes):"
echo ""
echo "    python ${REPO_DIR}/runner/run_sweep.py \\"
echo "        --sweep-config  ${REPO_DIR}/configs/sweep_configs.yaml \\"
echo "        --kfold-dir     ${KFOLD_DIR} \\"
echo "        --results-dir   \"${DRIVE_RESULTS}\" \\"
echo "        --repo-root     ${REPO_DIR}"
echo ""
echo "Per-fold dataset YAMLs:  ${KFOLD_DIR}/dataset_fold{0..9}.yaml"
echo "Results will be written to: ${DRIVE_RESULTS}/sweep_runs/"
