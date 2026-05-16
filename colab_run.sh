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
#   5. Downloads RDD2022 Japan to a Drive cache (skip if cached)
#   6. Converts XML annotations to YOLO format (skip if labels exist)
#   7. Generates stratified 10-fold splits (skip if folds exist)
#   8. Verifies the activation patch works end-to-end
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
DATASET_NAME="RDD2022_Japan"
DATASET_URL="https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Japan.zip"

# --- Paths ---
# Adjust DRIVE_ROOT if your Drive layout differs.
DRIVE_ROOT="/content/drive/MyDrive/Dissertation_Results"
DRIVE_CACHE="${DRIVE_ROOT}/cache"
DRIVE_RESULTS="${DRIVE_ROOT}/Results"

# Repo location: by default we assume this script lives at $REPO_DIR/colab_run.sh.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORK_DIR="/content/src"
ULTRA_DIR="${WORK_DIR}/ultralytics"
DATASET_ZIP="${DRIVE_CACHE}/${DATASET_NAME}.zip"
DATASET_DIR="${WORK_DIR}/Japan"
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
echo "[1/7] Setting up Ultralytics @ $ULTRALYTICS_TAG..."
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
echo "[2/7] Applying activation patch..."
bash "$REPO_DIR/patches/apply_patch.sh" "$REPO_DIR" "$ULTRA_DIR"

# ============================================================================
# Step 3: install dependencies
# ============================================================================
echo ""
echo "[3/7] Installing Ultralytics (editable) and supporting libraries..."
pip install -q -e "$ULTRA_DIR"
pip install -q iterative-stratification scikit-learn pyyaml
echo "      Done."

# ============================================================================
# Step 4: download RDD2022 Japan (cached on Drive)
# ============================================================================
echo ""
echo "[4/7] Acquiring RDD2022 Japan dataset..."
if [ -f "$DATASET_ZIP" ]; then
    SIZE_MB=$(du -m "$DATASET_ZIP" | cut -f1)
    echo "      Found cached zip on Drive (${SIZE_MB} MB): $DATASET_ZIP"
else
    echo "      Downloading from S3 to Drive cache (this is one-time, ~5GB)..."
    wget -q --show-progress -O "$DATASET_ZIP" "$DATASET_URL"
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "      Unzipping to $WORK_DIR..."
    unzip -q "$DATASET_ZIP" -d "$WORK_DIR"
    # The zip extracts to RDD2022_Japan/Japan; normalize the path.
    if [ -d "${WORK_DIR}/RDD2022_Japan/Japan" ] && [ ! -d "$DATASET_DIR" ]; then
        mv "${WORK_DIR}/RDD2022_Japan/Japan" "$DATASET_DIR"
        rmdir "${WORK_DIR}/RDD2022_Japan" 2>/dev/null || true
    fi
else
    echo "      Already unzipped at $DATASET_DIR (skipping)."
fi

# ============================================================================
# Step 5: convert XML to YOLO label format
# ============================================================================
echo ""
echo "[5/7] Converting VOC XML annotations to YOLO format..."
LABELS_DIR="${DATASET_DIR}/train/labels"
XML_DIR="${DATASET_DIR}/train/annotations/xmls"
N_LABELS=$(find "$LABELS_DIR" -name "*.txt" 2>/dev/null | wc -l || echo 0)
N_XMLS=$(find "$XML_DIR" -name "*.xml" 2>/dev/null | wc -l || echo 0)

if [ "$N_LABELS" -gt 0 ] && [ "$N_LABELS" -ge "$N_XMLS" ]; then
    echo "      Found $N_LABELS YOLO labels (>= $N_XMLS XMLs). Skipping conversion."
else
    python "$REPO_DIR/data_prep/xml_parser.py" \
        --xml-dir "$XML_DIR" \
        --labels-out-dir "$LABELS_DIR"
fi

# ============================================================================
# Step 6: generate 10-fold stratified splits
# ============================================================================
echo ""
echo "[6/7] Generating stratified 10-fold splits..."
if [ -f "${KFOLD_DIR}/dataset_fold9.yaml" ]; then
    echo "      Found existing 10-fold structure at $KFOLD_DIR. Skipping split."
else
    python "$REPO_DIR/data_prep/kfold_split.py" \
        --images-dir "${DATASET_DIR}/train/images" \
        --labels-dir "$LABELS_DIR" \
        --output-root "$KFOLD_DIR" \
        --n-folds 10 \
        --seed 42 \
        --class-names-yaml "$REPO_DIR/configs/dataset.yaml"
fi

# ============================================================================
# Step 7: verify the activation patch end-to-end
# ============================================================================
echo ""
echo "[7/7] Verifying activation patch..."
cd "$REPO_DIR"
python "$REPO_DIR/runner/verify_patch.py" \
    --model-yaml "$REPO_DIR/configs/model_eaf.yaml" \
    --expected-epsilon 0.1 \
    --expected-r 0.3
cd - >/dev/null

echo ""
echo "============================================================"
echo "Setup complete. Ready to launch the sweep."
echo "============================================================"
echo ""
echo "Next phase: implement runner/run_sweep.py and runner/aggregate_results.py."
echo "Per-fold YAMLs are at: $KFOLD_DIR/dataset_fold{0..9}.yaml"
