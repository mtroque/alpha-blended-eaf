#!/bin/bash
# apply_patch.sh
# ----------------------------------------------------------------------------
# Surgical integration of the AlphaBlendedEAFReLU activation into a cloned
# Ultralytics repository.
#
# What this script does:
#   1. Copies activation/custom_activations.py into ultralytics/nn/modules/
#   2. Appends the activation classes to ultralytics/nn/modules/__init__.py
#   3. Appends an import of those classes into ultralytics/nn/tasks.py so the
#      eval() call in parse_model() can resolve them.
#
# All edits are idempotent: re-running the script will not duplicate lines.
#
# Why this approach:
#   parse_model() in tasks.py contains the line
#     if act: Conv.default_act = eval(act)
#   which means any string in the model YAML's `activation:` key is evaluated
#   in tasks.py's module-level namespace. Adding our class to that namespace
#   (via an import) is the entire integration. No edits to parse_model itself.
# ----------------------------------------------------------------------------

set -euo pipefail

# --- Argument parsing ---
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <repo_root> <ultralytics_root>"
    echo ""
    echo "Arguments:"
    echo "  repo_root          Path to the cloned alpha-blended-eaf repository"
    echo "  ultralytics_root   Path to the cloned ultralytics repository"
    echo ""
    echo "Example:"
    echo "  $0 /content/alpha-blended-eaf /content/ultralytics"
    exit 1
fi

REPO_ROOT="$1"
ULTRA_ROOT="$2"

ACTIVATION_SRC="${REPO_ROOT}/activation/custom_activations.py"
ACTIVATION_DST="${ULTRA_ROOT}/ultralytics/nn/modules/custom_activations.py"
INIT_FILE="${ULTRA_ROOT}/ultralytics/nn/modules/__init__.py"
TASKS_FILE="${ULTRA_ROOT}/ultralytics/nn/tasks.py"

# --- Sanity checks ---
if [ ! -f "$ACTIVATION_SRC" ]; then
    echo "ERROR: activation source file not found: $ACTIVATION_SRC"
    exit 1
fi

if [ ! -d "${ULTRA_ROOT}/ultralytics/nn/modules" ]; then
    echo "ERROR: ultralytics/nn/modules directory not found in: $ULTRA_ROOT"
    echo "Did you clone the Ultralytics repository?"
    exit 1
fi

echo "Applying alpha-blended-eaf patch to Ultralytics..."
echo "  Repo:        $REPO_ROOT"
echo "  Ultralytics: $ULTRA_ROOT"
echo ""

# --- Step 1: Copy the activation module ---
cp "$ACTIVATION_SRC" "$ACTIVATION_DST"
echo "[1/3] Copied custom_activations.py -> ultralytics/nn/modules/"

# --- Step 2: Patch __init__.py ---
IMPORT_LINE='from .custom_activations import AlphaBlendedEAFReLU, EinsteinActivationFunction  # noqa: F401  # alpha-blended-eaf'

if grep -qF "alpha-blended-eaf" "$INIT_FILE"; then
    echo "[2/3] __init__.py already patched (skipping)"
else
    echo "" >> "$INIT_FILE"
    echo "# Custom activations for the alpha-blended-eaf research project" >> "$INIT_FILE"
    echo "$IMPORT_LINE" >> "$INIT_FILE"
    echo "[2/3] Appended import to ultralytics/nn/modules/__init__.py"
fi

# --- Step 3: Patch tasks.py ---
# The eval() in parse_model() resolves names from tasks.py's module globals.
# We add an import at the top of the file so AlphaBlendedEAFReLU is visible.
TASKS_IMPORT='from ultralytics.nn.modules import AlphaBlendedEAFReLU, EinsteinActivationFunction  # noqa: F401  # alpha-blended-eaf'

if grep -qF "alpha-blended-eaf" "$TASKS_FILE"; then
    echo "[3/3] tasks.py already patched (skipping)"
else
    # We need to insert our import AFTER the last existing
    # `from ultralytics.nn.modules import ...` statement. The existing one
    # may span multiple lines (multi-line parenthesized import), so we find
    # the start of the last such statement, then scan forward for its closing
    # `)` (or for a non-continuation line if it's a single-line import).
    START_LINE=$(grep -n "^from ultralytics.nn.modules import" "$TASKS_FILE" | tail -n 1 | cut -d: -f1)

    if [ -z "$START_LINE" ]; then
        echo "WARNING: could not find 'from ultralytics.nn.modules import' in tasks.py"
        echo "         Appending import at end of file as fallback."
        echo "" >> "$TASKS_FILE"
        echo "$TASKS_IMPORT" >> "$TASKS_FILE"
    else
        # Use awk to find the end of the import statement, then insert after it.
        # Logic:
        #   - Track when we're inside the target import statement.
        #   - "Inside" begins at $START_LINE.
        #   - If the first line contains "(" but not ")", we're in a multi-line
        #     import; keep going until we see ")".
        #   - If the first line contains both "(" and ")" or neither, it's a
        #     single-line import; end immediately.
        awk -v start="$START_LINE" -v insert="$TASKS_IMPORT" '
            NR == start {
                print
                if (index($0, "(") > 0 && index($0, ")") == 0) {
                    in_block = 1
                } else {
                    print insert
                }
                next
            }
            in_block == 1 {
                print
                if (index($0, ")") > 0) {
                    print insert
                    in_block = 0
                }
                next
            }
            { print }
        ' "$TASKS_FILE" > "${TASKS_FILE}.tmp"
        mv "${TASKS_FILE}.tmp" "$TASKS_FILE"
        echo "[3/3] Inserted import into ultralytics/nn/tasks.py (after import block starting at line $START_LINE)"
    fi
fi

echo ""
echo "Patch applied successfully."
echo ""
echo "Verify with:"
echo "  python -c 'from ultralytics.nn.modules import AlphaBlendedEAFReLU; print(AlphaBlendedEAFReLU(0.1, 0.4))'"
