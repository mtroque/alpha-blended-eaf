"""Generate stratified 10-fold splits for the RDD2022 Japan training set.

Stratification for object detection is non-trivial because each image can
contain multiple object classes. A naive random split risks producing folds
with very few or zero examples of rare classes (D40, D43, D44 in particular).

Strategy
--------
We use sklearn's MultilabelStratifiedKFold-equivalent (iterative stratification)
via the `iterative-stratification` package if available, falling back to
sklearn's StratifiedKFold on a "dominant class per image" label if not.

For each image, the dominant class is the one with the highest count (ties
broken by lower class id). This is a reasonable approximation that keeps the
script dependency-light.

Outputs
-------
For each fold k in 0..K-1, the script creates:

    <output_root>/foldK/
        train/images/  -> symlinks to source images in train folds
        train/labels/  -> symlinks to source labels in train folds
        val/images/    -> symlinks to source images in val fold
        val/labels/    -> symlinks to source labels in val fold

It also writes <output_root>/dataset_foldK.yaml for each fold, ready to be
passed to `yolo train data=...`.

Symlinks are used (instead of copies) to avoid 10x disk usage on Colab.
"""

import argparse
import os
import glob
import yaml
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Dict


def load_image_labels(images_dir: str, labels_dir: str) -> List[Tuple[str, str, List[int]]]:
    """Return a list of (image_path, label_path, class_ids) tuples.

    class_ids is the list of integer class IDs present in the label file
    (one entry per bounding box; duplicates allowed).
    """
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg"))) or \
                  sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    entries = []
    for img in image_paths:
        base = os.path.splitext(os.path.basename(img))[0]
        lbl = os.path.join(labels_dir, f"{base}.txt")
        class_ids: List[int] = []
        if os.path.exists(lbl):
            with open(lbl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_ids.append(int(line.split()[0]))
        entries.append((img, lbl, class_ids))
    return entries


def dominant_class(class_ids: List[int]) -> int:
    """Return the most frequent class in the image; ties broken by lower id.

    Returns -1 for images with no labels (negative samples).
    """
    if not class_ids:
        return -1
    counts = Counter(class_ids)
    # Sort by (-count, class_id) so ties prefer the lower id.
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def stratified_split(entries, n_folds: int, seed: int) -> List[List[int]]:
    """Return a list of length n_folds; each element is a list of entry indices
    that form the validation set for that fold.

    Tries iterative-stratification (multi-label aware) first; falls back to
    sklearn StratifiedKFold on dominant_class if the package isn't installed.
    """
    n = len(entries)
    indices = list(range(n))

    try:
        # Best option: multi-label stratification across all classes per image.
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        import numpy as np

        # Build a binary multi-label matrix: rows=images, cols=classes.
        n_classes = max((max(e[2]) for e in entries if e[2]), default=-1) + 1
        Y = np.zeros((n, max(n_classes, 1)), dtype=int)
        for i, (_, _, class_ids) in enumerate(entries):
            for c in class_ids:
                Y[i, c] = 1
        X = np.zeros((n, 1))  # placeholder

        mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = []
        for _, val_idx in mskf.split(X, Y):
            folds.append(val_idx.tolist())
        print("Stratification: multi-label (iterative-stratification)")
        return folds

    except ImportError:
        # Fallback: sklearn StratifiedKFold on dominant class.
        from sklearn.model_selection import StratifiedKFold

        labels = [dominant_class(e[2]) for e in entries]
        # StratifiedKFold can't handle classes with <n_folds members; merge -1
        # (negative samples) into a sentinel group and treat any singleton
        # class similarly. With 9 classes and ~10k Japan images this should
        # not occur, but we guard against it.
        label_counts = Counter(labels)
        valid_labels = [l for l, c in label_counts.items() if c >= n_folds]
        adjusted = [l if l in valid_labels else -999 for l in labels]

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = []
        for _, val_idx in skf.split(indices, adjusted):
            folds.append(val_idx.tolist())
        print("Stratification: dominant-class fallback (sklearn StratifiedKFold)")
        print("NOTE: install 'iterative-stratification' for better multi-label balance.")
        return folds


def symlink_force(src: str, dst: str) -> None:
    """Create a symlink dst -> src, replacing dst if it already exists."""
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def write_fold(
    fold_idx: int,
    val_indices: List[int],
    entries: list,
    output_root: str,
    class_names: Dict[int, str],
) -> str:
    """Materialize one fold's directory structure and dataset YAML.

    Returns the path to the written dataset_foldK.yaml.
    """
    fold_root = os.path.join(output_root, f"fold{fold_idx}")
    train_img = os.path.join(fold_root, "train", "images")
    train_lbl = os.path.join(fold_root, "train", "labels")
    val_img = os.path.join(fold_root, "val", "images")
    val_lbl = os.path.join(fold_root, "val", "labels")
    for d in (train_img, train_lbl, val_img, val_lbl):
        os.makedirs(d, exist_ok=True)

    val_set = set(val_indices)
    for i, (img, lbl, _) in enumerate(entries):
        img_name = os.path.basename(img)
        lbl_name = os.path.basename(lbl)
        if i in val_set:
            symlink_force(img, os.path.join(val_img, img_name))
            if os.path.exists(lbl):
                symlink_force(lbl, os.path.join(val_lbl, lbl_name))
        else:
            symlink_force(img, os.path.join(train_img, img_name))
            if os.path.exists(lbl):
                symlink_force(lbl, os.path.join(train_lbl, lbl_name))

    # Write the dataset YAML for this fold.
    yaml_path = os.path.join(output_root, f"dataset_fold{fold_idx}.yaml")
    yaml_doc = {
        "path": os.path.abspath(fold_root),
        "train": "train/images",
        "val": "val/images",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_doc, f, sort_keys=False)

    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Generate stratified 10-fold splits.")
    parser.add_argument(
        "--images-dir", required=True,
        help="Source images directory (e.g. /content/src/Japan/train/images)"
    )
    parser.add_argument(
        "--labels-dir", required=True,
        help="Source labels directory (e.g. /content/src/Japan/train/labels)"
    )
    parser.add_argument(
        "--output-root", required=True,
        help="Output root for fold directories (e.g. /content/src/Japan_kfold)"
    )
    parser.add_argument("--n-folds", type=int, default=10, help="Number of folds (default 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument(
        "--class-names-yaml", default=None,
        help="Optional path to a YAML with `names:` (uses configs/dataset.yaml if omitted)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Resolve class names.
    if args.class_names_yaml:
        with open(args.class_names_yaml) as f:
            class_names = yaml.safe_load(f).get("names", {})
    else:
        # Default: RDD2022 Japan classes.
        class_names = {
            0: "D00", 1: "D01", 2: "D10", 3: "D11",
            4: "D20", 5: "D40", 6: "D43", 7: "D44", 8: "D50",
        }

    print(f"Loading entries from:\n  images: {args.images_dir}\n  labels: {args.labels_dir}")
    entries = load_image_labels(args.images_dir, args.labels_dir)
    print(f"Loaded {len(entries)} image/label pairs.")

    # Print per-class image counts for sanity.
    per_class_images = defaultdict(int)
    for _, _, cids in entries:
        for c in set(cids):
            per_class_images[c] += 1
    print("\nPer-class image counts (images containing each class):")
    for c in sorted(per_class_images):
        print(f"  Class {c} ({class_names.get(c, '?')}): {per_class_images[c]}")

    print(f"\nGenerating {args.n_folds} stratified folds (seed={args.seed})...")
    fold_val_indices = stratified_split(entries, args.n_folds, args.seed)

    os.makedirs(args.output_root, exist_ok=True)
    print(f"\nWriting fold directories to: {args.output_root}")
    for k, val_idx in enumerate(fold_val_indices):
        yaml_path = write_fold(k, val_idx, entries, args.output_root, class_names)
        n_val = len(val_idx)
        n_train = len(entries) - n_val
        print(f"  fold{k}: train={n_train}, val={n_val} -> {yaml_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
