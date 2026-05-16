"""runner/run_sweep.py
Stage 1 sweep runner for the alpha-blended-eaf cross-validation experiment.

Runs 4 configurations (r = 0.3, 0.4, 0.5, 0.6) with epsilon fixed at 0.1,
each evaluated across 10 stratified folds. Total: 40 training runs.

Usage
-----
    python runner/run_sweep.py \\
        --sweep-config  configs/sweep_configs.yaml \\
        --kfold-dir     /content/src/Japan_kfold \\
        --results-dir   /content/drive/MyDrive/Dissertation_Results/Results \\
        --repo-root     /content/alpha-blended-eaf

    # Dry run: 1 config x 1 fold x 2 epochs (completes in ~10 minutes)
    python runner/run_sweep.py ... --dry-run

Resumability
------------
Each completed (config, fold) pair writes a `done.txt` marker to its
results subfolder. On restart, the script skips any pair that already has
a `done.txt`. Rerunning the script after a Colab disconnect is therefore safe:
it picks up exactly where the previous session left off.

Fail behaviour
--------------
If a fold crashes, the exception propagates immediately and the script exits.
This is intentional (fail-fast). The crashed fold will NOT have a `done.txt`,
so the next session will retry it from scratch.

Output layout (under --results-dir)
-------------------------------------
sweep_runs/
  stage1_eps{epsilon}_r{r}/
    fold{k}/
      results.csv      <- YOLO per-epoch metrics
      args.yaml        <- exact training arguments used
      done.txt         <- written on successful completion
    fold_metrics.csv   <- one row per fold (P, R, F1, mAP@0.5, mAP@0.5:0.95)
  all_runs.csv         <- every fold across every config, appended per fold
"""

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(data: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def config_name(epsilon: float, r: float) -> str:
    return f"stage1_eps{epsilon}_r{r}"


def is_done(fold_dir: Path) -> bool:
    return (fold_dir / "done.txt").exists()


def mark_done(fold_dir: Path) -> None:
    (fold_dir / "done.txt").write_text("ok\n")


def extract_metrics(results) -> dict:
    """Pull the final-epoch metrics from a YOLO training Results object.

    YOLO's results object stores metrics as attributes after training.
    We access them defensively because attribute names can shift slightly
    between minor Ultralytics versions.

    Returns a dict with keys: precision, recall, f1, map50, map5095.
    All values are floats, defaulting to -1.0 if the attribute is missing.
    """
    def get(attr, fallback=-1.0):
        try:
            val = getattr(results, attr, None)
            if val is None:
                # Some versions nest metrics under results.results_dict
                val = results.results_dict.get(attr, fallback)
            return float(val)
        except Exception:
            return fallback

    # Ultralytics stores precision/recall under metrics/box sub-objects
    # depending on version. We try both spellings.
    precision  = get("metrics/precision(B)")  or get("box/precision")  or get("precision",  -1.0)
    recall     = get("metrics/recall(B)")     or get("box/recall")     or get("recall",     -1.0)
    map50      = get("metrics/mAP50(B)")      or get("box/map50")      or get("map50",      -1.0)
    map5095    = get("metrics/mAP50-95(B)")   or get("box/map")        or get("map5095",    -1.0)

    # F1: derived, not directly stored.
    if precision > 0 and recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = -1.0

    return {
        "precision": round(precision, 6),
        "recall":    round(recall,    6),
        "f1":        round(f1,        6),
        "map50":     round(map50,     6),
        "map5095":   round(map5095,   6),
    }


def append_to_all_runs_csv(all_runs_csv: Path, row: dict) -> None:
    """Append one row to the global all_runs.csv, creating the file if needed."""
    fieldnames = [
        "config_name", "epsilon", "r", "fold",
        "precision", "recall", "f1", "map50", "map5095",
        "training_time_sec",
    ]
    write_header = not all_runs_csv.exists()
    with open(all_runs_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_fold_metrics_csv(config_dir: Path, fold_rows: list) -> None:
    """Write (or overwrite) the per-config fold_metrics.csv with μ ± σ summary."""
    if not fold_rows:
        return

    import statistics

    metrics = ["precision", "recall", "f1", "map50", "map5095"]
    fold_csv = config_dir / "fold_metrics.csv"

    fieldnames = ["fold"] + metrics + ["training_time_sec"]
    with open(fold_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in fold_rows:
            writer.writerow({k: row[k] for k in fieldnames})

        # Summary rows: mean and std across completed folds.
        completed = [r for r in fold_rows if r["precision"] >= 0]
        if len(completed) >= 2:
            mean_row = {"fold": "MEAN"}
            std_row  = {"fold": "STD"}
            for m in metrics:
                vals = [r[m] for r in completed if r[m] >= 0]
                mean_row[m] = round(statistics.mean(vals), 6) if vals else -1.0
                std_row[m]  = round(statistics.stdev(vals), 6) if len(vals) >= 2 else 0.0
            mean_row["training_time_sec"] = round(
                statistics.mean(r["training_time_sec"] for r in completed), 1
            )
            std_row["training_time_sec"] = round(
                statistics.stdev(r["training_time_sec"] for r in completed), 1
            ) if len(completed) >= 2 else 0.0
            writer.writerow(mean_row)
            writer.writerow(std_row)

    print(f"    Fold metrics written to: {fold_csv}")


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_one_fold(
    fold_idx: int,
    epsilon: float,
    r: float,
    kfold_dir: Path,
    fold_results_dir: Path,
    repo_root: Path,
    training_cfg: dict,
    dry_run: bool,
) -> dict:
    """Train one fold and return the extracted metrics dict.

    Generates a per-run model YAML with the correct activation expression,
    trains via the Ultralytics Python API, extracts metrics, then deletes
    weights to save Drive space.

    Raises on any training failure (fail-fast behaviour).
    """
    from ultralytics import YOLO

    # --- Build a per-run model YAML in a temp location ---
    base_model_yaml = load_yaml(str(repo_root / "configs" / "model_eaf.yaml"))
    base_model_yaml["activation"] = f"AlphaBlendedEAFReLU(epsilon={epsilon}, r={r})"

    run_model_yaml = fold_results_dir / "model_run.yaml"
    write_yaml(base_model_yaml, str(run_model_yaml))

    # --- Dataset YAML for this fold ---
    dataset_yaml = kfold_dir / f"dataset_fold{fold_idx}.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}\n"
            f"Did kfold_split.py run successfully?"
        )

    # --- Training arguments ---
    epochs = 2   if dry_run else training_cfg.get("epochs",       12)
    batch  = 2   if dry_run else training_cfg.get("batch",         4)
    imgsz  =     training_cfg.get("imgsz",       640)
    lr0    =     training_cfg.get("lr0",         1e-3)
    warmup =     0 if dry_run else training_cfg.get("warmup_epochs", 3.0)

    # Save the exact args used — useful for auditing and reproducibility.
    args_record = {
        "epsilon":        epsilon,
        "r":              r,
        "fold":           fold_idx,
        "epochs":         epochs,
        "batch":          batch,
        "imgsz":          imgsz,
        "optimizer":      training_cfg.get("optimizer",    "AdamW"),
        "lr0":            lr0,
        "warmup_epochs":  warmup,
        "amp":            training_cfg.get("amp",          False),
        "pretrained":     training_cfg.get("pretrained",   False),
        "seed":           0,
        "dry_run":        dry_run,
    }
    write_yaml(args_record, str(fold_results_dir / "args.yaml"))

    # --- Train ---
    model = YOLO(str(run_model_yaml))
    t0 = time.time()

    results = model.train(
        data          = str(dataset_yaml),
        epochs        = epochs,
        batch         = batch,
        imgsz         = imgsz,
        optimizer     = training_cfg.get("optimizer",   "AdamW"),
        lr0           = lr0,
        warmup_epochs = warmup,
        amp           = training_cfg.get("amp",         False),
        pretrained    = training_cfg.get("pretrained",  False),
        workers       = training_cfg.get("workers",     16),
        project       = str(fold_results_dir),
        name          = "train",
        exist_ok      = True,
        seed          = 0,
        plots         = False,   # skip plot generation to save time
        save          = True,
        verbose       = False,
    )
    elapsed = round(time.time() - t0, 1)

    # Copy YOLO's results.csv up one level so it's easier to find later.
    yolo_results_csv = fold_results_dir / "train" / "results.csv"
    if yolo_results_csv.exists():
        shutil.copy(yolo_results_csv, fold_results_dir / "results.csv")

    # --- Extract metrics ---
    metrics = extract_metrics(results)
    metrics["training_time_sec"] = elapsed

    # --- Delete weights to save Drive space ---
    weights_dir = fold_results_dir / "train" / "weights"
    if weights_dir.exists():
        shutil.rmtree(weights_dir)
        print(f"    Weights deleted ({weights_dir})")

    return metrics


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------

def run_sweep(args):
    # Load sweep config.
    cfg = load_yaml(args.sweep_config)
    stage1_cfg  = cfg["stage1"]
    training_cfg = cfg["training"]
    n_folds      = cfg.get("n_folds", 10)

    epsilon_fixed = stage1_cfg["fixed"]["epsilon"]
    r_values      = stage1_cfg["varied"]["r"]

    kfold_dir   = Path(args.kfold_dir)
    results_dir = Path(args.results_dir) / "sweep_runs"
    repo_root   = Path(args.repo_root)
    all_runs_csv = Path(args.results_dir) / "sweep_runs" / "all_runs.csv"

    results_dir.mkdir(parents=True, exist_ok=True)

    n_folds_to_run = 1 if args.dry_run else n_folds
    r_values_to_run = [r_values[0]] if args.dry_run else r_values

    total_planned = len(r_values_to_run) * n_folds_to_run
    total_done    = 0
    total_skipped = 0

    print("=" * 64)
    print(f"alpha-blended-eaf  |  Stage 1 Sweep")
    print(f"  epsilon (fixed):  {epsilon_fixed}")
    print(f"  r values:         {r_values_to_run}")
    print(f"  folds:            {n_folds_to_run}")
    print(f"  planned runs:     {total_planned}")
    print(f"  dry run:          {args.dry_run}")
    print(f"  results root:     {results_dir}")
    print("=" * 64)

    for r in r_values_to_run:
        cname      = config_name(epsilon_fixed, r)
        config_dir = results_dir / cname
        config_dir.mkdir(parents=True, exist_ok=True)

        fold_rows = []

        # Pre-populate fold_rows with any already-completed folds.
        for k in range(n_folds_to_run):
            fold_dir = config_dir / f"fold{k}"
            if is_done(fold_dir):
                # Try to read back saved metrics for the summary CSV.
                saved_args = fold_dir / "args.yaml"
                saved_metrics_row = {"fold": k, "precision": -1.0, "recall": -1.0,
                                     "f1": -1.0, "map50": -1.0, "map5095": -1.0,
                                     "training_time_sec": -1.0}
                # Check if there's a row in all_runs.csv already.
                if all_runs_csv.exists():
                    with open(all_runs_csv) as f:
                        for row in csv.DictReader(f):
                            if row["config_name"] == cname and int(row["fold"]) == k:
                                saved_metrics_row = {
                                    "fold":               k,
                                    "precision":          float(row["precision"]),
                                    "recall":             float(row["recall"]),
                                    "f1":                 float(row["f1"]),
                                    "map50":              float(row["map50"]),
                                    "map5095":            float(row["map5095"]),
                                    "training_time_sec":  float(row["training_time_sec"]),
                                }
                fold_rows.append(saved_metrics_row)

        print(f"\nConfig: {cname}")

        for k in range(n_folds_to_run):
            fold_dir = config_dir / f"fold{k}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            if is_done(fold_dir):
                total_skipped += 1
                print(f"  Fold {k:2d}/{n_folds_to_run-1}  [SKIPPED — already done]")
                continue

            run_label = f"  Fold {k:2d}/{n_folds_to_run-1}"
            print(f"{run_label}  [TRAINING]  r={r}  eps={epsilon_fixed}")

            metrics = train_one_fold(
                fold_idx         = k,
                epsilon          = epsilon_fixed,
                r                = r,
                kfold_dir        = kfold_dir,
                fold_results_dir = fold_dir,
                repo_root        = repo_root,
                training_cfg     = training_cfg,
                dry_run          = args.dry_run,
            )

            # Record results.
            all_runs_row = {
                "config_name":        cname,
                "epsilon":            epsilon_fixed,
                "r":                  r,
                "fold":               k,
                "precision":          metrics["precision"],
                "recall":             metrics["recall"],
                "f1":                 metrics["f1"],
                "map50":              metrics["map50"],
                "map5095":            metrics["map5095"],
                "training_time_sec":  metrics["training_time_sec"],
            }
            append_to_all_runs_csv(all_runs_csv, all_runs_row)
            mark_done(fold_dir)

            # Update fold_rows (replace pre-populated placeholder if present).
            fold_rows = [r2 for r2 in fold_rows if r2["fold"] != k]
            fold_rows.append({"fold": k, **metrics})

            total_done += 1
            t_sec = metrics["training_time_sec"]
            remaining = (total_planned - total_done - total_skipped)
            eta_min = round(remaining * t_sec / 60, 1) if t_sec > 0 else "?"
            print(
                f"    P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
                f"  F1={metrics['f1']:.4f}  mAP50={metrics['map50']:.4f}"
                f"  mAP50-95={metrics['map5095']:.4f}"
                f"  time={t_sec}s  ETA≈{eta_min}min"
            )

        # Write per-config summary CSV after all folds for this config.
        fold_rows_sorted = sorted(fold_rows, key=lambda x: x["fold"])
        write_fold_metrics_csv(config_dir, fold_rows_sorted)

    print("\n" + "=" * 64)
    print(f"Stage 1 sweep complete.")
    print(f"  Runs completed this session: {total_done}")
    print(f"  Runs skipped (already done): {total_skipped}")
    print(f"  All-runs CSV: {all_runs_csv}")
    print("=" * 64)

    if not args.dry_run:
        print("\nNext step: review all_runs.csv to determine the winning r value,")
        print("then set stage2.fixed.r in configs/sweep_configs.yaml before")
        print("building and running the Stage 2 sweep.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 sweep runner for alpha-blended-eaf experiments."
    )
    parser.add_argument(
        "--sweep-config",
        default="configs/sweep_configs.yaml",
        help="Path to sweep_configs.yaml (default: configs/sweep_configs.yaml)",
    )
    parser.add_argument(
        "--kfold-dir",
        required=True,
        help="Directory containing fold-specific YAMLs and image symlinks "
             "(e.g. /content/src/Japan_kfold)",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Root directory for results (e.g. "
             "/content/drive/MyDrive/Dissertation_Results/Results)",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Root of the alpha-blended-eaf repo (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 config x 1 fold x 2 epochs to verify the pipeline end-to-end.",
    )
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
