"""
runner/aggregate_results.py

Aggregate cross-validation sweep results and generate summary tables/plots.

Input:
  sweep_runs/all_runs.csv

Outputs:
  summary_by_config.csv
  map_vs_f1_by_config.png
  per_fold_metrics.png

Notes:
  Object detection does not normally use classification accuracy.
  This script reports:
    - precision
    - recall
    - F1
    - mAP@0.5
    - mAP@0.5:0.95
"""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    "precision",
    "recall",
    "f1",
    "map50",
    "map5095",
]


def read_rows(runs_csv: Path) -> list[dict]:
    if not runs_csv.exists():
        raise FileNotFoundError(f"Runs CSV not found: {runs_csv}")

    rows = []

    with open(runs_csv) as f:
        reader = csv.DictReader(f)

        for row in reader:
            parsed = {
                "config_name": row["config_name"],
                "epsilon": float(row["epsilon"]),
                "r": float(row["r"]),
                "fold": int(row["fold"]),
                "training_time_sec": float(row["training_time_sec"]),
            }

            for metric in METRICS:
                parsed[metric] = float(row[metric])

            rows.append(parsed)

    if not rows:
        raise ValueError(f"No rows found in {runs_csv}")

    return rows


def group_by_config(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = {}

    for row in rows:
        grouped.setdefault(row["config_name"], []).append(row)

    return grouped


def mean_std(values: list[float]) -> tuple[float, float]:
    valid = [v for v in values if v >= 0]

    if not valid:
        return -1.0, -1.0

    mean = statistics.mean(valid)
    std = statistics.stdev(valid) if len(valid) >= 2 else 0.0

    return round(mean, 6), round(std, 6)


def write_summary(grouped: dict[str, list[dict]], output_csv: Path) -> list[dict]:
    summary_rows = []

    for config_name, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda x: x["fold"])

        summary = {
            "config_name": config_name,
            "epsilon": rows_sorted[0]["epsilon"],
            "r": rows_sorted[0]["r"],
            "n_folds": len(rows_sorted),
        }

        for metric in METRICS:
            mean, std = mean_std([row[metric] for row in rows_sorted])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std

        time_mean, time_std = mean_std(
            [row["training_time_sec"] for row in rows_sorted]
        )

        summary["training_time_sec_mean"] = round(time_mean, 1)
        summary["training_time_sec_std"] = round(time_std, 1)

        summary_rows.append(summary)

    summary_rows = sorted(
        summary_rows,
        key=lambda x: (
            x["map50_mean"],
            x["f1_mean"],
        ),
        reverse=True,
    )

    fieldnames = [
        "config_name",
        "epsilon",
        "r",
        "n_folds",
    ]

    for metric in METRICS:
        fieldnames.append(f"{metric}_mean")
        fieldnames.append(f"{metric}_std")

    fieldnames.extend([
        "training_time_sec_mean",
        "training_time_sec_std",
    ])

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    return summary_rows


def plot_map_vs_f1(summary_rows: list[dict], output_png: Path) -> None:
    labels = [row["config_name"] for row in summary_rows]
    map50 = [row["map50_mean"] for row in summary_rows]
    f1 = [row["f1_mean"] for row in summary_rows]

    x = list(range(len(labels)))

    plt.figure(figsize=(max(10, len(labels) * 1.2), 6))
    plt.plot(x, map50, marker="o", label="mAP@0.5")
    plt.plot(x, f1, marker="o", label="F1")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Configuration")
    plt.ylabel("Score")
    plt.title("Mean mAP@0.5 vs F1 by Configuration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def plot_per_fold_metrics(grouped: dict[str, list[dict]], output_png: Path) -> None:
    plt.figure(figsize=(12, 7))

    for config_name, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda x: x["fold"])
        folds = [row["fold"] for row in rows_sorted]
        map50 = [row["map50"] for row in rows_sorted]

        plt.plot(
            folds,
            map50,
            marker="o",
            label=config_name,
        )

    plt.xlabel("Fold")
    plt.ylabel("mAP@0.5")
    plt.title("Per-fold mAP@0.5 by Configuration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def plot_metric_bars(summary_rows: list[dict], output_png: Path) -> None:
    labels = [row["config_name"] for row in summary_rows]
    x = list(range(len(labels)))

    plt.figure(figsize=(max(10, len(labels) * 1.2), 6))

    for metric in ["precision", "recall", "f1", "map50", "map5095"]:
        values = [row[f"{metric}_mean"] for row in summary_rows]
        plt.plot(x, values, marker="o", label=metric)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Configuration")
    plt.ylabel("Mean score")
    plt.title("Mean Metrics by Configuration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate sweep results and generate summary plots."
    )

    parser.add_argument(
        "--runs-csv",
        required=True,
        help="Path to sweep_runs/all_runs.csv",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary CSV and plots will be written.",
    )

    args = parser.parse_args()

    runs_csv = Path(args.runs_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(runs_csv)
    grouped = group_by_config(rows)

    summary_csv = output_dir / "summary_by_config.csv"
    map_vs_f1_png = output_dir / "map_vs_f1_by_config.png"
    per_fold_png = output_dir / "per_fold_map50_by_config.png"
    metrics_png = output_dir / "mean_metrics_by_config.png"

    summary_rows = write_summary(grouped, summary_csv)
    plot_map_vs_f1(summary_rows, map_vs_f1_png)
    plot_per_fold_metrics(grouped, per_fold_png)
    plot_metric_bars(summary_rows, metrics_png)

    best = summary_rows[0]

    print("\nAggregation complete.")
    print(f"Summary CSV: {summary_csv}")
    print(f"mAP vs F1 plot: {map_vs_f1_png}")
    print(f"Per-fold mAP plot: {per_fold_png}")
    print(f"Mean metrics plot: {metrics_png}")

    print("\nBest configuration by mAP@0.5 then F1:")
    print(f"  config: {best['config_name']}")
    print(f"  epsilon: {best['epsilon']}")
    print(f"  r: {best['r']}")
    print(f"  mAP@0.5 mean: {best['map50_mean']}")
    print(f"  F1 mean: {best['f1_mean']}")


if __name__ == "__main__":
    main()