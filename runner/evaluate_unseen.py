"""
runner/evaluate_unseen.py

Evaluate the FINAL trained model on unseen-country datasets.

Evaluations are run independently for:
  - India
  - China

No training occurs here.
"""

import argparse
import csv
from pathlib import Path

import yaml
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_metrics(results):
    def get(attr, fallback=-1.0):
        try:
            val = getattr(results, attr, None)
            if val is None:
                val = results.results_dict.get(attr, fallback)
            return float(val)
        except Exception:
            return fallback

    precision = get("metrics/precision(B)")
    recall = get("metrics/recall(B)")
    map50 = get("metrics/mAP50(B)")
    map5095 = get("metrics/mAP50-95(B)")

    if precision > 0 and recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = -1.0

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "map50": round(map50, 6),
        "map5095": round(map5095, 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate final model on unseen datasets."
    )

    parser.add_argument(
        "--config",
        default="configs/final_eval.yaml",
    )

    parser.add_argument(
        "--weights",
        required=True,
    )

    parser.add_argument(
        "--results-dir",
        required=True,
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    model = YOLO(args.weights)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for dataset_name in ["india", "china"]:

        dataset_yaml = cfg["datasets"][dataset_name]["dataset_yaml"]

        print(f"\nEvaluating on {dataset_name}...")

        results = model.val(
            data=dataset_yaml,
            plots=False,
            verbose=True,
        )

        metrics = extract_metrics(results)

        row = {
            "dataset": dataset_name,
            **metrics,
        }

        rows.append(row)

        print(metrics)

    out_csv = results_dir / "unseen_results.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "precision",
                "recall",
                "f1",
                "map50",
                "map5095",
            ],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"\nSaved results to: {out_csv}")


if __name__ == "__main__":
    main()