"""
runner/train_final.py

Train one FINAL alpha-blended-eaf model using the selected
hyperparameters from the sweep stages.

Unlike run_sweep.py:
  - no cross-validation
  - no parameter sweeps
  - trains exactly ONE model
  - intended for final evaluation on unseen datasets

Output layout
-------------
final_runs/
  final_eps{epsilon}_r{r}/
    train/
      weights/
        best.pt
        last.pt
    results.csv
    args.yaml
"""

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(data: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train final alpha-blended-eaf model."
    )

    parser.add_argument(
        "--config",
        default="configs/final_eval.yaml",
    )

    parser.add_argument(
        "--results-dir",
        required=True,
    )

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    training_cfg = cfg["training"]
    japan_cfg = cfg["datasets"]["japan"]

    epsilon = training_cfg["epsilon"]
    r = training_cfg["r"]

    results_root = Path(args.results_dir) / "final_runs"
    run_name = f"final_eps{epsilon}_r{r}"

    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Build per-run model YAML
    # -----------------------------------------------------------------------

    repo_root = Path(args.repo_root)

    base_model_yaml = load_yaml(
        str(repo_root / "configs" / "model_eaf.yaml")
    )

    base_model_yaml["activation"] = (
        f"AlphaBlendedEAFReLU(epsilon={epsilon}, r={r})"
    )

    run_model_yaml = run_dir / "model_run.yaml"

    write_yaml(base_model_yaml, str(run_model_yaml))

    # -----------------------------------------------------------------------
    # Save args for reproducibility
    # -----------------------------------------------------------------------

    args_record = {
        "epsilon": epsilon,
        "r": r,
        "epochs": training_cfg["epochs"],
        "batch": training_cfg["batch"],
        "imgsz": training_cfg["imgsz"],
        "optimizer": training_cfg["optimizer"],
        "lr0": training_cfg["lr0"],
        "warmup_epochs": training_cfg["warmup_epochs"],
        "amp": training_cfg["amp"],
        "pretrained": training_cfg["pretrained"],
    }

    write_yaml(args_record, str(run_dir / "args.yaml"))

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------

    model = YOLO(str(run_model_yaml))

    model.train(
        data=japan_cfg["dataset_yaml"],
        epochs=training_cfg["epochs"],
        batch=training_cfg["batch"],
        imgsz=training_cfg["imgsz"],
        optimizer=training_cfg["optimizer"],
        lr0=training_cfg["lr0"],
        warmup_epochs=training_cfg["warmup_epochs"],
        amp=training_cfg["amp"],
        pretrained=training_cfg["pretrained"],
        workers=training_cfg["workers"],
        project=str(run_dir),
        name="train",
        exist_ok=True,
        seed=0,
        plots=False,
        save=True,
        verbose=True,
    )

    yolo_results_csv = run_dir / "train" / "results.csv"

    if yolo_results_csv.exists():
        shutil.copy(
            yolo_results_csv,
            run_dir / "results.csv",
        )

    print("\nTraining complete.")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()