# alpha-blended-eaf

Cross-validated evaluation of the **Alpha-Blended Einstein Activation Function** on the [RDD2022](https://github.com/sekilab/RoadDamageDetector) Japan road-damage dataset, using YOLOv8s as the detector.

This repository accompanies an ongoing research project; results and the paper draft will be linked here once available.

## What's here

| Path | Purpose |
|---|---|
| `activation/custom_activations.py` | The `AlphaBlendedEAFReLU` PyTorch module |
| `patches/apply_patch.sh` | Idempotent script that integrates the activation into a cloned Ultralytics repo |
| `configs/` | Model, dataset, and sweep YAMLs |
| `data_prep/xml_parser.py` | Converts RDD2022 VOC annotations to YOLO format |
| `data_prep/kfold_split.py` | Generates stratified 10-fold splits with a fixed seed |
| `runner/verify_patch.py` | End-to-end sanity check that the activation injection actually works |
| `runner/run_sweep.py` | Cross-validated sweep runner for Stage 1 and Stage 2 experiments |
| `runner/train_final.py` | Trains the final selected configuration on the full Japan dataset |
| `runner/evaluate_unseen.py` | Evaluates the final model on unseen India and China datasets |
| `runner/aggregate_results.py` | Generates summary tables and plots from sweep results |
| `colab_run.sh` | Single entry-point bash script for Colab |
| `notebooks/EAF_RDD2022_Colab.ipynb` | One-cell Colab notebook that runs the pipeline |

## The activation function

```
              | EinsteinBase(x, r),                          if x < -ε
EAF(x)  =     | α·ReLU(x) + (1-α)·EinsteinBase(x),           if -ε ≤ x ≤ +ε
              | ReLU(x),                                     if x > +ε
```

where  α = (x + ε) / (2ε), and EinsteinBase(x, r) = n · tanh( ((1-r)/(1+r)) · tan(x/n) ) with n = 64.

The hyperparameter search follows a two-stage protocol:
- Stage 1 varies r while ε is fixed
- Stage 2 varies ε while r is fixed at the Stage 1 winner

Each configuration is evaluated with 10-fold stratified cross-validation on the RDD2022 Japan dataset using YOLOv8s. The final selected configuration is then trained on the full Japan dataset and evaluated on unseen India and China datasets.

## Running on Colab

1. Open `notebooks/EAF_RDD2022_Colab.ipynb` in Colab.
2. Set the runtime to GPU.
3. Update the `git clone` URL in the second cell to point to your fork of this repo.
4. Run the cells.

The pipeline is resumable. Completed folds write `done.txt` markers, allowing interrupted Colab sessions to continue from the last completed run.

## Running locally

```bash
# Clone this repo and a pinned Ultralytics in parallel.
git clone https://github.com/YOUR_USERNAME/alpha-blended-eaf.git
git clone --branch v8.4.50 --depth 1 https://github.com/ultralytics/ultralytics.git

# Apply the surgical patch.
bash alpha-blended-eaf/patches/apply_patch.sh \
    "$(pwd)/alpha-blended-eaf" \
    "$(pwd)/ultralytics"

# Install dependencies.
pip install -e ./ultralytics iterative-stratification scikit-learn pyyaml

# Verify the patch works.
python alpha-blended-eaf/runner/verify_patch.py \
    --model-yaml alpha-blended-eaf/configs/model_eaf.yaml

# Run Stage 1 sweep
python alpha-blended-eaf/runner/run_sweep.py \
    --stage stage1 \
    --sweep-config alpha-blended-eaf/configs/sweep_configs.yaml \
    --kfold-dir /path/to/Japan_kfold \
    --results-dir /path/to/results \
    --repo-root alpha-blended-eaf

# Aggregate results
python alpha-blended-eaf/runner/aggregate_results.py \
    --runs-csv /path/to/results/sweep_runs/all_runs.csv \
    --output-dir /path/to/results/sweep_runs
```

## Dependencies

Pinned versions used for the experiments:

| Package | Version |
|---|---|
| Python | ≥3.8 |
| Ultralytics | v8.4.50 |
| PyTorch | ≥1.8 |
| iterative-stratification | latest |
| scikit-learn | latest |

## Reproducibility

- All experiments use seed 42 for k-fold splitting.
- Per-fold weight initialisation is fixed (YOLOv8 `seed=0` passed in the training call).
- Compute environment: Google Colab GPU (T4/V100/A100, varies per session).

## License

AGPL-3.0, matching the upstream Ultralytics license.

## Citation

```bibtex
@misc{alpha_blended_eaf_2026,
  title  = {Alpha-Blended Einstein Activation Function for Road Damage Detection},
  author = {[primary author to be filled in]},
  year   = {2026},
  note   = {Code: https://github.com/YOUR_USERNAME/alpha-blended-eaf},
}
```
