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
| `colab_run.sh` | Single entry-point bash script for Colab |
| `notebooks/EAF_RDD2022_Colab.ipynb` | One-cell Colab notebook that runs the pipeline |

## The activation function

```
              | EinsteinBase(x, r),                          if x < -ε
EAF(x)  =     | α·ReLU(x) + (1-α)·EinsteinBase(x),           if -ε ≤ x ≤ +ε
              | ReLU(x),                                     if x > +ε
```

where  α = (x + ε) / (2ε), and EinsteinBase(x, r) = n · tanh( ((1-r)/(1+r)) · tan(x/n) ) with n = 64.

The hyperparameter sweep follows a two-stage protocol (Stage 1 varies r with ε fixed; Stage 2 varies ε with r fixed at the Stage 1 winner), each configuration evaluated with 10-fold stratified cross-validation.

## Running on Colab

1. Open `notebooks/EAF_RDD2022_Colab.ipynb` in Colab.
2. Set the runtime to GPU.
3. Update the `git clone` URL in the second cell to point to your fork of this repo.
4. Run the cells.

The script is resumable — Colab session limits won't lose your progress.

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
