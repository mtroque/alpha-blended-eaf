"""Verify that the alpha-blended-eaf patch was applied correctly.

Run this AFTER apply_patch.sh and BEFORE launching the sweep. It builds a
YOLOv8s model with AlphaBlendedEAFReLU as the activation and checks that the
Conv layers actually use the custom activation.

Catches the most common silent-failure modes:
  - Patch didn't apply (ImportError)
  - eval() resolved to a different class (wrong type after build)
  - Hyperparameters didn't propagate (epsilon/r values don't match the YAML)

Exit codes:
  0  all checks passed
  1  a check failed
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Verify the activation patch.")
    parser.add_argument(
        "--model-yaml",
        default="configs/model_eaf.yaml",
        help="Path to the model YAML (default: configs/model_eaf.yaml)",
    )
    parser.add_argument(
        "--expected-epsilon", type=float, default=0.1,
        help="Expected epsilon value (should match the YAML)",
    )
    parser.add_argument(
        "--expected-r", type=float, default=0.3,
        help="Expected r value (should match the YAML)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Verifying alpha-blended-eaf integration")
    print("=" * 60)

    # Check 1: import resolves.
    try:
        from ultralytics.nn.modules import AlphaBlendedEAFReLU, EinsteinActivationFunction
        print("[1/4] Import succeeded.")
    except ImportError as e:
        print(f"[1/4] FAIL: ImportError -- {e}")
        print("      The patch did not apply correctly. Re-run apply_patch.sh.")
        return 1

    # Check 2: direct instantiation with kwargs.
    try:
        act = AlphaBlendedEAFReLU(epsilon=args.expected_epsilon, r=args.expected_r)
        print(f"[2/4] Direct instantiation OK: {act}")
    except Exception as e:
        print(f"[2/4] FAIL: cannot instantiate AlphaBlendedEAFReLU -- {e}")
        return 1

    # Check 3: forward pass produces finite values.
    import torch
    try:
        x = torch.linspace(-5, 5, 11)
        y = act(x)
        if not torch.all(torch.isfinite(y)):
            print(f"[3/4] FAIL: non-finite output\n      input:  {x.tolist()}\n      output: {y.tolist()}")
            return 1
        print(f"[3/4] Forward pass OK.")
        print(f"      input:  {[f'{v:+.1f}' for v in x.tolist()]}")
        print(f"      output: {[f'{v:+.3f}' for v in y.tolist()]}")
    except Exception as e:
        print(f"[3/4] FAIL: forward pass crashed -- {e}")
        return 1

    # Check 4: end-to-end model build via YOLO YAML.
    try:
        from ultralytics import YOLO
        model = YOLO(args.model_yaml)
        # Walk the model to find a Conv layer and inspect its activation.
        from ultralytics.nn.modules import Conv
        conv_layer = None
        for m in model.model.modules():
            if isinstance(m, Conv):
                conv_layer = m
                break
        if conv_layer is None:
            print("[4/4] FAIL: no Conv layer found in the built model.")
            return 1
        act_class = type(conv_layer.act).__name__
        if act_class != "AlphaBlendedEAFReLU":
            print(f"[4/4] FAIL: Conv.act is {act_class}, expected AlphaBlendedEAFReLU.")
            print("      The YAML's activation: line may not have been parsed correctly.")
            return 1
        actual_eps = conv_layer.act.epsilon
        actual_r = conv_layer.act.r
        if abs(actual_eps - args.expected_epsilon) > 1e-9 or abs(actual_r - args.expected_r) > 1e-9:
            print(f"[4/4] FAIL: hyperparameter mismatch.")
            print(f"      YAML expected epsilon={args.expected_epsilon}, r={args.expected_r}")
            print(f"      Built  got      epsilon={actual_eps}, r={actual_r}")
            return 1
        print(f"[4/4] Model build OK: Conv[0].act = {conv_layer.act}")
    except Exception as e:
        print(f"[4/4] FAIL: model build crashed -- {e}")
        return 1

    print("=" * 60)
    print("All checks passed. Ready to launch the sweep.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
