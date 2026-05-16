"""Custom activation functions for YOLOv8 training experiments.

This module provides the Alpha-Blended Einstein Activation Function (EAF), a
piecewise activation that smoothly bridges the base Einstein function and ReLU
through a transition zone of half-width epsilon around x = 0.

The function is designed to combine the saturating, bounded response of the
Einstein function on negative inputs with the clean linear pass-through of
ReLU on positive inputs, while keeping the derivative continuous at the
boundary.

Mathematical form
-----------------
                | EinsteinBase(x, r),                       if x < -epsilon
    EAF(x) =    | alpha*ReLU(x) + (1-alpha)*EinsteinBase(x), if -epsilon <= x <= epsilon
                | ReLU(x),                                  if x > +epsilon

where  alpha = (x + epsilon) / (2 * epsilon)
       EinsteinBase(x, r) = n * tanh( ((1 - r) / (1 + r)) * tan(x / n) )

Hyperparameters tested in the sweep
-----------------------------------
    epsilon : {0.1, 0.2, 0.3, 0.4}  -- transition zone half-width
    r       : {0.3, 0.4, 0.5, 0.6}  -- Einstein velocity-ratio parameter
    n_val   : 64.0 (held constant)  -- saturation scale
"""

import torch
import torch.nn as nn


class EinsteinActivationFunction(nn.Module):
    """Base Einstein activation function.

    F(x) = n * tanh( ((1 - r) / (1 + r)) * tan(x / n) )

    This is the "unblended" core function. It is retained as a separate module
    so it can be imported and used independently for ablation studies.
    """

    def __init__(self, n_val: float = 64.0, r_val: float = 0.6):
        super().__init__()
        if r_val == -1.0:
            raise ValueError("Parameter 'r_val' cannot be -1.0 (division by zero in (1-r)/(1+r)).")
        self.register_buffer("const_n_val", torch.tensor(n_val, dtype=torch.float32))
        factor = (1.0 - r_val) / (1.0 + r_val)
        self.register_buffer("const_factor", torch.tensor(factor, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_div_n = x / self.const_n_val
        tan_x_div_n = torch.tan(x_div_n)
        # nan_to_num always runs (no GPU-CPU sync from torch.all/torch.any).
        tan_x_div_n = torch.nan_to_num(tan_x_div_n, nan=0.0, posinf=1e8, neginf=-1e8)
        return self.const_n_val * torch.tanh(self.const_factor * tan_x_div_n)


class AlphaBlendedEAFReLU(nn.Module):
    """Piecewise alpha-blended Einstein + ReLU activation.

    Three zones (controlled by epsilon):
        - Far negative   (x < -epsilon): pure Einstein
        - Transition     (-epsilon <= x <= +epsilon): linear blend
        - Far positive   (x > +epsilon): pure ReLU

    Both epsilon and r are exposed as constructor arguments so the cross-
    validated sweep can vary them independently.

    Implementation notes
    --------------------
    - Uses torch.where instead of masked indexing for GPU efficiency and
      autograd cleanliness.
    - No torch.any guards (which would cause GPU-CPU synchronization on every
      forward pass).
    - Both branches are computed densely; torch.where picks the right one.
    """

    def __init__(self, epsilon: float = 0.1, r: float = 0.6, n_val: float = 64.0):
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError(f"Parameter 'epsilon' must be > 0, got {epsilon}.")
        self.epsilon = float(epsilon)
        self.r = float(r)
        self.n_val = float(n_val)
        self.eaf = EinsteinActivationFunction(n_val=n_val, r_val=r)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eaf_out = self.eaf(x)
        relu_out = self.relu(x)
        # Position-dependent alpha within the transition zone, clamped to [0, 1].
        alpha = torch.clamp((x + self.epsilon) / (2.0 * self.epsilon), min=0.0, max=1.0)
        blended = alpha * relu_out + (1.0 - alpha) * eaf_out
        # Far-negative zone: pure EAF.
        out = torch.where(x < -self.epsilon, eaf_out, blended)
        # Far-positive zone: pure ReLU.
        out = torch.where(x > self.epsilon, relu_out, out)
        return out

    def extra_repr(self) -> str:
        return f"epsilon={self.epsilon}, r={self.r}, n_val={self.n_val}"
