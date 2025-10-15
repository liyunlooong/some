#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import cv2


COLOR_MIN = -1.0
COLOR_MAX = 1.0


def set_color_bounds(min_value: float, max_value: float = 1.0) -> None:
    """Configure the global color bounds used throughout the pipeline."""

    global COLOR_MIN, COLOR_MAX

    max_value = min(1.0, float(max_value))
    if max_value <= -1.0:
        raise ValueError("Maximum color bound must be greater than -1.0.")

    min_value = float(min_value)
    # Clamp to the supported [-1, 0] interval while allowing callers to shrink the range.
    min_value = max(-1.0, min(0.0, min_value))
    if min_value >= max_value:
        raise ValueError("Minimum color bound must be smaller than the maximum bound.")

    COLOR_MIN = min_value
    COLOR_MAX = max_value


def get_color_bounds():
    return COLOR_MIN, COLOR_MAX


def clamp_colors(value: torch.Tensor) -> torch.Tensor:
    min_v, max_v = get_color_bounds()
    return value.clamp(min=min_v, max=max_v)


def convert_uint8_to_color(value):
    min_v, max_v = get_color_bounds()
    scale = (max_v - min_v) / 255.0
    if torch.is_tensor(value):
        return value.to(dtype=torch.float32) * scale + min_v
    return value.astype(np.float32) * scale + min_v


def convert_color_to_uint8(value):
    min_v, max_v = get_color_bounds()
    if max_v <= min_v:
        raise ValueError("Color bounds must span a positive range.")
    scale = 255.0 / (max_v - min_v)
    if torch.is_tensor(value):
        return ((value.to(dtype=torch.float32) - min_v) * scale).clamp(0.0, 255.0)
    return np.clip((value.astype(np.float32) - min_v) * scale, 0.0, 255.0)


def inverse_tanh(x: torch.Tensor) -> torch.Tensor:
    """Compute the inverse hyperbolic tangent with clamping for numerical stability."""

    eps = torch.finfo(x.dtype).eps if torch.is_floating_point(x) else 1e-6
    clamped = x.clamp(min=-1 + eps, max=1 - eps)
    return torch.atanh(clamped)

def PILtoTorch(pil_image, resolution):
    # resized_image_PIL = cv2.resize(pil_image, resolution)
    resized_image_PIL = pil_image.resize(resolution)
    np_image = np.array(resized_image_PIL)
    tensor_image = torch.from_numpy(np_image).float()

    if tensor_image.ndim == 3:
        if tensor_image.shape[2] == 4:
            rgb = convert_uint8_to_color(tensor_image[..., :3])
            alpha = tensor_image[..., 3:4] / 255.0
            resized_image = torch.cat([rgb, alpha], dim=2)
        else:
            resized_image = convert_uint8_to_color(tensor_image)
    else:
        resized_image = convert_uint8_to_color(tensor_image.unsqueeze(dim=-1))

    return resized_image.permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
