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

import math
import os
import sys
from argparse import ArgumentParser, Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                elif t == list or t == tuple:
                    group.add_argument(
                        "--" + key,
                        ("-" + key[0:1]),
                        nargs="+",
                        type=type(value[0]),
                        default=value,
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list or t == tuple:
                    group.add_argument(
                        "--" + key, nargs="+", type=type(value[0]), default=value
                    )
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cpu"
        self.render_spline = False
        self.use_neural_network = False
        self.eval = False
        self.num_additional_pts = 10000
        self.additional_size_multi = 1.0
        self.num_spline_frames = 480
        self.glo_latent_dim = 64
        self.max_opacity = 0.99
        self.color_min = -1.0
        self.tmin = 0.2
        self.enable_mip_splatting = False  # Add the missing attribute
        self.low_pass_2d_kernel_size = 3 # Add the missing attribute
        self.low_pass_3d_kernel_size = 3 # Add the missing attribute

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if not hasattr(g, "color_min"):
            setattr(g, "color_min", getattr(args, "color_min", -1.0))
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = True
        self.compute_cov3D_python = False
        self.enable_GLO = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

def _consume_color_min_override(parser: ArgumentParser, extras):
    override = None
    idx = 0
    while idx < len(extras):
        token = extras[idx]
        if token.startswith("--color_min"):
            if token == "--color_min":
                if idx + 1 >= len(extras):
                    parser.error("--color_min expects a value")
                value_token = extras[idx + 1]
                idx += 2
            elif token.startswith("--color_min="):
                value_token = token.split("=", 1)[1]
                idx += 1
            else:
                parser.error("unrecognized arguments: " + " ".join(extras[idx:]))
            try:
                override = float(value_token)
            except ValueError:
                parser.error("--color_min expects a floating point value")
        else:
            parser.error("unrecognized arguments: " + " ".join(extras[idx:]))
    return override


def parse_args_with_color_min(parser: ArgumentParser, argv):
    args, extras = parser.parse_known_args(argv)
    override = _consume_color_min_override(parser, extras)
    if override is not None:
        setattr(args, "color_min", override)
    elif not hasattr(args, "color_min"):
        setattr(args, "color_min", -1.0)
    return args


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000

        self.betas = [0.9, 0.999]

        self.position_lr_final = 4e-7 #0.0000004
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.position_lr_init = 4e-5 #0.00004

        self.glo_lr = 0.01
        self.glo_network_lr = 0.00005

        self.feature_lr = 0.0025
        self.feature_rest_lr = 0.00025
        self.bg_lr = 0.0
        self.opacity_lr = 0.0125
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.min_opacity = 0.005
        self.min_split_opacity = 0.01
        self.percent_dense = 0.0025
        self.lambda_dssim = 0.2
        self.max_lambda_dssim = 0.8  # Add the missing attribute
        self.max_dssim_iteration = 15000  # Add the missing attribute

        self.lambda_anisotropic = 1e-1
        self.lambda_distortion = 0
        self.sh_up_interval = 2000

        self.densification_interval = 200
        self.opacity_reset_interval = 300000
        self.densify_from_iter = 1500
        self.densify_until_iter = 16_000

        self.densify_grad_threshold: float = 2.5e-7

        self.clone_grad_threshold: float = 1e-1

        self.center_pixel = False
        self.fallback_xy_grad = False

        self.random_background = False
        
        # Bilateral grid parameters
        self.use_bilateral_grid = False
        self.bilateral_grid_shape = [16, 16, 8]
        self.bilateral_grid_lr = 0.003  # Match gsplat's default
        
        # Default to 10.0 (gsplat's value) but keep it configurable
        # Explicitly set as float to ensure command line arguments work with decimal values
        self.lambda_tv: float = 10.0
        
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parse_args_with_color_min(parser, cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
