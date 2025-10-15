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

import time
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from gaussian_renderer.ever import splinerender
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, set_color_bounds, clamp_colors, convert_color_to_uint8
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, parse_args_with_color_min
from icecream import ic
import random
import math
import cv2
import numpy as np
import torch.nn.functional as F
from typing import List
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, ChainedScheduler

renderFunc = splinerender
# renderFunc = render
from scene.dataset_readers import ProjectionType

from scene.cameras import Camera
from utils.lib_bilagrid import BilateralGrid, total_variation_loss, slice

PREVIEW_RES_FACTOR = 1

def project(xyz, wct):
    p_hom = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device="cuda")], dim=1)
    p_view = (p_hom @ wct)
    pix2d = p_view[:, :2] / p_view[:, 2:3]
    return pix2d, p_view[:, 2]

def inv_project(xy, dist, inv_wvt):
    N = xy.shape[0]
    pad = torch.ones((N, 1), device="cuda")
    p_hom = torch.cat([xy * dist.reshape(-1, 1), dist.reshape(-1, 1), pad], dim=1) @ inv_wvt
    return p_hom[:, :3]

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def set_glo_vector(viewpoint_cam, gaussians, camera_inds):
    camera_ind = camera_inds[viewpoint_cam.uid]
    viewpoint_cam.glo_vector = torch.cat(
        [gaussians.glo[camera_ind], torch.tensor([
                math.log(
                viewpoint_cam.iso * viewpoint_cam.exposure / 1000),
            ], device=gaussians.glo.device)
         ]
    )

def training(dataset : ModelParams, opt : OptimizationParams, pipe : PipelineParams, testing_iterations : List[int], saving_iterations : List[int], checkpoint_iterations : List[int], checkpoint, debug_from):
    set_color_bounds(dataset.color_min, 1.0)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=checkpoint, shuffle=True)
    
    train_cameras = scene.getTrainCameras()
    camera_inds = {}
    camera_inds_back = {}
    for i, camera in enumerate(train_cameras):
        camera_inds[camera.uid] = i
        camera_inds_back[i] = camera.uid
    
    # ----- Initialize bilateral grid if enabled -----
    bil_grids = None
    bil_optimizer = None
    if opt.use_bilateral_grid:
        print("\nInitializing Bilateral Grid:")
        print(f"- Grid shape: {opt.bilateral_grid_shape}")
        print(f"- Learning rate: {opt.bilateral_grid_lr}")
        print(f"- TV loss weight: {opt.lambda_tv}")
        bil_grids = BilateralGrid(
            len(train_cameras),
            grid_X=opt.bilateral_grid_shape[0],
            grid_Y=opt.bilateral_grid_shape[1],
            grid_W=opt.bilateral_grid_shape[2],
        ).to("cuda")
        bil_optimizer = torch.optim.Adam([bil_grids.grids], lr=opt.bilateral_grid_lr, eps=1e-15)
        
        # Create a chained scheduler with warmup like in gsplat
        # First 1000 iterations: linear warmup from 1% to 100% of learning rate
        # Then exponential decay to 1% of initial learning rate by the end of training
        bil_warmup = LinearLR(bil_optimizer, start_factor=0.01, total_iters=1000)
        bil_decay = ExponentialLR(bil_optimizer, gamma=0.01**(1.0/opt.iterations))
        bil_scheduler = ChainedScheduler([bil_warmup, bil_decay])
        
        print(f"- Number of grids: {len(train_cameras)}")
        print("- Using LinearLR warmup + ExponentialLR decay scheduler")
        print("Bilateral Grid initialized successfully!\n")
    # ------------------------------------------------
    
    renderFunc = splinerender
    background = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    # gaussians.load_ply("output/a5911cf7-0/point_cloud/iteration_30000/point_cloud.ply")
    # gaussians.load_ply("/home/amai/Downloads/point_cloud.ply")
    # gaussians.load_ply("/home/amai/3DGS/output/20e2f33c-e/point_cloud/iteration_30000/point_cloud.ply", legacy_compat=True)
    # gaussians.load_ply("/home/amai/gaussian-splatting/output/242678df-0/point_cloud/iteration_30000/point_cloud.ply")
    # gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    gaussians.initialize_glo(len(viewpoint_stack), dataset.glo_latent_dim)

    # opt.densification_interval = len(viewpoint_stack)

    clone_grad_threshold = opt.clone_grad_threshold
    densify_grad_threshold = opt.densify_grad_threshold

    gaussians.training_setup(opt)

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                # custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                custom_cam, do_training, _, _, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    viewpoint_cam = train_cameras[0]
                    set_glo_vector(viewpoint_cam, gaussians, camera_inds)
                    # custom_cam.model = viewpoint_cam.model
                    # custom_cam.distortion_params = viewpoint_cam.distortion_params
                    custom_cam.model=ProjectionType.PERSPECTIVE
                    custom_cam.glo_vector = viewpoint_cam.glo_vector
                    image_width = custom_cam.image_width
                    image_height = custom_cam.image_height
                    custom_cam.image_width = image_width // PREVIEW_RES_FACTOR
                    custom_cam.image_height = image_height // PREVIEW_RES_FACTOR
                    net_image = renderFunc(custom_cam, gaussians, pipe, background, scaling_modifer, random=False, tmin=0)["render"]
                    net_image = convert_color_to_uint8(clamp_colors(net_image)).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                    net_image = cv2.resize(net_image, (image_width, image_height))
                    net_image_bytes = memoryview(net_image)
                network_gui.send(net_image_bytes, dataset.source_path)
                torch.cuda.empty_cache()
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % opt.sh_up_interval == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        set_glo_vector(viewpoint_cam, gaussians, camera_inds)
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = renderFunc(viewpoint_cam, gaussians, pipe, bg, random=not opt.center_pixel)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

       
        #if hasattr(viewpoint_cam, "alpha_mask") and viewpoint_cam.alpha_mask is not None:
        #    alpha_mask = viewpoint_cam.alpha_mask.cuda()
        #    image *= alpha_mask

        if viewpoint_cam.gt_alpha_mask is not None:
            alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            image *= alpha_mask    

        # ----- Apply bilateral grid transformation if enabled -----
        if opt.use_bilateral_grid:
            # torch.cuda.synchronize()
            # Get camera ID for this viewpoint
            camera_id = camera_inds[viewpoint_cam.uid]
            # cv2.imshow('og', image.permute(1, 2, 0).detach().cpu().numpy())
            
            # Reshape image for bilateral grid transformation
            img_for_bil = image.permute(1, 2, 0).reshape(-1, 3)
            
            # Create image IDs tensor (all pixels have same image ID)
            img_ids = torch.full((img_for_bil.shape[0],), camera_id, 
                                  device="cuda", dtype=torch.long)
            
            # Apply bilateral transformation
            st = time.time()
            h, w = image.shape[1], image.shape[2]
            y_coords, x_coords = torch.meshgrid(
                (torch.arange(h, device="cuda") + 0.5) / h,
                (torch.arange(w, device="cuda") + 0.5) / w,
                indexing="ij"
            )
            coords = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
            transformed = slice(bil_grids, coords, img_for_bil, img_ids)
            # transformed = slice_for_single_image(bil_grids.grids, coords, img_for_bil, img_ids)
            # print(transformed.keys(), transformed["rgb"])
            
            # Reshape back to original format
            image = transformed["rgb"].reshape(image.shape[1], image.shape[2], 3).permute(2, 0, 1)
            # torch.cuda.synchronize()
            # cv2.imshow('new guy', image.permute(1, 2, 0).detach().cpu().numpy())
            # cv2.waitKey(1)
            # print(time.time()-st, image.shape)
        # --------------------------------------------------------

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        scaling = gaussians.get_scaling
        anisotropic_loss = ((1-gaussians.get_opacity.detach()).reshape(-1)*((scaling.max(dim=-1).values - scaling.min(dim=-1).values)))[visibility_filter].mean()
        size_loss = (scaling.sqrt()).mean()
        lambda_dssim = opt.lambda_dssim

        fast_loss = size_loss + (1-gaussians.get_opacity).mean()
        
        distortion_loss = render_pkg['distortion_loss'].mean()# if iteration > 2000 else 0
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (
            1.0 - ssim(image, gt_image)
        ).clip(min=0, max=1) + opt.lambda_distortion * distortion_loss + opt.lambda_anisotropic * anisotropic_loss 
        
        # ----- Add total variation loss for bilateral grid if enabled -----
        tvloss = None
        if opt.use_bilateral_grid:
            # Use the configurable lambda_tv parameter (default is 10.0)
            tvloss = opt.lambda_tv * total_variation_loss(bil_grids.grids)
            loss += tvloss
        # --------------------------------------------------------------
        
        if torch.isnan(loss).any():
            print("nan")
            continue
        loss.backward()

        if opt.fallback_xy_grad:
            view = viewpoint_cam

            w = view.image_width  # // 4
            h = view.image_height  # // 4
            fx = 0.5 * w / np.tan(0.5 * view.FoVx)  # original focal length
            fy = 0.5 * h / np.tan(0.5 * view.FoVy)  # original focal length
            K = torch.tensor([
                [fx, 0, w/2, 0],
                [0, fy, h/2, 0],
                [0, 0, 1, 0],
            ], device="cuda").float()
            invK = torch.tensor([
                [1/fx, 0, -w/2/fx],
                [0, 1/fy, -h/2/fy],
                [0, 0, 1],
                [0, 0, 0],
            ], device="cuda").float()
            device = "cuda"
            
            mask = visibility_filter.reshape(-1)
            wct = view.world_view_transform.cuda().float()
            full_wct = torch.eye(4, device="cuda")
            full_wct[:, :3] = wct @ K.T
            inv_wct = torch.linalg.inv(full_wct)
            pix2d, distance = project(gaussians.get_xyz[mask], wct)
            _, (xy_g, d_g, _) = torch.autograd.functional.vjp(inv_project, (pix2d, distance, inv_wct), gaussians.get_xyz.grad[mask])
            viewspace_point_tensor.grad[mask, :2] = xy_g
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_info = {"Loss": f"{ema_loss_for_log:.{7}f}", "Num Prim": gaussians.get_xyz.shape[0], "I": f"{render_pkg['iters'].float().mean():.1f}"}
                if opt.use_bilateral_grid and tvloss is not None:
                    progress_info["BG_TV"] = f"{tvloss.item():.{5}f}"
                progress_bar.set_postfix(progress_info)
                progress_bar.update(10)
                
            # Debug bilateral grid activity
            if opt.use_bilateral_grid and iteration % 1000 == 0:
                current_lr = bil_optimizer.param_groups[0]['lr']
                grid_mean = bil_grids.grids.abs().mean().item()
                grid_std = bil_grids.grids.std().item()
                print(f"[DEBUG] Iter {iteration} - Grid stats: mean={grid_mean:.6f}, std={grid_std:.6f}, LR={current_lr:.6f}")
                
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, renderFunc, (pipe, background), camera_inds, bil_grids, tvloss)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration > opt.densify_from_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter:

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and iteration % opt.opacity_reset_interval > opt.densification_interval:
                    gaussians.densify_and_prune(densify_grad_threshold, opt.min_opacity, scene.cameras_extent, 1000, clone_grad_threshold, opt.min_split_opacity)
                    torch.cuda.empty_cache()
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity(0.005)
                    torch.cuda.empty_cache()
            else:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                    and iteration % opt.opacity_reset_interval > opt.densification_interval
                ):
                    gaussians.update_death_mark()
                    prune_mask = ((gaussians.get_minor_axis_opacity < opt.min_opacity).squeeze())
                    gaussians.prune_points(prune_mask)
                    # print(f"Pruned {prune_mask.sum()} primitives. Mean Opacity: {gaussians.get_opacity.mean()}")
                    torch.cuda.empty_cache()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                # ----- Update bilateral grid if enabled -----
                if opt.use_bilateral_grid:
                    # torch.cuda.synchronize()
                    # st = time.time()
                    bil_optimizer.step()
                    bil_optimizer.zero_grad(set_to_none=True)
                    bil_scheduler.step()
                    # torch.cuda.synchronize()
                    # print(time.time()-st)
                # ------------------------------------------

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, camera_inds, bilateral_grid=None, tvloss=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        
        # Log bilateral grid stats if enabled
        if bilateral_grid is not None and tvloss is not None:
            tb_writer.add_scalar('bilateral_grid/tv_loss', tvloss.item(), iteration)
            # Log histogram of grid values
            tb_writer.add_histogram('bilateral_grid/values', bilateral_grid.grids, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' :
                               [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                for idx in range(5, 30, 5)]
                               })

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    set_glo_vector(viewpoint, scene.gaussians, camera_inds)
                    image = clamp_colors(renderFunc(viewpoint, scene.gaussians, *renderArgs, random=False)["render"])
                    
                    # Apply bilateral grid transformation if enabled
                    if bilateral_grid is not None and config['name'] == 'train':  # Only apply to train views that have matching camera IDs
                        try:
                            camera_id = camera_inds[viewpoint.uid]
                            h, w = image.shape[1], image.shape[2]
                            
                            y_coords, x_coords = torch.meshgrid(
                                torch.linspace(0, 1, h, device="cuda"),
                                torch.linspace(0, 1, w, device="cuda"),
                                indexing='ij'
                            )
                            coords = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
                            
                            img_for_bil = image.permute(1, 2, 0).reshape(-1, 3)
                            img_ids = torch.full((img_for_bil.shape[0],), camera_id, device="cuda", dtype=torch.long)
                            
                            transformed = slice(bil_grids.grids,
                                                bil_grids.grid_X, bil_grids.grid_Y, bil_grids.grid_W,
                                                coords, img_for_bil, img_ids)
                            image = transformed["rgb"].reshape(h, w, 3).permute(2, 0, 1)
                        except KeyError:
                            # Skip if camera_id isn't found
                            pass
                    
                    gt_image = clamp_colors(viewpoint.original_image.to("cuda"))
                    if tb_writer and (idx < 5):
                        render_for_log = convert_color_to_uint8(image) / 255.0
                        gt_for_log = convert_color_to_uint8(gt_image) / 255.0
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), render_for_log[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_for_log[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parse_args_with_color_min(parser, sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.checkpoint_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dataset = lp.extract(args)
    if not hasattr(dataset, "color_min"):
        setattr(dataset, "color_min", getattr(args, "color_min", -1.0))
    opt_params = op.extract(args)
    pipe_params = pp.extract(args)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    training(dataset, opt_params, pipe_params, args.save_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
