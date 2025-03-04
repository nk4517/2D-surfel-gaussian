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

import os
import time
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, norm_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import fov2focal
import open3d as o3d
import numpy as np
import torchvision
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# vis_grad(gaussians._opacity.grad, gaussians._xyz, iteration, ".")

def vis_grad(grad,xyz,idx, depth_path):
# def visualize(idx=0):
    '''
    Input:
    rendering: (1,H,W) tensor
    depth: (1,H,W) tensor
    K: (4,4) tensor
    '''
    
    
    X = xyz[:, 0].detach().cpu().numpy()
    Y = xyz[:, 1].detach().cpu().numpy()
    Z = xyz[:, 2].detach().cpu().numpy()

    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(np.stack((X, Y, Z), axis=-1).reshape(-1, 3))
    colors = grad.expand(-1, 3).reshape(-1, 3).cpu().numpy()
    grad_local = grad.cpu().numpy().squeeze()
    # ==0 : white, >0: red, <0: blue
    colors[grad_local>0] = np.array([1, 1, 1]) - np.array([0, 1, 1]) * grad_local[grad_local>0].reshape(-1, 1) * 10
    colors[grad_local<=0] = np.array([1, 1, 1]) + np.array([1, 0, 1]) * grad_local[grad_local<=0].reshape(-1, 1) * 10

    colors = np.clip(colors, 0, 1)

    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(os.path.join(depth_path, f"output{idx:05d}.ply"), pcd)

def vis_value(value,xyz,idx, depth_path):
    X = xyz[:, 0].detach().cpu().numpy()
    Y = xyz[:, 1].detach().cpu().numpy()
    Z = xyz[:, 2].detach().cpu().numpy()

    os.makedirs(depth_path, exist_ok=True)

    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(np.stack((X, Y, Z), axis=-1).reshape(-1, 3))
    colors = np.ones((len(X), 3))
    colors -= np.array([0, 1, 1]) * value.detach().cpu().numpy().reshape(-1, 1)

    colors = np.clip(colors, 0, 1)

    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(os.path.join(depth_path, f"output{idx:05d}.ply"), pcd)
 



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    Ld_value = torch.zeros(1, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        Ld_value *= 0.0
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, Ld_value=Ld_value)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, Ld_value=Ld_value)
        image, viewspace_point_tensor, visibility_filter, radii, depth, ray_P, ray_M, blend_normal = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"], render_pkg["ray_P"], render_pkg["ray_M"], render_pkg["normal"]
        ray_P.retain_grad()
        ray_M.retain_grad()
        depth.retain_grad()
        # if (ray_P == 0.0).sum() > 0:
        #     print("Zero ray_P")
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        newdepth = torch.clamp(depth, 0.1)
        newdepth.retain_grad()
        fx = fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width)
        fy = fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height)
        Ln, depth_norm, loss_map = norm_loss(ray_P, ray_M, newdepth, fx, fy, viewpoint_cam.image_width, viewpoint_cam.image_height)
        # torchvision.utils.save_image(image, f"image_{iteration:05d}.png")
        # nandepth = depth.clone().detach()
        # nandepth = torch.clip(nandepth, 0, 10)
        # torchvision.utils.save_image(nandepth/10.0, f"depth_{iteration:05d}.png")
        if torch.isnan(Ln):
            print("Nan Loss")
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.05 * Ln
        loss.backward()

        # gradient clipping
        gradnorm = torch.nn.utils.clip_grad_norm_(gaussians.get_paramlist(), 2e-4)

        # if iteration % 50 == 1:
        #     vis_value(gaussians.get_opacity, gaussians._xyz, iteration, "./value_GT")
        #     vis_grad(gaussians._opacity.grad, gaussians._xyz, iteration, ".")
        
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, Ln, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), Ld_value, gradnorm)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                # gaussians.zero_z()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        unique_str = time.strftime("%m%d-%H%M%S") 
        args.model_path = os.path.join("./output/", unique_str[0:10] + "-2D-" + args.source_path.split("/")[-1])
        
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

def training_report(tb_writer, iteration, Ll1, Ln, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, Ld_value, gradnorm):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/Ld_value', Ld_value.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/norm_loss', Ln.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration) 
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_scalar('mean_opacity', scene.gaussians.get_opacity.mean(), iteration)
        tb_writer.add_scalar('mean_xyz_grad_mean', scene.gaussians.get_xyz.grad.norm(dim=1).mean(), iteration)
        tb_writer.add_scalar('grad_norm', gradnorm, iteration)
        if hasattr(scene.gaussians, "clone_pts_num"):
            tb_writer.add_scalar('densify/clone', scene.gaussians.clone_pts_num, iteration)
            tb_writer.add_scalar('densify/split', scene.gaussians.split_pts_num, iteration)
            tb_writer.add_scalar('densify/prune', scene.gaussians.prune_pts_num, iteration)
            tb_writer.add_scalar('densify/viewspace_grad', scene.gaussians.viewspace_grad, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = render_pkg["render"]
                    image = torch.clamp(image, 0.0, 1.0)
                    render_normal = render_pkg["normal"]
                    render_normal /= render_normal.norm(dim=0, keepdim=True) 
                    render_normal_color = (1 - render_normal) * 0.5
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    Ln, depth_norm, loss_map = norm_loss(render_pkg["ray_P"], render_pkg["ray_M"], render_pkg["depth"], fov2focal(viewpoint.FoVx, viewpoint.image_width), fov2focal(viewpoint.FoVy, viewpoint.image_height), viewpoint.image_width, viewpoint.image_height)
                    norm_color = (1 - depth_norm) * 0.5
                    similar_map = ((render_normal * depth_norm).sum(dim=0, keepdim=True) + 1) / 2.0

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_depthnorm/norm".format(viewpoint.image_name), norm_color[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_normal".format(viewpoint.image_name), render_normal_color[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_lossmap/loss".format(viewpoint.image_name), loss_map[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_lossmap/similar".format(viewpoint.image_name), similar_map[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_ray_P".format(viewpoint.image_name), render_pkg["ray_P"][None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*1000 for i in range(30)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[i*1000 for i in range(30)])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[4000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
