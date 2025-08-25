#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import commentjson as json

import numpy as np

import sys
import time

from common import *
from render_utils import render_img_training_view

from shutil import copyfile
from tqdm import tqdm

import pyngp as ngp # noqa

from torch.utils.tensorboard import SummaryWriter

def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--name", default="neus", type=str, required=True)
	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified.")
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float, help="set the distance from the camera at which training rays start for nerf. <0 means use ngp default")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--save_mesh", action="store_true")
	parser.add_argument("--save_mesh_path", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=512, type=int, help="Sets the resolution for the marching cubes grid.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	parser.add_argument("--log_interval", type=int, default=500, help="Interval for logging evaluation metrics during training.")
	parser.add_argument("--configuration", choices=["baseline", "surface", "volume", "hybrid"], default="baseline", help="Rendering configuration: baseline (original), surface (SDF + surface features), volume (SDF + divergence features), or hybrid (SDF + surface + divergence features)")

	# Loss configuration
	parser.add_argument("--loss_mode", choices=["baseline", "surface"], default="baseline", help="Loss computation mode: baseline volumetric loss or surface-weighted loss")
	parser.add_argument("--occupancy_warmup_steps", type=int, default=1000, help="Number of warm-up steps for occupancy alpha clamping in surface mode")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images.")

	## na_test
	parser.add_argument('--test_camera_view', type=int,default=0)
	parser.add_argument('--test', action='store_true')
	parser.add_argument('--render_img_HW', type=int, default=None)
	parser.add_argument("--shaded_mesh", action='store_true')
	parser.add_argument("--white_bkgd", action='store_true')

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()

	# Parse dataset path to extract dataset and scene names
	dataset_path = args.scene if args.scene else ""
	dataset_name = "unknown"
	scene_name = "unknown"
	
	if dataset_path:
		# Extract dataset and scene from path like /path/to/nerf_synthetic/lego/transforms_train.json
		path_parts = dataset_path.split('/')
		if len(path_parts) >= 2:
			# Find the dataset name (e.g., nerf_synthetic)
			for i, part in enumerate(path_parts):
				if part in ['nerf_synthetic', 'nerf_real', 'mipnerf360', 'tanks_and_temples', 'deepvoxels']:
					dataset_name = part
					# Scene name is the next part
					if i + 1 < len(path_parts):
						scene_name = path_parts[i + 1]
					break
	
	# Create output path: dataset/scene/config/method_name
	args.output_path = os.path.join(dataset_name, scene_name, args.configuration, args.name)
	os.makedirs(os.path.join(args.output_path,"checkpoints"), exist_ok=True)
	os.makedirs(os.path.join(args.output_path,"mesh"), exist_ok=True)
	os.makedirs(os.path.join(args.output_path,"logs"), exist_ok=True)
	
	time_name = time.strftime("%m_%d_%H_%M", time.localtime())
	writer = SummaryWriter(log_dir=os.path.join(args.output_path, 'logs', time_name))
	
	# Initialize training log file
	training_log_path = os.path.join(args.output_path, "training_log.txt")
	with open(training_log_path, "w") as f:
		f.write(f"NeuS2 Training Log\n")
		f.write(f"="*50 + "\n")
		f.write(f"Dataset: {dataset_name}\n")
		f.write(f"Scene: {scene_name}\n")
		f.write(f"Configuration: {args.configuration}\n")
		f.write(f"Method: {args.name}\n")
		f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
		f.write(f"Output Path: {args.output_path}\n")
		f.write(f"Network Config: {args.network}\n")
		f.write(f"Log Interval: {args.log_interval}\n")
		f.write(f"="*50 + "\n\n")

	mode = ngp.TestbedMode.Nerf 
	configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")

	base_network = os.path.join(configs_dir, "dtu.json")
	network = args.network if args.network else base_network
	if not os.path.isabs(network):
		network = os.path.join(configs_dir, network)

	def file_backup(output_path, config_path):
		rec_dir = os.path.join(output_path, 'recording')
		os.makedirs(rec_dir, exist_ok=True)
		copyfile(config_path, os.path.join(rec_dir, 'config.json'))
		
		filepath = os.path.join('src', 'testbed_nerf.cu')
		if os.path.exists(filepath):
			copyfile(filepath, os.path.join(rec_dir, 'testbed_nerf.cu'))
		filepath = os.path.join('src', 'testbed.cu')
		if os.path.exists(filepath):
			copyfile(filepath, os.path.join(rec_dir, 'testbed.cu'))
		filepath = os.path.join('include', 'neural-graphics-primitives', 'nerf_network.h')
		if os.path.exists(filepath):
			copyfile(filepath, os.path.join(rec_dir, 'nerf_network.h'))
	
	file_backup(args.output_path, network)

	testbed = ngp.Testbed(mode)
	testbed.nerf.sharpen = float(args.sharpen)

	# Propagate loss configuration to the testbed
	if hasattr(testbed, "loss_mode"):
		testbed.loss_mode = args.loss_mode
	if hasattr(testbed, "occupancy_warmup_steps"):
		testbed.occupancy_warmup_steps = int(args.occupancy_warmup_steps)

	# Handle rendering configuration
	print(f"Using rendering configuration: {args.configuration}")
	if args.configuration == "baseline":
		# Use existing, untouched NeuS2 code
		pass
	elif args.configuration == "surface":
		# Surface configuration - will be implemented in Task 5
		print("Surface configuration selected - implementation coming in Task 5")
		pass
	elif args.configuration == "volume":
		print(f"Using VOLUME configuration: 46D SDF output -> 15D divergence features + 1D SDF = 16D RGB input")
		# Volume configuration logic will be handled in C++/CUDA
	elif args.configuration == "hybrid":
		print(f"Using HYBRID configuration: 46D SDF output -> 15D surface features + 15D divergence features + 1D SDF = 31D RGB input")
		# Hybrid configuration logic will be handled in C++/CUDA
	else:
		print(f"Using BASELINE configuration: 16D SDF output -> 16D RGB input (original NeuS2)")

	if mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	if args.scene:
		scene = args.scene
		testbed.load_training_data(scene)


	if args.load_snapshot:
		print("Loading snapshot ", args.load_snapshot)
		testbed.load_snapshot(args.load_snapshot)
	else:
		testbed.reload_network_from_file(network)


	if args.test:
		log_path = os.path.join(args.output_path, f"eval_log.txt")
		log_ptr = open(log_path, "w+")

		if args.load_snapshot:
			print("Loading snapshot ", args.load_snapshot)
			testbed.load_snapshot(args.load_snapshot)
		else:
			print("specify a checkpoint path")
			exit(1)
		
		args.save_mesh_path = os.path.join(args.output_path,"mesh",f"{-1}.obj")
		if args.save_mesh_path and args.save_mesh:
			res = args.marching_cubes_res or 256
			print(f"Generating mesh via marching cubes and saving to {args.save_mesh_path}. Resolution=[{res},{res},{res}]")
			testbed.compute_and_save_marching_cubes_mesh(args.save_mesh_path, [res, res, res])
			
		render_img_training_view(args, testbed, log_ptr, args.scene)

	else:
		ref_transforms = {}
		if args.screenshot_transforms: # try to load the given file straight away
			print("Screenshot transforms from ", args.screenshot_transforms)
			with open(args.screenshot_transforms) as f:
				ref_transforms = json.load(f)

		if args.gui:
			# Pick a sensible GUI resolution depending on arguments.
			sw = args.width or 1920
			sh = args.height or 1080
			while sw*sh > 1920*1080*4:
				sw = int(sw / 2)
				sh = int(sh / 2)
			testbed.init_window(sw, sh)

		testbed.shall_train = args.train if args.gui else True


		testbed.nerf.render_with_camera_distortion = True

		network_stem = os.path.splitext(os.path.basename(network))[0]

		if args.near_distance >= 0.0:
			print("NeRF training ray near_distance ", args.near_distance)
			testbed.nerf.training.near_distance = args.near_distance

		if args.nerf_compatibility:
			print(f"NeRF compatibility mode enabled")

			# Prior nerf papers accumulate/blend in the sRGB
			# color space. This messes not only with background
			# alpha, but also with DOF effects and the likes.
			# We support this behavior, but we only enable it
			# for the case of synthetic nerf data where we need
			# to compare PSNR numbers to results of prior work.
			testbed.color_space = ngp.ColorSpace.SRGB

			# No exponential cone tracing. Slightly increases
			# quality at the cost of speed. This is done by
			# default on scenes with AABB 1 (like the synthetic
			# ones), but not on larger scenes. So force the
			# setting here.
			testbed.nerf.cone_angle_constant = 0

			# Optionally match nerf paper behaviour and train on a
			# fixed white bg. We prefer training on random BG colors.
			# testbed.background_color = [1.0, 1.0, 1.0, 1.0]
			# testbed.nerf.training.random_bg_color = False

		old_training_step = 0
		n_steps = args.n_steps
		if n_steps < 0:
			n_steps = 100000

		args.save_snapshot = os.path.join(args.output_path,"checkpoints",f"{n_steps}.msgpack")
		args.save_mesh_path = os.path.join(args.output_path,"mesh",f"{n_steps}.obj")
		
		# Log training parameters
		training_log_path = os.path.join(args.output_path, "training_log.txt")
		with open(training_log_path, "a") as f:
			f.write(f"Training Parameters:\n")
			f.write(f"  Total Steps: {n_steps}\n")
			f.write(f"  Checkpoint Path: {args.save_snapshot}\n")
			f.write(f"  Mesh Path: {args.save_mesh_path}\n")
			f.write(f"  Log Interval: {args.log_interval}\n")
			f.write(f"\n")

		tqdm_last_update = 0
		if n_steps > 0:
			with tqdm(desc="Training", total=n_steps, unit="step") as t:
				while testbed.frame():
					if testbed.want_repl():
						repl(testbed)
					# What will happen when training is done?
					if testbed.training_step >= n_steps:
						if args.gui:
							testbed.shall_train = False
						else:
							break

					# Update progress bar
					if testbed.training_step < old_training_step or old_training_step == 0:
						old_training_step = 0
						t.reset()

					now = time.monotonic()
					if now - tqdm_last_update > 0.1:
						t.update(testbed.training_step - old_training_step)
						t.set_postfix(loss=testbed.loss)
						old_training_step = testbed.training_step
						tqdm_last_update = now

					# writer.add_scalar('psnr', s_val.mean(), self.iter_step)
					if testbed.training_step % 20 == 0:
						writer.add_scalar('loss/rgb_loss', testbed.loss, testbed.training_step)
						writer.add_scalar('loss/ek_loss', testbed.ek_loss, testbed.training_step)
						writer.add_scalar('loss/mask_loss', testbed.mask_loss, testbed.training_step)
					
					# Periodic evaluation during training
					if testbed.training_step % args.log_interval == 0 and testbed.training_step > 0:
						# Set model to eval mode for evaluation
						testbed.shall_train = False
						
						# Select a fixed image from training dataset for evaluation
						eval_image_idx = 0  # Use first training image
						testbed.set_camera_to_training_view(eval_image_idx)
						
						# Get ground truth image dimensions
						training_data = testbed.nerf.training.dataset
						if hasattr(training_data, 'images') and len(training_data.images) > 0:
							gt_image = training_data.images[eval_image_idx]
							h, w = gt_image.shape[:2]
							
							# Render current model output
							rendered_image = testbed.render(w, h, 8, True)  # 8 samples per pixel
							
							# Convert to tensor format for metrics
							gt_tensor = torch.from_numpy(gt_image[..., :3]).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0
							rendered_tensor = torch.from_numpy(rendered_image[..., :3]).float().unsqueeze(0).permute(0, 3, 1, 2)
							
							# Calculate metrics
							from scripts.utils.metrics import MetricsCalculator
							metrics_calc = MetricsCalculator(device='cuda')
							metrics = metrics_calc.calculate_all_metrics(rendered_tensor, gt_tensor)
							
							# Log to tensorboard
							writer.add_scalar('eval/psnr', metrics['psnr'], testbed.training_step)
							writer.add_scalar('eval/ssim', metrics['ssim'], testbed.training_step)
							writer.add_scalar('eval/lpips', metrics['lpips'], testbed.training_step)
							
							# Log to file
							log_path = os.path.join(args.output_path, f"{args.name}_eval_log.txt")
							with open(log_path, "a") as f:
								f.write(f"Iteration {testbed.training_step}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.3f}, LPIPS={metrics['lpips']:.3f}\n")
							
							print(f"Evaluation at step {testbed.training_step}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.3f}, LPIPS={metrics['lpips']:.3f}")
						
						# Set model back to train mode
						testbed.shall_train = True


		if args.save_snapshot:
			print("Saving snapshot ", args.save_snapshot)
			testbed.save_snapshot(args.save_snapshot, False)
			
			# Log completion
			training_log_path = os.path.join(args.output_path, "training_log.txt")
			with open(training_log_path, "a") as f:
				f.write(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
				f.write(f"Final checkpoint saved to: {args.save_snapshot}\n")
				f.write(f"="*50 + "\n")

		
		res = args.marching_cubes_res or 256
		print(f"Generating mesh via marching cubes and saving to {args.save_mesh_path}. Resolution=[{res},{res},{res}]")
		testbed.compute_and_save_marching_cubes_mesh(args.save_mesh_path, [res, res, res])

		log_path = os.path.join(args.output_path, f"eval_log.txt")
		log_ptr = open(log_path, "w+")
		render_img_training_view(args, testbed, log_ptr, args.scene)
		


		if args.test_transforms:
			print("Evaluating test transforms from ", args.test_transforms)
			with open(args.test_transforms) as f:
				test_transforms = json.load(f)
			data_dir=os.path.dirname(args.test_transforms)
			totmse = 0
			totpsnr = 0
			totssim = 0
			totcount = 0
			minpsnr = 1000
			maxpsnr = 0

			# Evaluate metrics on black background
			testbed.background_color = [0.0, 0.0, 0.0, 1.0]

			# Prior nerf papers don't typically do multi-sample anti aliasing.
			# So snap all pixels to the pixel centers.
			testbed.snap_to_pixel_centers = True
			spp = 8

			testbed.nerf.rendering_min_transmittance = 1e-4
			import pdb
			pdb.set_trace()
			if "from_na" in test_transforms.keys():
				pass
			else:
				testbed.fov_axis = 0
				testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
				testbed.shall_train = False

			with tqdm(list(enumerate(test_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
				for i, frame in t:
					p = frame["file_path"]
					if "." not in p:
						p = p + ".png"
					ref_fname = os.path.join(data_dir, p)
					if not os.path.isfile(ref_fname):
						ref_fname = os.path.join(data_dir, p + ".png")
						if not os.path.isfile(ref_fname):
							ref_fname = os.path.join(data_dir, p + ".jpg")
							if not os.path.isfile(ref_fname):
								ref_fname = os.path.join(data_dir, p + ".jpeg")
								if not os.path.isfile(ref_fname):
									ref_fname = os.path.join(data_dir, p + ".exr")

					ref_image = read_image(ref_fname)

					# NeRF blends with background colors in sRGB space, rather than first
					# transforming to linear space, blending there, and then converting back.
					# (See e.g. the PNG spec for more information on how the `alpha` channel
					# is always a linear quantity.)
					# The following lines of code reproduce NeRF's behavior (if enabled in
					# testbed) in order to make the numbers comparable.
					if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
						# Since sRGB conversion is non-linear, alpha must be factored out of it
						ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
						ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
						ref_image[...,:3] *= ref_image[...,3:4]
						ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
						ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])

					if i == 0:
						write_image("ref.png", ref_image)

					testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
					# testbed.set_camera_to_training_view(i)
					image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

					if i == 0:
						write_image("out.png", image)

					diffimg = np.absolute(image - ref_image)
					diffimg[...,3:4] = 1.0
					if i == 0:
						write_image("diff.png", diffimg)

					A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
					R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
					mse = float(compute_error("MSE", A, R))
					ssim = float(compute_error("SSIM", A, R))
					totssim += ssim
					totmse += mse
					psnr = mse2psnr(mse)
					totpsnr += psnr
					minpsnr = psnr if psnr<minpsnr else minpsnr
					maxpsnr = psnr if psnr>maxpsnr else maxpsnr
					totcount = totcount+1
					t.set_postfix(psnr = totpsnr/(totcount or 1))
					# break

			psnr_avgmse = mse2psnr(totmse/(totcount or 1))
			psnr = totpsnr/(totcount or 1)
			ssim = totssim/(totcount or 1)
			print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")

		if args.save_mesh_path:
			res = args.marching_cubes_res or 256
			print(f"Generating mesh via marching cubes and saving to {args.save_mesh_path}. Resolution=[{res},{res},{res}]")
			testbed.compute_and_save_marching_cubes_mesh(args.save_mesh_path, [res, res, res])

		if args.width:
			if ref_transforms:
				testbed.fov_axis = 0
				testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
				if not args.screenshot_frames:
					args.screenshot_frames = range(len(ref_transforms["frames"]))
				print(args.screenshot_frames)
				for idx in args.screenshot_frames:
					f = ref_transforms["frames"][int(idx)]
					cam_matrix = f["transform_matrix"]
					testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
					outname = os.path.join(args.screenshot_dir, os.path.basename(f["file_path"]))

					# Some NeRF datasets lack the .png suffix in the dataset metadata
					if not os.path.splitext(outname)[1]:
						outname = outname + ".png"

					print(f"rendering {outname}")
					image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
					os.makedirs(os.path.dirname(outname), exist_ok=True)
					write_image(outname, image)
			elif args.screenshot_dir:
				outname = os.path.join(args.screenshot_dir, args.scene + "_" + network_stem)
				print(f"Rendering {outname}.png")
				image = testbed.render(args.width, args.height, args.screenshot_spp, True)
				if os.path.dirname(outname) != "":
					os.makedirs(os.path.dirname(outname), exist_ok=True)
				write_image(outname + ".png", image)



