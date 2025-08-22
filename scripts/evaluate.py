#!/usr/bin/env python3

import argparse
import os
import json
import numpy as np
import torch
from tqdm import tqdm
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.metrics import MetricsCalculator
from common import *
from render_utils import render_img_training_view
import pyngp as ngp

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained NeuS2 model on test dataset")
    
    parser.add_argument("--checkpoint_path", required=True, help="Path to the trained model checkpoint (.msgpack)")
    parser.add_argument("--test_json_path", required=True, help="Path to transforms_test.json file")
    parser.add_argument("--background_color", choices=["white", "black"], default="black", help="Background color for rendering")
    parser.add_argument("--output_path", default="evaluation_results", help="Output directory for results")
    parser.add_argument("--spp", type=int, default=8, help="Samples per pixel for rendering")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize testbed
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    
    # Load trained model
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    testbed.load_snapshot(args.checkpoint_path)
    
    # Load test dataset
    print(f"Loading test dataset from: {args.test_json_path}")
    with open(args.test_json_path, 'r') as f:
        test_transforms = json.load(f)
    
    # Set background color
    if args.background_color == "white":
        testbed.background_color = [1.0, 1.0, 1.0, 1.0]
    else:  # black
        testbed.background_color = [0.0, 0.0, 0.0, 1.0]
    
    # Set rendering parameters
    testbed.snap_to_pixel_centers = True
    testbed.nerf.rendering_min_transmittance = 1e-4
    
    # Initialize metrics calculator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics_calc = MetricsCalculator(device=device)
    
    # Get test data directory
    data_dir = os.path.dirname(args.test_json_path)
    
    # Initialize metrics storage
    all_metrics = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    min_psnr = float('inf')
    max_psnr = 0.0
    
    print(f"Evaluating {len(test_transforms['frames'])} test images...")
    
    # Evaluate each test image
    with tqdm(test_transforms['frames'], desc="Evaluating test images") as pbar:
        for i, frame in enumerate(pbar):
            # Get image path
            p = frame["file_path"]
            if "." not in p:
                p = p + ".png"
            
            # Try different image extensions
            ref_fname = os.path.join(data_dir, p)
            if not os.path.isfile(ref_fname):
                for ext in ['.png', '.jpg', '.jpeg', '.exr']:
                    ref_fname = os.path.join(data_dir, p + ext)
                    if os.path.isfile(ref_fname):
                        break
            
            if not os.path.isfile(ref_fname):
                print(f"Warning: Could not find image file for frame {i}")
                continue
            
            # Load ground truth image
            ref_image = read_image(ref_fname)
            
            # Handle sRGB conversion if needed
            if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
                ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], 
                                            out=np.zeros_like(ref_image[...,:3]), 
                                            where=ref_image[...,3:4] != 0)
                ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
                ref_image[...,:3] *= ref_image[...,3:4]
                ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
                ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])
            
            # Set camera matrix
            testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
            
            # Render image
            h, w = ref_image.shape[:2]
            rendered_image = testbed.render(w, h, args.spp, True)
            
            # Convert to tensor format for metrics
            gt_tensor = torch.from_numpy(ref_image[..., :3]).float().unsqueeze(0).permute(0, 3, 1, 2)
            rendered_tensor = torch.from_numpy(rendered_image[..., :3]).float().unsqueeze(0).permute(0, 3, 1, 2)
            
            # Normalize ground truth to [0, 1] if needed
            if gt_tensor.max() > 1.0:
                gt_tensor = gt_tensor / 255.0
            
            # Calculate metrics
            metrics = metrics_calc.calculate_all_metrics(rendered_tensor, gt_tensor)
            
            # Store metrics
            frame_metrics = {
                'frame': i,
                'file_path': frame["file_path"],
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'lpips': metrics['lpips']
            }
            all_metrics.append(frame_metrics)
            
            # Update totals
            total_psnr += metrics['psnr']
            total_ssim += metrics['ssim']
            total_lpips += metrics['lpips']
            min_psnr = min(min_psnr, metrics['psnr'])
            max_psnr = max(max_psnr, metrics['psnr'])
            
            # Update progress bar
            pbar.set_postfix({
                'PSNR': f"{metrics['psnr']:.2f}",
                'SSIM': f"{metrics['ssim']:.3f}",
                'LPIPS': f"{metrics['lpips']:.3f}"
            })
            
            # Print individual results
            print(f"Frame {i} ({frame['file_path']}): PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.3f}, LPIPS={metrics['lpips']:.3f}")
    
    # Calculate averages
    num_images = len(all_metrics)
    if num_images > 0:
        avg_psnr = total_psnr / num_images
        avg_ssim = total_ssim / num_images
        avg_lpips = total_lpips / num_images
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS SUMMARY")
        print("="*60)
        print(f"Total images evaluated: {num_images}")
        print(f"Background color: {args.background_color}")
        print(f"Samples per pixel: {args.spp}")
        print("\nAVERAGE METRICS:")
        print(f"PSNR: {avg_psnr:.2f}")
        print(f"SSIM: {avg_ssim:.3f}")
        print(f"LPIPS: {avg_lpips:.3f}")
        print(f"\nPSNR Range: [{min_psnr:.2f}, {max_psnr:.2f}]")
        
        # Save detailed results
        results_file = os.path.join(args.output_path, "evaluation_results.txt")
        with open(results_file, 'w') as f:
            f.write("NeuS2 Model Evaluation Results\n")
            f.write("="*40 + "\n\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Test dataset: {args.test_json_path}\n")
            f.write(f"Background color: {args.background_color}\n")
            f.write(f"Samples per pixel: {args.spp}\n")
            f.write(f"Total images: {num_images}\n\n")
            
            f.write("PER-IMAGE RESULTS:\n")
            f.write("-" * 20 + "\n")
            for metrics in all_metrics:
                f.write(f"Frame {metrics['frame']} ({metrics['file_path']}): "
                       f"PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.3f}, "
                       f"LPIPS={metrics['lpips']:.3f}\n")
            
            f.write("\nSUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average PSNR: {avg_psnr:.2f}\n")
            f.write(f"Average SSIM: {avg_ssim:.3f}\n")
            f.write(f"Average LPIPS: {avg_lpips:.3f}\n")
            f.write(f"PSNR Range: [{min_psnr:.2f}, {max_psnr:.2f}]\n")
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Save metrics as JSON for further analysis
        json_file = os.path.join(args.output_path, "evaluation_metrics.json")
        with open(json_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_images': num_images,
                    'background_color': args.background_color,
                    'spp': args.spp,
                    'average_psnr': avg_psnr,
                    'average_ssim': avg_ssim,
                    'average_lpips': avg_lpips,
                    'psnr_range': [min_psnr, max_psnr]
                },
                'per_frame_metrics': all_metrics
            }, f, indent=2)
        
        print(f"Metrics JSON saved to: {json_file}")
        
    else:
        print("No images were successfully evaluated.")

if __name__ == "__main__":
    main() 