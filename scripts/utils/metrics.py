import torch
import torch.nn as nn
import lpips
from torchmetrics import StructuralSimilarityIndexMeasure
import numpy as np

class MetricsCalculator:
    def __init__(self, device='cuda'):
        """
        Initialize metrics calculator with SSIM and LPIPS models.
        
        Args:
            device: Device to run metrics on ('cuda' or 'cpu')
        """
        self.device = device
        
        # Initialize LPIPS model
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        
        # Initialize SSIM model
        self.ssim_fn = StructuralSimilarityIndexMeasure().to(device)
        
    def calculate_psnr(self, pred, target):
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        Args:
            pred: Predicted image tensor (N, C, H, W) in [0, 1]
            target: Target image tensor (N, C, H, W) in [0, 1]
            
        Returns:
            PSNR value
        """
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    def calculate_ssim(self, pred, target):
        """
        Calculate Structural Similarity Index.
        
        Args:
            pred: Predicted image tensor (N, C, H, W) in [0, 1]
            target: Target image tensor (N, C, H, W) in [0, 1]
            
        Returns:
            SSIM value
        """
        # Ensure images are in correct format for torchmetrics
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
            
        return self.ssim_fn(pred, target)
    
    def calculate_lpips(self, pred, target):
        """
        Calculate Learned Perceptual Image Patch Similarity.
        
        Args:
            pred: Predicted image tensor (N, C, H, W) in [0, 1]
            target: Target image tensor (N, C, H, W) in [0, 1]
            
        Returns:
            LPIPS value
        """
        # LPIPS expects images in [-1, 1] range
        pred_lpips = pred * 2.0 - 1.0
        target_lpips = target * 2.0 - 1.0
        
        # Ensure images are in correct format for LPIPS
        if pred_lpips.dim() == 3:
            pred_lpips = pred_lpips.unsqueeze(0)
        if target_lpips.dim() == 3:
            target_lpips = target_lpips.unsqueeze(0)
            
        with torch.no_grad():
            lpips_value = self.lpips_fn(pred_lpips, target_lpips)
        
        return lpips_value.mean()
    
    def calculate_all_metrics(self, pred, target):
        """
        Calculate all metrics (PSNR, SSIM, LPIPS) at once.
        
        Args:
            pred: Predicted image tensor (N, C, H, W) in [0, 1]
            target: Target image tensor (N, C, H, W) in [0, 1]
            
        Returns:
            Dictionary with all metric values
        """
        # Ensure both tensors are on the same device
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        # Ensure images are in [0, 1] range
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        metrics = {
            'psnr': self.calculate_psnr(pred, target).item(),
            'ssim': self.calculate_ssim(pred, target).item(),
            'lpips': self.calculate_lpips(pred, target).item()
        }
        
        return metrics
    
    def format_metrics_string(self, metrics):
        """
        Format metrics into a readable string.
        
        Args:
            metrics: Dictionary with metric values
            
        Returns:
            Formatted string
        """
        return f"PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.3f}, LPIPS={metrics['lpips']:.3f}" 