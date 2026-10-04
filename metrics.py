import numpy as np
import cv2
import torch
from pytorch_msssim import ssim as msssim

def calculate_psnr(img_pred, img_gt, data_range=1.0):
    """Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = np.mean((img_pred.astype(np.float64) - img_gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))

def calculate_ssim(img_pred_tensor, img_gt_tensor):
    """
    Structural Similarity Index (SSIM).
    Expects 4D torch tensors [B, C, H, W] normalized in [0, 1].
    """
    return float(msssim(img_pred_tensor, img_gt_tensor, data_range=1.0, win_size=7).item())

def calculate_sam(img_pred, img_gt):
    """
    Spectral Angle Mapper (SAM) in degrees.
    Expects numpy arrays of shape [H, W, 3] in range [0, 1].
    """
    pred = img_pred.astype(np.float64)
    gt = img_gt.astype(np.float64)
    
    dot_product = np.sum(pred * gt, axis=-1)
    norm_pred = np.linalg.norm(pred, axis=-1)
    norm_gt = np.linalg.norm(gt, axis=-1)
    
    denom = norm_pred * norm_gt
    valid_mask = denom > 1e-7
    
    cos_theta = np.zeros_like(dot_product)
    cos_theta[valid_mask] = np.clip(dot_product[valid_mask] / denom[valid_mask], -1.0, 1.0)
    
    angles_rad = np.arccos(cos_theta[valid_mask])
    sam_deg = np.mean(np.degrees(angles_rad))
    return float(sam_deg)

def calculate_enl(image_np):
    """Equivalent Number of Looks (ENL) to quantify despeckling performance."""
    if image_np.ndim == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = image_np.astype(np.float32)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    if std_val == 0:
        return 0.0
    return float((mean_val ** 2) / (std_val ** 2))

def calculate_epi(original_sar, despeckled_sar):
    """Edge Preservation Index (EPI) using directional gradients."""
    if original_sar.ndim == 3:
        orig = cv2.cvtColor(original_sar, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        orig = original_sar.astype(np.float32)
        
    if despeckled_sar.ndim == 3:
        filt = cv2.cvtColor(despeckled_sar, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        filt = despeckled_sar.astype(np.float32)

    diff_orig = np.abs(orig[1:, :] - orig[:-1, :])
    diff_filt = np.abs(filt[1:, :] - filt[:-1, :])
    sum_orig = np.sum(diff_orig)
    sum_filt = np.sum(diff_filt)
    if sum_orig == 0:
        return 1.0
    return float(sum_filt / sum_orig)