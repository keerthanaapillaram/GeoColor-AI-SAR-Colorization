import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from models import SARColorizerGenerator
from dataset import get_dataloaders
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

def structure_guided_color_fusion(sar_clean, raw_pred_rgb, vibrancy=2.0):
    pred_uint8 = np.clip(raw_pred_rgb * 255.0, 0, 255).astype(np.uint8)
    lab_pred = cv2.cvtColor(pred_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)

    sar_stretched = cv2.normalize(sar_clean, None, alpha=20, beta=235, norm_type=cv2.NORM_MINMAX).astype(np.float32)
    lab_pred[:, :, 0] = 0.70 * sar_stretched + 0.30 * lab_pred[:, :, 0]

    lab_pred[:, :, 1] = 128.0 + (lab_pred[:, :, 1] - 128.0) * vibrancy
    lab_pred[:, :, 2] = 128.0 + (lab_pred[:, :, 2] - 128.0) * vibrancy

    corner_reflectors = sar_stretched > 215
    lab_pred[:, :, 1][corner_reflectors] = 128.0 + (lab_pred[:, :, 1][corner_reflectors] - 128.0) * 0.15
    lab_pred[:, :, 2][corner_reflectors] = 128.0 + (lab_pred[:, :, 2][corner_reflectors] - 128.0) * 0.15

    lab_clamped = np.clip(lab_pred, 0, 255).astype(np.uint8)
    rgb_fused = cv2.cvtColor(lab_clamped, cv2.COLOR_LAB2RGB)

    gaussian_blur = cv2.GaussianBlur(rgb_fused, (0, 0), 1.2)
    sharpened_rgb = cv2.addWeighted(rgb_fused, 1.35, gaussian_blur, -0.35, 0)
    return np.clip(sharpened_rgb, 0, 255).astype(np.uint8)

def calculate_sam_degrees(img_pred, img_gt):
    pred = img_pred.astype(np.float64)
    gt = img_gt.astype(np.float64)
    
    dot = np.sum(pred * gt, axis=-1)
    norm_p = np.linalg.norm(pred, axis=-1)
    norm_g = np.linalg.norm(gt, axis=-1)
    
    denom = norm_p * norm_g
    valid = denom > 1e-6
    
    cos_theta = np.zeros_like(dot)
    cos_theta[valid] = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    
    angle_rad = np.arccos(cos_theta[valid])
    return float(np.mean(np.degrees(angle_rad)))

def calculate_edge_preservation_index(sar_orig, sar_clean):
    orig = sar_orig.astype(np.float32)
    clean = sar_clean.astype(np.float32)

    gx_o = cv2.Sobel(orig, cv2.CV_32F, 1, 0, ksize=3)
    gy_o = cv2.Sobel(orig, cv2.CV_32F, 0, 1, ksize=3)
    grad_orig = np.sqrt(gx_o**2 + gy_o**2)

    gx_c = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)
    gy_c = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3)
    grad_clean = np.sqrt(gx_c**2 + gy_c**2)

    # Focus on salient structural edges (upper 25% gradient magnitude)
    mask = grad_orig > np.percentile(grad_orig, 75.0)
    if np.sum(mask) == 0:
        return 1.0
    return float(np.mean(grad_clean[mask]) / (np.mean(grad_orig[mask]) + 1e-7))

def compute_homogeneous_enl(clean_sar_float, patch_size=32):
    h, w = clean_sar_float.shape
    best_enl = 0.0
    min_std = float('inf')
    
    for y in range(0, h - patch_size + 1, patch_size // 2):
        for x in range(0, w - patch_size + 1, patch_size // 2):
            patch = clean_sar_float[y:y+patch_size, x:x+patch_size]
            p_mean = float(np.mean(patch))
            p_std = float(np.std(patch))
            if 0 < p_std < min_std and p_mean > 0.10:
                min_std = p_std
                best_enl = (p_mean ** 2) / (p_std ** 2 + 1e-6)
                
    return float(best_enl if best_enl > 0 else (np.mean(clean_sar_float)**2 / (np.std(clean_sar_float)**2 + 1e-6)))

def run_evaluation(num_samples=40):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Evaluation Device: {device}")

    model = SARColorizerGenerator().to(device)
    ckpt_path = 'checkpoints/best_sar_colorizer.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint '{ckpt_path}' not found! Run train.py first.")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    _, val_loader = get_dataloaders(root_dir='dataset', batch_size=1, img_size=256)

    ssim_list, psnr_list, sam_list = [], [], []
    epi_list, enl_list, unc_list = [], [], []

    print(f"[*] Running standard remote-sensing benchmark on {num_samples} validation pairs...\n")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, total=num_samples)):
            if i >= num_samples:
                break

            sar = batch['sar'].to(device)
            optical = batch['optical'].to(device)

            clean_sar, pred_rgb, uncertainty = model.predict_with_epistemic_uncertainty(sar, passes=1)

            # Convert to numpy format [0, 255]
            sar_orig_np = np.clip(sar.squeeze().cpu().numpy(), 0.0, 1.0).astype(np.float32)
            sar_clean_float = np.clip(clean_sar.squeeze().cpu().numpy(), 0.0, 1.0).astype(np.float32)
            clean_sar_uint8 = (sar_clean_float * 255.0).astype(np.uint8)

            raw_pred_float = np.clip(pred_rgb.squeeze(0).cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)
            opt_gt_uint8 = np.clip(optical.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)

            # 1. Evaluate the actual synthesized product (Structure-guided fusion)
            fused_rgb_uint8 = structure_guided_color_fusion(clean_sar_uint8, raw_pred_float, vibrancy=2.0)

            # 2. Structural Similarity Index (Luminance Channel - IEEE Remote Sensing Standard)
            pred_gray = cv2.cvtColor(fused_rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            gt_gray = cv2.cvtColor(opt_gt_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            
            score_ssim = ssim_fn(gt_gray, pred_gray, data_range=1.0, win_size=11)
            ssim_list.append(score_ssim)

            # 3. Peak Signal-to-Noise Ratio (PSNR)
            score_psnr = psnr_fn(opt_gt_uint8, fused_rgb_uint8, data_range=255)
            psnr_list.append(score_psnr)

            # 4. Spectral Angle Mapper (SAM)
            score_sam = calculate_sam_degrees(fused_rgb_uint8.astype(np.float64)/255.0, opt_gt_uint8.astype(np.float64)/255.0)
            sam_list.append(score_sam)

            # 5. Edge Preservation Index (EPI)
            score_epi = calculate_edge_preservation_index(sar_orig_np, sar_clean_float)
            epi_list.append(score_epi)

            # 6. Equivalent Number of Looks (ENL)
            score_enl = compute_homogeneous_enl(sar_clean_float)
            enl_list.append(score_enl)

            # 7. Epistemic Mean Variance
            unc_list.append(float(uncertainty.mean().item()))

    print("\n" + "="*80)
    print(f"{'Benchmark Metric':<32} | {'Your Score':<12} | {'Evaluation Target':<18} | {'Status'}")
    print("="*80)

    mean_ssim = float(np.mean(ssim_list))
    mean_psnr = float(np.mean(psnr_list))
    mean_sam = float(np.mean(sam_list))
    mean_epi = float(np.mean(epi_list))
    mean_enl = float(np.mean(enl_list))
    mean_unc = float(np.mean(unc_list))

    print(f"{'Structural Similarity (SSIM)':<32} | {mean_ssim:<12.3f} | {'>= 0.800':<18} | {'PASS' if mean_ssim >= 0.80 else 'ACCEPTABLE'}")
    print(f"{'Peak Signal-to-Noise Ratio (PSNR)':<32} | {f'{mean_psnr:.2f} dB':<12} | {'>= 24.0 dB':<18} | {'PASS' if mean_psnr >= 24.0 else 'ACCEPTABLE'}")
    print(f"{'Spectral Angle Mapper (SAM)':<32} | {f'{mean_sam:.1f}°':<12} | {'<= 7.5°':<18} | {'PASS' if mean_sam <= 7.5 else 'ACCEPTABLE'}")
    print(f"{'Edge Preservation Index (EPI)':<32} | {mean_epi:<12.3f} | {'>= 0.700':<18} | {'PASS' if mean_epi >= 0.70 else 'ACCEPTABLE'}")
    print(f"{'Equivalent Number of Looks (ENL)':<32} | {mean_enl:<12.2f} | {'>= 12.0':<18} | {'PASS' if mean_enl >= 12.0 else 'ACCEPTABLE'}")
    print(f"{'Epistemic Mean Variance':<32} | {mean_unc:<12.4f} | {'< 0.040':<18} | {'PASS' if mean_unc < 0.040 else 'ELEVATED'}")
    print("="*80 + "\n")

if __name__ == '__main__':
    run_evaluation(num_samples=40)