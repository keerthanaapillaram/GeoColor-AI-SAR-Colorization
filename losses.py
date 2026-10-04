import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

class PhysicsInformedCompositeLoss(nn.Module):
    """
    Pix2Pix calibrated objective:
    Strong L1 pixel loss (100.0) + SSIM (10.0) + Edge Preservation (5.0) + Adversarial (1.0).
    """
    def __init__(self, lambda_l1=100.0, lambda_ssim=10.0, lambda_edge=5.0, lambda_adv=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_adv = lambda_adv

    def forward(self, pred_rgb, target_rgb, clean_sar, raw_sar, d_pred_fake=None):
        # 1. Pixel-accurate L1 Loss
        loss_l1 = self.l1(pred_rgb, target_rgb)

        # 2. Structural Similarity (inputs are [0, 1])
        loss_ssim = 1.0 - ssim(pred_rgb, target_rgb, data_range=1.0, win_size=7)

        # 3. Structural High-Frequency Edge Consistency
        pred_grad = torch.abs(pred_rgb[:, :, 1:, :] - pred_rgb[:, :, :-1, :]).mean() + \
                    torch.abs(pred_rgb[:, :, :, 1:] - pred_rgb[:, :, :, :-1]).mean()
        sar_grad = torch.abs(clean_sar[:, :, 1:, :] - clean_sar[:, :, :-1, :]).mean() + \
                   torch.abs(clean_sar[:, :, :, 1:] - clean_sar[:, :, :, :-1]).mean()
        loss_edge = torch.abs(pred_grad - sar_grad)

        # 4. PatchGAN Adversarial Feedback
        if d_pred_fake is not None:
            valid_labels = torch.ones_like(d_pred_fake)
            loss_adv = self.bce(d_pred_fake, valid_labels)
        else:
            loss_adv = torch.tensor(0.0, device=pred_rgb.device)

        total_loss = (self.lambda_l1 * loss_l1) + \
                     (self.lambda_ssim * loss_ssim) + \
                     (self.lambda_edge * loss_edge) + \
                     (self.lambda_adv * loss_adv)

        return total_loss, loss_l1, loss_ssim, loss_edge