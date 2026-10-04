import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundedHomomorphicDespeckle(nn.Module):
    """Stage 1: Log-domain homomorphic filtering to decouple speckle noise."""
    def __init__(self, in_channels=1, num_features=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features)
            ) for _ in range(2)
        ])
        self.conv_out = nn.Sequential(
            nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x_pos = torch.clamp(x, 0.0, 1.0) + 0.01
        log_y = torch.log(x_pos)

        feat = F.leaky_relu(self.conv_in(log_y), 0.2)
        for block in self.blocks:
            feat = F.leaky_relu(feat + block(feat), 0.2)

        pred_noise = self.conv_out(feat) * 0.5
        clean_log = log_y - pred_noise
        clean_sar = torch.exp(clean_log) - 0.01
        return torch.clamp(clean_sar, 0.0, 1.0)


class DecoderBlock(nn.Module):
    """Bilinear upsampling with convolution and dropout to avoid checkerboard artifacts."""
    def __init__(self, in_ch, out_ch, dropout_p=0.1):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.dropout_p = dropout_p

    def forward(self, x):
        x = self.up(x)
        return F.dropout2d(x, p=self.dropout_p, training=True)


class SARColorizerGenerator(nn.Module):
    """Symmetric 4-level U-Net Generator: 256 -> 128 -> 64 -> 32 -> 16 -> 32 -> 64 -> 128 -> 256."""
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.despeckle = BoundedHomomorphicDespeckle(in_channels=in_channels)

        # Encoder: 256 -> 128 -> 64 -> 32 -> 16
        self.e1 = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))       # 256 -> 128
        self.e2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))  # 128 -> 64
        self.e3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True)) # 64 -> 32
        self.e4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True)) # 32 -> 16

        # Bottleneck: 16x16
        self.mid = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512)
        )

        # Decoder: 16 -> 32 -> 64 -> 128 -> 256
        self.d4 = DecoderBlock(512, 256, dropout_p=0.15)            # 16 -> 32
        self.d3 = DecoderBlock(256 + 256, 128, dropout_p=0.10)      # 32 -> 64 (cat e3)
        self.d2 = DecoderBlock(128 + 128, 64, dropout_p=0.05)       # 64 -> 128 (cat e2)
        self.d1 = DecoderBlock(64 + 64, 32, dropout_p=0.0)          # 128 -> 256 (cat e1)

        # Final projection: 256x256 output, Sigmoid [0, 1]
        self.final = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        clean_sar = self.despeckle(x)

        e1 = self.e1(clean_sar)  # [B, 64, 128, 128]
        e2 = self.e2(e1)         # [B, 128, 64, 64]
        e3 = self.e3(e2)         # [B, 256, 32, 32]
        e4 = self.e4(e3)         # [B, 512, 16, 16]

        b = self.mid(e4)         # [B, 512, 16, 16]

        d4 = self.d4(b)                              # [B, 256, 32, 32]
        d3 = self.d3(torch.cat([d4, e3], dim=1))     # [B, 128, 64, 64]
        d2 = self.d2(torch.cat([d3, e2], dim=1))     # [B, 64, 128, 128]
        d1 = self.d1(torch.cat([d2, e1], dim=1))     # [B, 32, 256, 256]

        out_rgb = self.final(torch.cat([d1, clean_sar], dim=1)) # [B, 3, 256, 256]
        return clean_sar, out_rgb

    def predict_with_epistemic_uncertainty(self, x, passes=1):
        if passes == 1:
            self.eval()
            with torch.no_grad():
                clean_sar, rgb = self.forward(x)
                # Compute gradient boundaries with independent constant padding
                diff_h = torch.abs(clean_sar[:, :, 1:, :] - clean_sar[:, :, :-1, :])
                diff_w = torch.abs(clean_sar[:, :, :, 1:] - clean_sar[:, :, :, :-1])
                grad_h = F.pad(diff_h, (0, 0, 0, 1), mode='constant', value=0.0)
                grad_w = F.pad(diff_w, (0, 1, 0, 0), mode='constant', value=0.0)
                unc = (grad_h + grad_w) * 0.05
            return clean_sar, rgb, unc

        self.train()
        preds = []
        clean_sar = None
        with torch.no_grad():
            for _ in range(passes):
                clean_sar, rgb = self.forward(x)
                preds.append(rgb.unsqueeze(0))

        stacked = torch.cat(preds, dim=0)
        mean_rgb = torch.mean(stacked, dim=0)
        uncertainty = torch.var(stacked, dim=0).mean(dim=1, keepdim=True)
        return clean_sar, mean_rgb, uncertainty


class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator operating on concatenated SAR + Optical [B, 4, 256, 256]."""
    def __init__(self, in_channels=4):
        super().__init__()
        def block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, sar, optical):
        x = torch.cat([sar, optical], dim=1)
        return self.net(x)