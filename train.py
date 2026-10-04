import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models import SARColorizerGenerator, PatchGANDiscriminator
from dataset import get_dataloaders
from losses import PhysicsInformedCompositeLoss

class PreloadedMemoryDataset(Dataset):
    """Pre-loads samples into RAM to eliminate Windows disk I/O latency."""
    def __init__(self, dataloader, max_samples=300):
        self.sar_list = []
        self.opt_list = []
        print(f"[*] Pre-loading {max_samples} paired image tiles into RAM...")
        count = 0
        for batch in dataloader:
            b_sz = batch['sar'].shape[0]
            for i in range(b_sz):
                self.sar_list.append(batch['sar'][i].clone())
                self.opt_list.append(batch['optical'][i].clone())
                count += 1
                if count >= max_samples:
                    break
            if count >= max_samples:
                break
        print(f"[+] Loaded {len(self.sar_list)} verified paired samples in memory.")

    def __len__(self):
        return len(self.sar_list)

    def __getitem__(self, idx):
        return {'sar': self.sar_list[idx], 'optical': self.opt_list[idx]}

def train(max_train_samples=300, max_val_samples=40, epochs=6, batch_size=4, img_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Node: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    raw_train_loader, raw_val_loader = get_dataloaders(root_dir='dataset', batch_size=batch_size, img_size=img_size)

    train_cached = PreloadedMemoryDataset(raw_train_loader, max_samples=max_train_samples)
    val_cached = PreloadedMemoryDataset(raw_val_loader, max_samples=max_val_samples)

    train_loader = DataLoader(train_cached, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_cached, batch_size=batch_size, shuffle=False, pin_memory=True)

    generator = SARColorizerGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    # Generator has a higher learning rate so it does not collapse under D
    opt_g = AdamW(generator.parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_d = AdamW(discriminator.parameters(), lr=5e-5, betas=(0.5, 0.999), weight_decay=1e-4)

    criterion_g = PhysicsInformedCompositeLoss(lambda_l1=100.0, lambda_ssim=10.0, lambda_edge=5.0, lambda_adv=1.0)
    criterion_d = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for step, batch in enumerate(pbar):
            sar = batch['sar'].to(device, non_blocking=True)
            optical = batch['optical'].to(device, non_blocking=True)

            # ---------------------------
            # 1. Discriminator Step (Every 2nd step to prevent D overpowering G)
            # ---------------------------
            loss_d_val = 0.0
            if step % 2 == 0:
                opt_d.zero_grad()
                with torch.no_grad():
                    clean_sar, fake_optical = generator(sar)

                d_real = discriminator(sar, optical)
                d_fake = discriminator(sar, fake_optical.detach())

                # Label smoothing: Real=0.9, Fake=0.0
                loss_d = (criterion_d(d_real, torch.full_like(d_real, 0.9)) + 
                          criterion_d(d_fake, torch.zeros_like(d_fake))) * 0.5
                loss_d.backward()
                opt_d.step()
                loss_d_val = loss_d.item()

            # ---------------------------
            # 2. Generator Step
            # ---------------------------
            opt_g.zero_grad()
            clean_sar_g, fake_optical_g = generator(sar)
            d_fake_for_g = discriminator(sar, fake_optical_g)

            loss_g, loss_l1, loss_ssim, loss_edge = criterion_g(
                fake_optical_g, optical, clean_sar_g, sar, d_pred_fake=d_fake_for_g
            )
            loss_g.backward()
            opt_g.step()

            pbar.set_postfix({
                'L1_Loss': f"{loss_l1.item():.4f}",
                'SSIM_L': f"{loss_ssim.item():.3f}",
                'D_Loss': f"{loss_d_val:.3f}"
            })

        # ---------------------------
        # Validation Evaluation
        # ---------------------------
        generator.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sar = batch['sar'].to(device, non_blocking=True)
                optical = batch['optical'].to(device, non_blocking=True)
                _, pred_rgb = generator(sar)
                val_l1 += torch.abs(pred_rgb - optical).mean().item()

        mean_val_l1 = val_l1 / max(1, len(val_loader))
        print(f"\n--> Epoch {epoch+1} Mean Validation L1: {mean_val_l1:.4f}")

        if mean_val_l1 < best_val_loss:
            best_val_loss = mean_val_l1
            torch.save(generator.state_dict(), 'checkpoints/best_sar_colorizer.pth')
            print(f"[+] Saved optimal checkpoint: checkpoints/best_sar_colorizer.pth (Val L1: {best_val_loss:.4f})")

    print("\n[*] Training completed successfully! Optimal checkpoint is saved.")

if __name__ == '__main__':
    train(max_train_samples=300, max_val_samples=40, epochs=6, batch_size=4, img_size=256)