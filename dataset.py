import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def find_dataset_root(candidate_name='dataset'):
    cwd = Path.cwd()
    search_paths = [
        cwd / candidate_name,
        cwd.parent / candidate_name,
        cwd / "archive (2)" / candidate_name,
        cwd.parent / "archive (2)" / candidate_name,
        cwd
    ]
    for p in search_paths:
        if p.exists() and (list(p.rglob('*s1*.png')) or list(p.rglob('*.png'))):
            return p
    return cwd / candidate_name


class RobustSARDataset(Dataset):
    """Strict paired Sentinel-1 SAR to Sentinel-2 Optical dataset loader."""
    def __init__(self, root_dir='dataset', img_size=256, is_train=True, db_min=-25.0, db_max=0.0):
        self.root_dir = find_dataset_root(root_dir)
        self.img_size = img_size
        self.db_min = db_min
        self.db_max = db_max
        self.is_train = is_train

        all_files = list(self.root_dir.rglob('*.png')) + list(self.root_dir.rglob('*.tif'))
        records = []

        for path in all_files:
            p_str = str(path).lower()
            is_s1 = ('\\s1\\' in p_str or '/s1/' in p_str or '_s1_' in path.name.lower())

            if is_s1:
                # Find matching s2 optical file
                opt_str = str(path).replace(f"{os.sep}s1{os.sep}", f"{os.sep}s2{os.sep}").replace("_s1_", "_s2_")
                if not os.path.exists(opt_str):
                    alt = str(path).replace('\\s1\\', '\\s2\\').replace('/s1/', '/s2/').replace('_s1_', '_s2_')
                    if os.path.exists(alt):
                        opt_str = alt
                    else:
                        continue  # Discard unpaired images

                match = re.search(r'(rois\d+_[a-za-z]+_s\d+_\d+)', path.name, re.IGNORECASE)
                roi_id = match.group(1) if match else path.stem.split('_')[0]

                records.append({
                    'sar_path': str(path),
                    'opt_path': opt_str,
                    'roi_id': roi_id,
                    'file_name': path.name
                })

        self.df = pd.DataFrame(records)
        if len(self.df) == 0:
            raise RuntimeError(f"No paired SAR-Optical files found under '{self.root_dir}'.")

        unique_rois = self.df['roi_id'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_rois)
        
        split_idx = max(1, int(len(unique_rois) * 0.85))
        train_rois = set(unique_rois[:split_idx])
        val_rois = set(unique_rois[split_idx:])

        if self.is_train:
            self.df = self.df[self.df['roi_id'].isin(train_rois)].reset_index(drop=True)
        else:
            self.df = self.df[self.df['roi_id'].isin(val_rois)].reset_index(drop=True)
            if len(self.df) == 0:
                self.df = self.df.iloc[split_idx:].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _read_sar(self, path):
        raw_img = Image.open(path).convert('L')
        arr = np.array(raw_img).astype(np.float32) / 255.0
        # Radiometric decibel stretch
        arr_db = self.db_min + arr * (self.db_max - self.db_min)
        norm_sar = np.clip((arr_db - self.db_min) / (self.db_max - self.db_min), 0.0, 1.0)
        img = Image.fromarray((norm_sar * 255.0).astype(np.uint8)).resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        return T.ToTensor()(img)  # Range [0, 1]

    def _read_optical(self, path):
        img = Image.open(path).convert('RGB').resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        return T.ToTensor()(img)  # Range [0, 1]

    def __getitem__(self, idx):
        idx = int(idx.item()) if hasattr(idx, 'item') else int(idx)
        row = self.df.iloc[idx]
        return {
            'sar': self._read_sar(row['sar_path']),
            'optical': self._read_optical(row['opt_path']),
            'filename': row['file_name']
        }

def get_dataloaders(root_dir='dataset', batch_size=8, img_size=256):
    train_set = RobustSARDataset(root_dir=root_dir, img_size=img_size, is_train=True)
    val_set = RobustSARDataset(root_dir=root_dir, img_size=img_size, is_train=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader