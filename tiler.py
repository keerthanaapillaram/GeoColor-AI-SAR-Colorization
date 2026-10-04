import numpy as np
import torch

def generate_hann_2d(size):
    h = np.hanning(size)
    w = np.hanning(size)
    window = np.outer(h, w)
    return window / (np.max(window) + 1e-7)

def process_large_swath_seamless(model, large_sar_array, device, tile_size=256, overlap=64):
    """
    Slices and reconstructs large-scale satellite scenes with 2D Hann window blending
    to avoid seam boundaries.
    """
    model.eval()
    h, w = large_sar_array.shape
    stride = tile_size - overlap

    pad_h = (stride - (h - tile_size) % stride) % stride
    pad_w = (stride - (w - tile_size) % stride) % stride

    padded_sar = np.pad(large_sar_array, ((0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = padded_sar.shape

    output_rgb = np.zeros((3, ph, pw), dtype=np.float32)
    weight_canvas = np.zeros((ph, pw), dtype=np.float32)
    hann_window = generate_hann_2d(tile_size)

    for y in range(0, ph - tile_size + 1, stride):
        for x in range(0, pw - tile_size + 1, stride):
            patch = padded_sar[y:y+tile_size, x:x+tile_size]
            tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            tensor = (tensor - 0.5) / 0.5

            with torch.no_grad():
                _, pred_patch = model(tensor)

            pred_np = ((pred_patch.squeeze(0).cpu().numpy() + 1.0) * 0.5)

            for c in range(3):
                output_rgb[c, y:y+tile_size, x:x+tile_size] += pred_np[c] * hann_window
            weight_canvas[y:y+tile_size, x:x+tile_size] += hann_window

    output_rgb /= np.maximum(weight_canvas, 1e-5)
    output_rgb = output_rgb[:, :h, :w]
    return np.clip(output_rgb, 0.0, 1.0)