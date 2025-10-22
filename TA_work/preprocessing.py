import os
import glob
import torch
import zstandard
import io
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio
import rioxarray
import torch.nn.functional as F

DATA_DIR = "DiffScaler/data/"
STATIC_DIR = os.path.join(DATA_DIR, "static_var/")
YEARS = [str(y) for y in range(2019, 2021)]

def decompress_zst_pt(filepath):
    with open(filepath, "rb") as f:
        compressed = f.read()
    dctx = zstandard.ZstdDecompressor()
    decompressed = io.BytesIO(dctx.decompress(compressed))
    data = torch.load(decompressed, weights_only=False)
    return data

def load_static_tif(filepath):
    ds = xr.open_dataset(filepath, engine="rasterio")
    arr = ds['band_data'][0].values
    arr = np.nan_to_num(arr)
    std = np.std(arr)
    if std < 1e-6:
        std = 1.0  # NaN handling by avoiding division by zero
    arr = (arr - np.mean(arr)) / std
    return torch.tensor(arr, dtype=torch.float32)


def load_land_cover(filepath):
    ds = xr.open_dataset(filepath, engine="rasterio")
    bands = []
    for i in range(ds['band_data'].shape[0]):
        arr = ds['band_data'][i].values
        arr = np.nan_to_num(arr)
        std = np.std(arr)
        if std < 1e-6:
            std = 1.0  # NaN handling by avoiding division by zero
        arr = (arr - np.mean(arr)) / std
        bands.append(torch.tensor(arr, dtype=torch.float32))
    return torch.stack(bands)


def collate_fn(batch):
    """
    Upsample low-res temperature to high-res dimensions and combine with HR static inputs.
    This creates fuzzy upsampled inputs for the UNet to refine.
    """
    # Stack the individual components
    low_2mt = torch.stack([b["low_2mT"] for b in batch])  # [B, 1, 84, 72]
    high_2mt = torch.stack([b["high_2mT"] for b in batch])  # [B, 1, 672, 576]
    dem = torch.stack([b["dem"] for b in batch])  # [B, 1, 672, 576]
    lat = torch.stack([b["lat"] for b in batch])  # [B, 1, 672, 576]
    lc = torch.stack([b["lc"] for b in batch])   # [B, num_lc_bands, 672, 576]
    
    # Get high-resolution target size
    target_size = high_2mt.shape[-2:]  # (672, 576)
    
    # Upsample low-res temperature to match high-res dimensions (bilinear for smoothness)
    low_2mt_upsampled = F.interpolate(
        low_2mt, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
    
    # Concatenate upsampled temperature with high-res static inputs
    # This creates: [B, 1+1+1+num_lc_bands, 672, 576] - all at high resolution
    combined_input = torch.cat([low_2mt_upsampled, dem, lat, lc], dim=1)
    
    return combined_input, high_2mt
def get_file_list():
    files = []
    for year in YEARS:
        folder = os.path.join(DATA_DIR, year)
        high_files = sorted(glob.glob(os.path.join(folder, "*_high_2mT.pt.zst")))
        for hf in high_files:
            date = os.path.basename(hf).split("_")[0]
            low_file = hf.replace("_high_2mT.pt.zst", "_low.pt.zst")
            if os.path.exists(low_file):
                files.append((hf, low_file, date))
    return files