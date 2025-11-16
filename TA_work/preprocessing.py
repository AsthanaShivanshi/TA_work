import os
import glob
import torch
import zstandard
import io
import numpy as np
import xarray as xr
import torch.nn.functional as F
import json

DATA_DIR = "DiffScaler/data/"
STATIC_DIR = os.path.join(DATA_DIR, "static_var/")
YEARS = [str(y) for y in range(2020,2021)]



def compute_stats(tensor):
    mean = tensor.mean().item()
    std = tensor.std().item()
    return mean, std


def decompress_zst_pt(filepath):
    with open(filepath, "rb") as f:
        compressed = f.read()

        
    dctx = zstandard.ZstdDecompressor()
    decompressed = io.BytesIO(dctx.decompress(compressed))
    data = torch.load(decompressed, weights_only=False)
    return data

def load_static_tif(filepath, mean=None, std=None):
    ds = xr.open_dataset(filepath, engine="rasterio")
    arr = ds['band_data'][0].values
    arr = np.nan_to_num(arr)
    if mean is not None and std is not None:
        arr = (arr - mean) / std
    return torch.tensor(arr, dtype=torch.float32)

def load_land_cover(filepath, means=None, stds=None):
    ds = xr.open_dataset(filepath, engine="rasterio")
    bands = []
    for i in range(ds['band_data'].shape[0]):
        arr = ds['band_data'][i].values
        arr = np.nan_to_num(arr)
        if means is not None and stds is not None:
            arr = (arr - means[i]) / stds[i]
        bands.append(torch.tensor(arr, dtype=torch.float32))
    return torch.stack(bands)


def collate_fn(batch):

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


def load_and_normalise(static_dir, val_frac=0.15, test_frac=0.15, save_stats_json=None):
    dem = load_static_tif(os.path.join(static_dir, "dtm_2km_domain_trim_EPSG3035.tif"))
    lat = load_static_tif(os.path.join(static_dir, "lat_2km_domain_trim_EPSG3035.tif"))
    lc = load_land_cover(os.path.join(static_dir, "land_cover_classes_2km_domain_trim_EPSG3035.tif"))
    dem_mean, dem_std = compute_stats(dem)
    lat_mean, lat_std = compute_stats(lat)
    lc_means, lc_stds = [], []
    for i in range(lc.shape[0]):
        m, s = compute_stats(lc[i])
        lc_means.append(m)
        lc_stds.append(s)
    # Normalised
    dem = load_static_tif(os.path.join(static_dir, "dtm_2km_domain_trim_EPSG3035.tif"), mean=dem_mean, std=dem_std)
    lat = load_static_tif(os.path.join(static_dir, "lat_2km_domain_trim_EPSG3035.tif"), mean=lat_mean, std=lat_std)
    lc = load_land_cover(os.path.join(static_dir, "land_cover_classes_2km_domain_trim_EPSG3035.tif"), means=lc_means, stds=lc_stds)

    # Compute stats for low-res 2m temperature using only the training split
    file_list = get_file_list()
    N = len(file_list) * 24
    n_val = int(val_frac * N)
    n_test = int(test_frac * N)
    n_train = N - n_val - n_test
    train_indices = list(range(n_train))
    train_file_list = [file_list[i // 24] for i in train_indices]

    low_2mt_values = []
    for hf, lf, date in train_file_list:
        low_data = decompress_zst_pt(lf)
        for hour in range(24):
            arr = low_data[hour]["2mT"].float().numpy()
            low_2mt_values.append(arr)
    low_2mt_values = np.stack(low_2mt_values)  # shape: [N, 84, 72]
    low_2mt_mean = float(np.mean(low_2mt_values))
    low_2mt_std = float(np.std(low_2mt_values))

    stats = {
        "dem": [dem_mean, dem_std],
        "lat": [lat_mean, lat_std],
        "lc_means": lc_means,
        "lc_stds": lc_stds,
        "low_2mt_mean": low_2mt_mean,
        "low_2mt_std": low_2mt_std
    }
    if save_stats_json is not None:
        with open(save_stats_json, "w") as f:
            json.dump(stats, f)
    return {"dem": dem, "lat": lat, "lc": lc}, stats


#To drastically reduce the size of the dataset, first 6 months are selected 

def compute_stats(tensor):
    mean = tensor.mean().item()
    std = tensor.std().item()
    return mean, std


def decompress_zst_pt(filepath):
    with open(filepath, "rb") as f:
        compressed = f.read()
    dctx = zstandard.ZstdDecompressor()
    decompressed = io.BytesIO(dctx.decompress(compressed))
    data = torch.load(decompressed, weights_only=False)
    return data

def load_static_tif(filepath, mean=None, std=None):
    ds = xr.open_dataset(filepath, engine="rasterio")
    arr = ds['band_data'][0].values
    arr = np.nan_to_num(arr)
    if mean is not None and std is not None:
        arr = (arr - mean) / std
    return torch.tensor(arr, dtype=torch.float32)

def load_land_cover(filepath, means=None, stds=None):
    ds = xr.open_dataset(filepath, engine="rasterio")
    bands = []
    for i in range(ds['band_data'].shape[0]):
        arr = ds['band_data'][i].values
        arr = np.nan_to_num(arr)
        if means is not None and stds is not None:
            arr = (arr - means[i]) / stds[i]
        bands.append(torch.tensor(arr, dtype=torch.float32))
    return torch.stack(bands)


def collate_fn(batch):
    """
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


def load_and_normalise(static_dir, val_frac=0.15, test_frac=0.15, save_stats_json=None):
    dem = load_static_tif(os.path.join(static_dir, "dtm_2km_domain_trim_EPSG3035.tif"))
    lat = load_static_tif(os.path.join(static_dir, "lat_2km_domain_trim_EPSG3035.tif"))
    lc = load_land_cover(os.path.join(static_dir, "land_cover_classes_2km_domain_trim_EPSG3035.tif"))
    dem_mean, dem_std = compute_stats(dem)
    lat_mean, lat_std = compute_stats(lat)
    lc_means, lc_stds = [], []
    for i in range(lc.shape[0]):
        m, s = compute_stats(lc[i])
        lc_means.append(m)
        lc_stds.append(s)
    # Normalised
    dem = load_static_tif(os.path.join(static_dir, "dtm_2km_domain_trim_EPSG3035.tif"), mean=dem_mean, std=dem_std)
    lat = load_static_tif(os.path.join(static_dir, "lat_2km_domain_trim_EPSG3035.tif"), mean=lat_mean, std=lat_std)
    lc = load_land_cover(os.path.join(static_dir, "land_cover_classes_2km_domain_trim_EPSG3035.tif"), means=lc_means, stds=lc_stds)

    # Compute stats for low-res 2m temperature using only the training split
    file_list = get_file_list()
    N = len(file_list) * 24
    n_val = int(val_frac * N)
    n_test = int(test_frac * N)
    n_train = N - n_val - n_test
    train_indices = list(range(n_train))
    train_file_list = [file_list[i // 24] for i in train_indices]

    low_2mt_values = []
    for hf, lf, date in train_file_list:
        low_data = decompress_zst_pt(lf)
        for hour in range(24):
            arr = low_data[hour]["2mT"].float().numpy()
            low_2mt_values.append(arr)
    low_2mt_values = np.stack(low_2mt_values)  # shape: [N, 84, 72]
    low_2mt_mean = float(np.mean(low_2mt_values))
    low_2mt_std = float(np.std(low_2mt_values))

    stats = {
        "dem": [dem_mean, dem_std],
        "lat": [lat_mean, lat_std],
        "lc_means": lc_means,
        "lc_stds": lc_stds,
        "low_2mt_mean": low_2mt_mean,
        "low_2mt_std": low_2mt_std
    }
    if save_stats_json is not None:
        with open(save_stats_json, "w") as f:
            json.dump(stats, f)
    return {"dem": dem, "lat": lat, "lc": lc}, stats


#You can change the way you sample data: here , to be mindful of capturing diurnal as well as seasonal variations, we take the first week of every month from 2020

def get_file_list():
    high_files = sorted(glob.glob(os.path.join(DATA_DIR, "2020/*_high_2mT.pt.zst")))
    low_files = sorted(glob.glob(os.path.join(DATA_DIR, "2020/*_low.pt.zst")))
    files = []
    for hf, lf in zip(high_files, low_files):
        date = os.path.basename(hf).split('_')[0]  # e.g., '2020-05-04'
        files.append((hf, lf, date))
    print(f"Using {len(files)} samples from 2020.")
    return files