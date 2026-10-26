"""Inference helpers for the UHI dashboard (no Streamlit code here)."""

import os

import numpy as np
import tifffile
import torch

from models.uhi_model import UHIModel

# Saved from the training run (computed on the training split only).
LST_MEAN, LST_STD = 35.650158, 2.4236267
PM25_MEAN, PM25_STD = 41.769737, 18.954445

# GLC-FCS30D code -> class index. Stored +1 so that 0 means "no data" (as in training).
CLASS_TO_INDEX = {10: 0, 11: 1, 20: 2, 52: 3, 62: 4, 72: 5, 182: 6, 183: 7, 190: 8, 210: 9}

PATCH, STRIDE = 32, 8


def find_checkpoint(root):
    for folder in ("Checkpoints", "checkpoints"):
        path = os.path.join(root, folder, "uhi_model_v1.pth")
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find uhi_model_v1.pth inside the Checkpoints folder.")


def load_model(path):
    model = UHIModel(
        input_size=(PATCH, PATCH),
        base_channels=32,
        landcover_num_classes=11,
        landcover_embed_dim=8,
        reduction_ratio=8,
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def load_stack(path):
    """Reads a 3-band GeoTIFF exported from Earth Engine: LST, PM2.5, land cover."""
    data = tifffile.imread(path)
    if data.shape[0] != 3:  # some files store bands last
        data = np.moveaxis(data, -1, 0)
    lst, pm25, landcover = data
    return lst, pm25, landcover


def _starts(size):
    starts = list(range(0, size - PATCH + 1, STRIDE))
    if starts[-1] != size - PATCH:  # make sure the last rows/columns are covered too
        starts.append(size - PATCH)
    return starts


def predict_map(model, lst, pm25, landcover, smooth=True):
    """Predicts UHII for the whole city by sliding 32x32 patches and blending them.

    Returns (uhii_map, lst_map). Pixels outside the city boundary are NaN.
    """
    if np.isnan(lst).all() or np.isnan(pm25).all():
        raise ValueError("This file has no valid temperature or PM2.5 pixels.")

    outside = np.isnan(landcover)  # land cover is clipped to the boundary
    lst_filled = np.where(np.isnan(lst), np.nanmean(lst), lst)
    pm25_filled = np.where(np.isnan(pm25), np.nanmean(pm25), pm25)

    lc_idx = np.zeros(landcover.shape, dtype=np.int64)
    for code, idx in CLASS_TO_INDEX.items():
        lc_idx[landcover == code] = idx + 1

    lst_n = (lst_filled - LST_MEAN) / LST_STD
    pm25_n = (pm25_filled - PM25_MEAN) / PM25_STD

    if smooth:  # centre-weighted blending hides the seams between patches
        win = np.hanning(PATCH + 2)[1:-1]
        weight = np.outer(win, win)
    else:
        weight = np.ones((PATCH, PATCH))

    height, width = lst.shape
    total = np.zeros((height, width))
    weights = np.zeros((height, width))
    positions = [(y, x) for y in _starts(height) for x in _starts(width)]

    with torch.no_grad():
        for i in range(0, len(positions), 16):
            chunk = positions[i:i + 16]
            lst_b = torch.tensor(np.stack([lst_n[y:y + PATCH, x:x + PATCH] for y, x in chunk]), dtype=torch.float32).unsqueeze(1)
            pm_b = torch.tensor(np.stack([pm25_n[y:y + PATCH, x:x + PATCH] for y, x in chunk]), dtype=torch.float32).unsqueeze(1)
            lc_b = torch.tensor(np.stack([lc_idx[y:y + PATCH, x:x + PATCH] for y, x in chunk]), dtype=torch.long)
            preds = model(lst_b, pm_b, lc_b).numpy()[:, 0]
            for (y, x), pred in zip(chunk, preds):
                total[y:y + PATCH, x:x + PATCH] += pred * weight
                weights[y:y + PATCH, x:x + PATCH] += weight

    uhii = total / np.maximum(weights, 1e-9)
    uhii[outside] = np.nan
    lst_map = np.where(outside, np.nan, lst)
    return uhii, lst_map
