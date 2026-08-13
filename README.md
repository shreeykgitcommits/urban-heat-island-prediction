# Urban Heat Island Intensity Prediction — Pune

Predicts Urban Heat Island Intensity (UHII) for Pune using a multi-stream
deep learning model that fuses satellite Land Surface Temperature (LST),
PM2.5 air quality, and land cover data.

## Overview

Urban areas tend to be significantly warmer than their surrounding rural
areas due to human activity, dense construction, and reduced vegetation —
a phenomenon known as the Urban Heat Island (UHI) effect. This project
predicts UHI Intensity at 1km resolution across Pune using a custom
multi-stream CNN with attention-based fusion.

## Data Sources

All data is pulled via the Google Earth Engine Python API:

| Source | Dataset | Resolution |
|---|---|---|
| Land Surface Temperature | MODIS MOD11A2 | 1km, 8-day composite |
| PM2.5 | ECMWF CAMS Near-Real-Time | ~40km, 3-hourly (averaged to annual) |
| Land Cover | GLC-FCS30D | 30m, annual (resampled to 1km) |

Data covers **2016–2020** (limited by PM2.5 data availability, which starts
mid-2016), clipped to Pune's administrative boundary.

## Model Architecture

- **Raster Encoder** (used separately for LST and PM2.5): 3-stage
  Conv-BatchNorm-ReLU stack with stride-2 downsampling.
- **Land Cover Encoder**: category embedding layer → CoordConv (adds
  spatial position awareness) → residual dilated convolution block →
  CBAM attention (channel + spatial).
- **Attention Fusion U-Net**: concatenates all three encoded streams,
  applies attention gates to weight each stream's spatial contribution,
  and decodes back to a full-resolution UHII prediction map.

All modules are implemented from scratch in PyTorch (see `models/`).

## Results (v1)

- Dataset: 1,045 overlapping 32×32 patches extracted from 5 years of data
  via sliding-window sampling (needed due to Pune's relatively small
  169×205 pixel grid at 1km resolution).
- Split: 80% train / 10% val / 10% test
- **Test MSE: 0.3497**
- Training curve and sample prediction visualizations are in `outputs/`.

This is a first-pass, from-scratch result on a genuinely small dataset —
future work includes expanding the date range, adding data augmentation,
and tuning hyperparameters to close the gap further.

## Project Structure

```
├── config.py                  # central configuration (city, dates, hyperparameters)
├── models/
│   ├── cbam.py                 # channel + spatial attention module
│   ├── raster_encoder.py       # encoder for LST / PM2.5
│   ├── landcover_encoder.py    # encoder for categorical land cover
│   ├── attention_unet.py       # attention-gated fusion + decoder
│   └── uhi_model.py            # full end-to-end model
├── notebooks/
│   └── data_pipeline_and_training.ipynb   # GEE data pull, preprocessing, training
├── outputs/
│   ├── prediction_comparison.png
│   └── training_curve.png
└── checkpoints/
    └── uhi_model_v1.pth
```

## Acknowledgements

Architecture inspired by published research on multi-modal attention
fusion for UHI prediction. All code in this repository was written
independently.
