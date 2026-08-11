"""
Central configuration for the Urban Heat Island (UHI) prediction project.

Keeping all tunable parameters here means switching cities, date ranges,
or model hyperparameters never requires touching the data pipeline,
model, or training code.
"""

# ---------------------------------------------------------------------------
# Region of interest
# ---------------------------------------------------------------------------
# Change CITY_NAME and BOUNDARY_SOURCE to retarget the whole pipeline to a
# different city. Nothing else in the codebase needs to change.
CITY_NAME = "Pune"

# GEE FeatureCollection used to clip all rasters to the city boundary.
# FAO GAUL gives administrative boundaries for most cities worldwide.
BOUNDARY_SOURCE = "FAO/GAUL/2015/level2"
BOUNDARY_FILTER_FIELD = "ADM2_NAME"
BOUNDARY_FILTER_VALUE = CITY_NAME

# ---------------------------------------------------------------------------
# Temporal range
# ---------------------------------------------------------------------------
START_YEAR = 2015
END_YEAR = 2020  # Kept short for the proof-of-concept; widen once validated.

# ---------------------------------------------------------------------------
# Data sources (Earth Engine asset IDs)
# ---------------------------------------------------------------------------
LST_COLLECTION = "MODIS/061/MOD11A2"           # 8-day, 1km LST
PM25_COLLECTION = "ECMWF/CAMS/NRT"             # monthly PM2.5 (fallback source)
LANDCOVER_COLLECTION = "projects/sat-io/open-datasets/GLC-FCS30D/annual"

# ---------------------------------------------------------------------------
# Grid / resampling
# ---------------------------------------------------------------------------
TARGET_RESOLUTION_M = 1000   # 1 km grid, matches LST native resolution
PATCH_SIZE = 64              # pixel size of training patches (64x64)

# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
LANDCOVER_NUM_CLASSES = 37
LANDCOVER_EMBED_DIM = 8
ENCODER_BASE_CHANNELS = 32
ATTENTION_REDUCTION_RATIO = 8   # CBAM channel-attention reduction

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"
