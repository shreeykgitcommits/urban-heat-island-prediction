"""Pune Urban Heat Island dashboard.  Run from the repo root:  streamlit run app/app.py"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # so `models` can be imported

from uhi_inference import find_checkpoint, load_model, load_stack, predict_map  # noqa: E402

BG = "#f3f6f8"

st.set_page_config(page_title="Pune heat island monitor", layout="wide")
st.title("Pune heat island monitor")
st.write(
    "Predicted urban heat island intensity (UHII) for Pune, from satellite land surface "
    "temperature, PM2.5 and land cover."
)

files = glob.glob(os.path.join(ROOT, "data", "uhi_stack_*.tif"))
if not files:
    st.error("No data found. Copy uhi_stack_LIVE.tif and the yearly files into the data folder, then reload.")
    st.stop()
files.sort(key=lambda f: ("LIVE" not in f, f))


def label(path):
    name = os.path.basename(path).replace("uhi_stack_", "").replace(".tif", "")
    return "Latest pull (live)" if name == "LIVE" else f"Historical {name}"


options = {label(f): f for f in files}
choice = st.sidebar.selectbox("Data to show", list(options))
smooth = st.sidebar.checkbox("Blend patch seams", value=True)
st.sidebar.caption("UHII is measured against the average rural temperature in the same file. Below 0 means cooler than that baseline.")


@st.cache_resource
def get_model():
    return load_model(find_checkpoint(ROOT))


@st.cache_data
def run(path, smooth, modified):
    lst, pm25, landcover = load_stack(path)
    return predict_map(get_model(), lst, pm25, landcover, smooth)


path = options[choice]
try:
    uhii, lst_map = run(path, smooth, os.path.getmtime(path))
except Exception as err:  # show a readable message instead of a traceback
    st.error(f"Could not run the model on this file: {err}")
    st.stop()

inside = ~np.isnan(uhii)
coverage = 100 * np.mean(~np.isnan(lst_map[inside]))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Average UHII", f"{np.nanmean(uhii):+.2f} °C")
c2.metric("Hottest area", f"{np.nanmax(uhii):+.2f} °C")
c3.metric("Coolest area", f"{np.nanmin(uhii):+.2f} °C")
c4.metric("Clear-sky temperature coverage", f"{coverage:.0f}%")


def draw(data, title, unit):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#dde3e8")
    image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(image, ax=ax, shrink=0.8, label=unit)
    return fig


left, right = st.columns(2)
with left:
    fig = draw(lst_map, "Satellite land surface temperature", "°C")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Grey areas have no clear satellite reading in this period (mostly clouds).")
with right:
    fig = draw(uhii, "Predicted heat island intensity", "°C vs rural baseline")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The model fills the gaps using PM2.5 and land cover.")

with st.expander("How this works"):
    st.write(
        "Three satellite sources go in: temperature (MODIS), PM2.5 (CAMS) and land cover (GLC-FCS30D). "
        "Each has its own encoder. Attention gates then decide which source matters most at each location, "
        "and a decoder draws the heat map. The model was trained on Pune data from 2016 to 2020 "
        "(test MSE 0.35, a first version). The latest pull is the most recent data file saved in the data folder."
    )
