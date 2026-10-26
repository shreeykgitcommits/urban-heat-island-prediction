"""Pune Urban Heat Island dashboard.  Run from the repo root:  streamlit run app/app.py"""

import glob
import os
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # so `models` can be imported

from uhi_inference import (  # noqa: E402
    LAND_COVER_NAMES, find_checkpoint, load_geo, load_model, load_stack, predict_map,
)

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
    if name == "LIVE":
        return "Latest pull (live)"
    if name.endswith("PROJECTED"):
        return "Projected " + name.split("_")[0] + " (estimate)"
    return f"Historical {name}"


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
    uhii, lst_map = predict_map(get_model(), lst, pm25, landcover, smooth)
    pm25_map = np.where(np.isnan(landcover), np.nan, pm25)
    return uhii, lst_map, pm25_map, landcover


path = options[choice]
try:
    uhii, lst_map, pm25_map, landcover = run(path, smooth, os.path.getmtime(path))
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

# ---- hover information -------------------------------------------------
height, width = uhii.shape
geo = load_geo(path)
if geo:  # real longitude / latitude of every pixel centre
    west, north, dx, dy = geo
    xs = west + (np.arange(width) + 0.5) * dx
    ys = north - (np.arange(height) + 0.5) * dy
    where = "Lon %{x:.3f}°E, Lat %{y:.3f}°N"
    xlab, ylab = "Longitude (°E)", "Latitude (°N)"
else:  # file has no location tags: fall back to pixel numbers
    xs, ys = np.arange(width), height - 1 - np.arange(height)
    where = "Column %{x}, row %{y} (from bottom)"
    xlab, ylab = "Column", "Row (from bottom)"

land = np.full(landcover.shape, "No data", dtype="<U40")
for code, name in LAND_COVER_NAMES.items():
    land[landcover == code] = name


def as_text(values, fmt, unit, missing):
    shown = np.char.add(np.char.mod(fmt, values), unit)
    return np.where(np.isnan(values), missing, shown)


lst_txt = as_text(lst_map, "%.1f", " °C", "no clear reading")
pm_txt = as_text(pm25_map, "%.0f", " µg/m³", "no data")
land_text = np.char.add("Land: ", land)
uhii_text = np.char.add(
    np.char.add(np.char.add("Surface temp: ", lst_txt), "<br>PM2.5: "),
    np.char.add(np.char.add(pm_txt, "<br>Land: "), land),
)


def draw(z, text, title, bar, template):
    fig = go.Figure(go.Heatmap(
        z=z, x=xs, y=ys, text=text, colorscale="Magma", hoverongaps=False,
        hovertemplate=template, colorbar=dict(title=bar, thickness=14),
    ))
    fig.update_layout(
        title=title, height=470, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor=BG, plot_bgcolor="#dde3e8", font=dict(color="#14212b"),
        xaxis=dict(title=xlab, showgrid=False),
        yaxis=dict(title=ylab, showgrid=False, scaleanchor="x"),
    )
    return fig


st.caption("Move the cursor over a map to see the temperature, position and land type at that spot.")
left, right = st.columns(2)
with left:
    st.plotly_chart(draw(
        lst_map, land_text, "Satellite land surface temperature", "°C",
        "<b>%{z:.1f} °C</b> surface temperature<br>" + where + "<br>%{text}<extra></extra>",
    ))
    st.caption("Grey areas have no clear satellite reading in this period (mostly clouds).")
with right:
    st.plotly_chart(draw(
        uhii, uhii_text, "Predicted heat island intensity", "°C vs rural",
        "<b>%{z:+.2f} °C</b> vs rural baseline<br>" + where + "<br>%{text}<extra></extra>",
    ))
    st.caption("The model fills the gaps using PM2.5 and land cover.")

with st.expander("How this works"):
    st.write(
        "Three satellite sources go in: temperature (MODIS), PM2.5 (CAMS) and land cover (GLC-FCS30D). "
        "Each has its own encoder. Attention gates then decide which source matters most at each location, "
        "and a decoder draws the heat map. The model was trained on Pune data from 2016 to 2020 "
        "(test MSE 0.35, a first version). The latest pull is the most recent data file saved in the data folder."
    )
