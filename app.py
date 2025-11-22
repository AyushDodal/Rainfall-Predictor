import os
import time
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
import plotly.express as px
import folium
from streamlit.components.v1 import html as st_html

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Rainfall — Live Dashboard (Singapore)",
    layout="wide",
    initial_sidebar_state="auto",
)

# Backend endpoint (set as Streamlit secret or env)
BACKEND_URL = st.secrets.get("BACKEND_URL", os.getenv("BACKEND_URL", "http://localhost:8000"))

# How many recent points to show in chart
CHART_POINTS = 50

# Map center Singapore
SG_CENTER = (1.3521, 103.8198)
MAP_ZOOM = 12

# -------------------------
# Helpers
# -------------------------
@st.cache_data(ttl=60)
def fetch_recent_predictions(n: int = 500):
    """
    Fetch recent predictions from backend.
    Endpoint: GET {BACKEND_URL}/predictions/recent?n=...
    Returns a DataFrame.
    """
    url = f"{BACKEND_URL}/predictions/recent?n={n}"
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        payload = r.json()          # {"items": [...], "count": ...}
        items = payload.get("items", [])
        df = pd.DataFrame(items)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        return df
    except Exception as e:
        print(f"fetch_recent_predictions failed: {e}")
        return pd.DataFrame()



def sample_data():
    """Small sample DataFrame if backend not available (for local dev)."""
    now = pd.Timestamp.utcnow()
    rows = []
    coords = [
        ("sg-01", 1.29027, 103.851959),
        ("sg-02", 1.352083, 103.819836),
        ("sg-03", 1.280095, 103.850949),
        ("sg-04", 1.300000, 103.800000),
        ("sg-05", 1.320000, 103.830000),
    ]
    for i, (lid, lat, lon) in enumerate(coords):
        rows.append({
            "location_id": lid,
            "lat": lat,
            "lon": lon,
            "timestamp": now - pd.Timedelta(minutes=i),
            "probability": float(max(0, min(1, 0.2 + i * 0.15))),
            "predicted_label": int(i % 2 == 0),
            "model_version": "vLOCAL"
        })
    return pd.DataFrame(rows)


# Produce a tiny stable tuple of records used to build the map cache key
def records_from_df(df: pd.DataFrame):
    if df.empty:
        return tuple()
    latest = df.sort_values("timestamp").groupby("location_id", as_index=False).last()
    recs = []
    for _, r in latest.iterrows():
        recs.append((
            str(r["location_id"]),
            float(r["lat"]),
            float(r["lon"]),
            float(r.get("probability", 0.0)),
            int(r.get("predicted_label", 0))
        ))
    return tuple(recs)


@st.cache_data(ttl=30)
def make_map_html(records, center=SG_CENTER, zoom=MAP_ZOOM):
    """
    Build folium map and return HTML. Cached to avoid reinitialisation blink.
    records: tuple of (location_id, lat, lon, prob, label)
    """
    m = folium.Map(location=center, zoom_start=zoom, control_scale=True)
    # add tiles for nicer look (optional). Comment/update if external tile access blocked.
    folium.TileLayer("OpenStreetMap").add_to(m)

    for lid, lat, lon, prob, lbl in records:
        icon_html = "💧" if (lbl == 1 or prob >= 0.5) else "⚪"
        popup_html = f"<b>{lid}</b><br/>Prob: {prob:.2f}<br/>Predicted: {lbl}"
        folium.Marker(
            location=(lat, lon),
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.DivIcon(html=f"<div style='font-size:22px'>{icon_html}</div>")
        ).add_to(m)

    # Return stable HTML representation
    return m._repr_html_()


# -------------------------
# UI - Background & header
# -------------------------
BG_IMAGE = st.secrets.get("BG_IMAGE_URL", os.getenv("BG_IMAGE_URL", ""))
if BG_IMAGE:
    page_bg_css = f"""
    <style>
    .stApp {{
      background-image: url("{BG_IMAGE}");
      background-size: cover;
      background-attachment: fixed;
      background-position: center;
      background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_css, unsafe_allow_html=True)

st.markdown(
    "<div style='padding:18px 0px 6px 0px'><h1 style='margin:0px'>Rainfall — Live Dashboard (Singapore)</h1>"
    "<div style='color: #6c757d;'>Real-time rainfall probabilities — OpenWeatherMap → XGBoost pipeline</div></div>",
    unsafe_allow_html=True
)

# -------------------------
# Layout: two columns (map | right sidebar)
# -------------------------
map_col, right_col = st.columns([2.2, 1])

# Sidebar summary (small)
with st.sidebar:
    st.header("Live stats")
    st.write("This dashboard fetches recent predictions from the model server.")
    st.markdown("---")
    last_update_text = st.empty()
    model_version_text = st.empty()
    st.markdown("---")
    st.caption("Controls")
    REFRESH = st.button("Refresh now (force)")

# -------------------------
# Fetch Data (use _force to bypass cache on button press)
# -------------------------
#_force = int(time.time()) if REFRESH else None

if REFRESH:
    st.cache_data.clear()
    st.rerun()


df = fetch_recent_predictions(n=1000)
if df.empty:
    df = sample_data()
    st.warning("Using local sample data (backend not reachable). Set BACKEND_URL and redeploy.")
else:
    st.success(f"Fetched {len(df):,} recent predictions")

# Update last update & model info
if not df.empty and "timestamp" in df.columns:
    latest_ts = df["timestamp"].max()
    # ensure tz-aware string if possible
    try:
        last_update_text.write(f"Last updated: {latest_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    except Exception:
        last_update_text.write(f"Last updated: {latest_ts}")
else:
    last_update_text.write("Last updated: -")

if "model_version" in df.columns and not df["model_version"].isna().all():
    model_version_text.write(f"Model: {df['model_version'].mode().iat[0]}")
else:
    model_version_text.write("Model: -")

# -------------------------
# Right column cards (summary metrics)
# -------------------------
with right_col:
    st.header("Quick overview")
    total_ingested = df.shape[0]
    if "probability" in df.columns and "predicted_label" in df.columns:
        avg_prob = df["probability"].mean()
        pct_rain = 100.0 * (df["predicted_label"].sum() / max(1, len(df)))
    else:
        avg_prob = None
        pct_rain = None

    st.metric("Total ingested (recent rows)", f"{total_ingested:,}")
    st.metric("Avg predicted prob (recent)", f"{avg_prob:.2f}" if avg_prob is not None else "—")
    st.metric("Predicted rain (%)", f"{pct_rain:.1f}%" if pct_rain is not None else "—")

    st.markdown("---")
    st.subheader("Model & Info")
    st.write("Model details, notes, and metrics can be fetched from the model server.")
    if st.button("Reload model info"):
        st.cache_data.clear()  # clear cached fetches
        st.rerun()


# -------------------------
# Map: folium with markers (cached HTML => no blinking)
# -------------------------
with map_col:
    st.subheader("Map (click marker for details)")

    records = records_from_df(df)
    map_html = make_map_html(records, center=SG_CENTER, zoom=MAP_ZOOM)
    # render cached HTML (no blinking)
    st_html(map_html, height=600, scrolling=False)

# -------------------------
# Live Chart: rolling average of probability
# -------------------------
with st.container():
    st.subheader("Model — live average probability (recent points)")
    if not df.empty and "probability" in df.columns:
        df_sorted = df.sort_values("timestamp")
        df_grouped = df_sorted.groupby("timestamp").agg({"probability": "mean"}).reset_index()
        df_grouped = df_grouped.tail(CHART_POINTS)
        if not df_grouped.empty:
            fig = px.line(df_grouped, x="timestamp", y="probability",
                          labels={"timestamp": "Time", "probability": "Avg probability"},
                          range_y=[0, 1],
                          height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No aggregated points for chart.")
    else:
        st.info("No probability data to plot.")

# -------------------------
# Footer: project description & tips
# -------------------------
st.markdown("---")
with st.expander("About this project"):
    st.write(
        """
        This demo ingests OpenWeatherMap observations for Singapore, computes features & labels,
        trains an XGBoost classifier with time-split evaluation, and serves predictions via a FastAPI endpoint.
        """
    )

# Note: we intentionally do not auto-refresh continuously to avoid blinking.
# Use the "Refresh now (force)" button to force an immediate refresh (it passes a changing _force to the cached fetch).
