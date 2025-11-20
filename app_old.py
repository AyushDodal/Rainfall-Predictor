# app.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components

st.set_page_config(page_title="Rainfall — Live Dashboard", layout="wide")

BACKEND = st.secrets.get("BACKEND_URL", "http://localhost:8000")  # set in Streamlit Cloud secrets

st.markdown("# Rainfall — Live Dashboard (Singapore)")
st.markdown("Real-time rainfall probabilities — OpenWeatherMap → XGBoost pipeline")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Map (click marker for details)")
    # We'll embed a small leaflet map via HTML (populated from backend)
    MAP_HTML_PLACEHOLDER = st.empty()

with col2:
    st.subheader("Live stats")
    # placeholders
    stat_ing = st.empty()
    stat_train = st.empty()
    stat_auc = st.empty()
    st.divider()
    st.markdown("**About**")
    st.markdown("Ingest → features → label → train → serve (FastAPI).")

# fetch data
@st.cache_data(ttl=15)
def fetch_data():
    try:
        r = requests.get(f"{BACKEND}/predictions/recent", timeout=8)
        preds = r.json().get("results", [])
        # convert to DataFrame
        df = pd.DataFrame(preds)
        if not df.empty:
            df["scored_at"] = pd.to_datetime(df["scored_at"])
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

df = fetch_data()

# update stats
stat_ing.markdown(f"**Total ingested (recent rows):** {len(df):,}")
# train & metrics from /models/latest if available
try:
    meta = requests.get(f"{BACKEND}/models/latest", timeout=6).json()
except:
    meta = {}

stat_train.markdown(f"**Training samples:** {meta.get('n_samples', '—')}")
stat_auc.markdown(f"**AUC (test)**: {meta.get('metrics', {}).get('roc_auc', '—')}")

# build a simple Leaflet map embedding with markers
def make_leaflet_html(rows):
    if rows.empty:
        return "<div>No data yet</div>"
    # keep unique latest per location
    latest = rows.sort_values("scored_at").groupby("location_id").tail(1)
    markers_js = []
    for _, r in latest.iterrows():
        lat = r.get("lat") or r.get("payload.coord.lat") or r.get("features.lat")
        lon = r.get("lon") or r.get("payload.coord.lon") or r.get("features.lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        prob = float(r.get("pred_prob", 0))
        label = r.get("location_id", "")
        popup = f"{label}<br>Prob: {prob*100:.1f}%<br>{r['scored_at']}"
        markers_js.append(f"L.marker([{lat},{lon}]).addTo(mymap).bindPopup({popup!r});")
    markers_code = "\n".join(markers_js)
    html = f"""
    <div id='mapid' style='height:480px;'></div>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
      var mymap = L.map('mapid').setView([1.3521,103.8198], 11);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors'
      }}).addTo(mymap);
      {markers_code}
    </script>
    """
    return html

MAP_HTML_PLACEHOLDER.components.html(make_leaflet_html(df), height=500)

# show a simple live chart of average prob
st.subheader("Live average probability")
if not df.empty:
    df_recent = df.set_index("scored_at").resample("1T").pred_prob.mean().fillna(0).tail(60)
    st.line_chart(df_recent)
else:
    st.info("No prediction data yet.")

# footer
st.caption("Live app — set BACKEND_URL in Streamlit secrets to your public FastAPI endpoint.")
