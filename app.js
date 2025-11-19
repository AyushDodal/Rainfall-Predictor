// app.js - static frontend logic
// Configure these before deploying
const API_BASE = (window.__API_BASE__ || '') || 'http://localhost:8000'; // <-- set your public backend here when deploying
const MAPBOX_TOKEN = ''; // optional: add your Mapbox token here for Mapbox tiles

// how often to refresh (ms)
const REFRESH_INTERVAL = 30_000;
const CHART_POINTS = 40;

let map, markers = {};
let chart;

function chooseTileLayer() {
  if (MAPBOX_TOKEN && MAPBOX_TOKEN.length > 5) {
    return L.tileLayer(`https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{z}/{x}/{y}?access_token=${MAPBOX_TOKEN}`, {
      tileSize: 512, zoomOffset: -1, attribution: '© Mapbox © OpenStreetMap'
    });
  }
  // fallback to OSM tiles
  return L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
  });
}

function initMap() {
  map = L.map('map', {center: [1.3521, 103.8198], zoom: 11});
  chooseTileLayer().addTo(map);
}

function makeMarker(st) {
  // raindrop emoji icon for high prob; blue dot otherwise
  const prob = (st.pred_prob || 0);
  const isRain = prob >= 0.3; // threshold to show raindrop
  const emoji = isRain ? '💧' : '📍';
  const icon = L.divIcon({
    html: `<div style="font-size:20px;line-height:1">${emoji}</div>`,
    className: 'my-div-icon',
    iconSize: [24, 24],
    iconAnchor: [12, 24]
  });
  return L.marker([st.lat, st.lon], {icon});
}

async function fetchPredictions() {
  try {
    const res = await fetch(`${API_BASE}/predictions/recent`);
    if (!res.ok) throw new Error('Failed to fetch predictions');
    const body = await res.json();
    return body.results || [];
  } catch (e) {
    console.error(e);
    return [];
  }
}

async function fetchModelMeta() {
  try {
    const res = await fetch(`${API_BASE}/models/latest`);
    if (!res.ok) return null;
    const body = await res.json();
    return body;
  } catch (e) {
    return null;
  }
}

function updateStatsFromPreds(preds) {
  // compute aggregated stats locally if no /stats endpoint provided
  const total = preds.length;
  document.getElementById('stat-ingested').innerText = total.toLocaleString();
  // training samples & metrics might be provided by /models/latest; leave blank for now
}

function updateModelPanel(meta) {
  if (!meta) return;
  document.getElementById('model-version').innerText = `Model: ${meta.model_version || meta.model}`;
  document.getElementById('last-updated').innerText = `Last updated: ${meta.created_at || meta.updated || ''}`;
  if (meta.metrics) {
    document.getElementById('stat-trained').innerText = (meta.n_samples||'—');
    document.getElementById('stat-auc').innerText = (meta.metrics.roc_auc||'—');
    document.getElementById('stat-prec').innerText = (meta.metrics.precision||'—');
    document.getElementById('stat-rec').innerText = (meta.metrics.recall||'—');
  }
}

function updateMap(preds) {
  // expects each pred doc to have location_id, lat, lon, pred_prob, scored_at
  // derive latest per location
  const latest = {};
  preds.forEach(p => {
    if (!p.location_id) return;
    if (!latest[p.location_id] || new Date(p.scored_at) > new Date(latest[p.location_id].scored_at)) {
      latest[p.location_id] = p;
    }
  });
  // add / update markers
  Object.values(latest).forEach(st => {
    if (!st.lat || !st.lon) return;
    if (markers[st.location_id]) {
      markers[st.location_id].setLatLng([st.lat, st.lon]);
      markers[st.location_id].bindPopup(popupHtml(st));
    } else {
      const m = makeMarker(st).addTo(map);
      m.bindPopup(popupHtml(st));
      markers[st.location_id] = m;
    }
  });
}

function popupHtml(st) {
  const prob = (st.pred_prob*100).toFixed(1);
  return `<div style="font-size:13px">
    <div style="font-weight:700">${st.location_id}</div>
    <div>Prob (next hour): <strong>${prob}%</strong></div>
    <div style="font-size:11px;color:#666">${new Date(st.scored_at).toLocaleString()}</div>
  </div>`;
}

function initChart() {
  const ctx = document.getElementById('liveChart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Avg predicted probability',
        data: [],
        borderColor: 'rgba(14,165,233,0.95)',
        backgroundColor: 'rgba(14,165,233,0.12)',
        tension: 0.25,
        pointRadius: 0
      }]
    },
    options: {
      scales: { x: { display: true }, y: { min:0, max:1 } },
      plugins: { legend: { display: false } }
    }
  });
}

function pushChartPoint(value) {
  const now = new Date().toLocaleTimeString();
  chart.data.labels.push(now);
  chart.data.datasets[0].data.push(value);
  if (chart.data.labels.length > CHART_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update();
}

async function refreshAll() {
  const preds = await fetchPredictions();
  if (!preds.length) return;
  updateStatsFromPreds(preds);
  updateMap(preds);
  // compute avg pred across most recent records
  const avg = preds.reduce((s,p)=>s + (p.pred_prob||0), 0) / preds.length;
  pushChartPoint(avg);
  // fetch model meta if available
  const meta = await fetchModelMeta();
  if (meta) updateModelPanel(meta);
}

async function boot() {
  initMap();
  initChart();
  await refreshAll();
  setInterval(refreshAll, REFRESH_INTERVAL);
  document.getElementById('refreshBtn').addEventListener('click', refreshAll);
}

document.addEventListener('DOMContentLoaded', boot);
