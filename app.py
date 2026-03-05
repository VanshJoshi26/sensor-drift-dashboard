import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_data():
    dfs = []
    for i in range(1, 11):
        try:
            df = pd.read_csv(f'data_batch_{i}.csv')
            # ADD BATCH COLUMN IF MISSING
            if 'batch' not in df.columns:
                df['batch'] = i
            dfs.append(df)
        except FileNotFoundError:
            st.error(f"Missing data_batch_{i}.csv")
            return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

df = load_data()
if df.empty:
    st.stop()

# DEBUG: SHOW COLUMNS
st.sidebar.markdown("### 📋 Available Columns")
st.sidebar.write(df.columns.tolist())

st.set_page_config(page_title="SmartSense Guardian", layout="wide")
st.title("🛡️ SmartSense Guardian: Sensor Drift & Fouling Dashboard")

# ROBUST SIDEBAR - Works with ANY columns
st.sidebar.header("Inputs")
batch_sel = st.sidebar.selectbox("Batch", sorted(df['batch'].unique()), index=-1)

# Find sensor columns (R* or T*)
sensor_cols = [col for col in df.columns if 'R' in col or 'T' in col]
if sensor_cols:
    sensor_sel = st.sidebar.selectbox("Sensor", sensor_cols[:8])
else:
    sensor_sel = df.columns[0]  # First numeric column

# Analyte - SAFE version
if 'analyte' in df.columns:
    analyte_sel = st.sidebar.selectbox("Analyte", sorted(df['analyte'].unique()))
    data_sel = df[(df['batch'] == batch_sel) & (df['analyte'] == analyte_sel)]
else:
    st.sidebar.warning("No 'analyte' column found")
    data_sel = df[df['batch'] == batch_sel]

# PLOTS
col1, col2 = st.columns(2)
with col1:
    fig_ts = px.line(data_sel, x=data_sel.index, y=sensor_sel, 
                     title=f"Sensor {sensor_sel} - Batch {batch_sel}")
    st.plotly_chart(fig_ts, use_container_width=True)

with col2:
    signal = data_sel[sensor_sel].values
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1)[:N//2]
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=xf, y=np.abs(yf[:N//2]), name='FFT'))
    fig_fft.update_layout(title="FFT: Fouling Detection")
    st.plotly_chart(fig_fft)

# METRICS
scaler = StandardScaler()
scaled_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
iso_forest = IsolationForest(contamination=0.1)
anomaly_scores = iso_forest.decision_function(scaled_signal.reshape(-1, 1))
drift_prob = 1 - np.mean(anomaly_scores)

risk_score = min(100, drift_prob * 100 * 0.8)

col1, col2, col3 = st.columns(3)
col1.metric("Drift Probability", f"{drift_prob:.1%}")
col2.metric("Anomaly Ratio", f"{(anomaly_scores < 0).mean():.1%}")
col3.metric("Risk Score", f"{risk_score:.0f}/100")

# RECOMMENDATIONS
if risk_score > 80:
    st.error("🚨 **CALIBRATE IMMEDIATELY**")
elif risk_score > 60:
    st.warning("🧽 **TRIGGER CLEANING**")
else:
    st.success("✅ **Normal Operation**")
