import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft, fftfreq
import warnings
import os
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SmartSense Guardian", layout="wide")
st.title("🛡️ SmartSense Guardian: Sensor Intelligence")

# SIMULATED DATA - Works EVEN WITHOUT CSV FILES
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n_samples = 1000
    data = {
        'R9_1': np.random.normal(0.8, 0.1, n_samples) + np.linspace(0, 0.2, n_samples),  # Drift
        'R9_2': np.random.normal(0.9, 0.15, n_samples),
        'T_1': np.random.normal(25, 2, n_samples),
        'T_2': np.random.normal(26, 2.5, n_samples),
        'batch': np.repeat([1,5,10], n_samples//3),
        'analyte': np.tile(['Ethanol', 'Ethylene', 'Ammonia'], n_samples//3)
    }
    return pd.DataFrame(data)

# TRY LOADING CSV, FALLBACK TO SIMULATED
try:
    df = pd.read_csv('data_batch_1.csv')  # Test one file
    st.success("✅ Real data loaded!")
    # Load all batches here...
except:
    df = create_sample_data()
    st.info("🎲 Using simulated Gas Sensor data (CSV files optional)")

st.sidebar.markdown("### 📊 Dataset")
st.sidebar.metric("Rows", len(df))
st.sidebar.metric("Sensors", len([c for c in df.columns if 'R' in c or 'T' in c]))

# CONTROLS
st.sidebar.header("🎛️ Controls")
batch_sel = st.sidebar.selectbox("Batch", sorted(df['batch'].unique()), index=2)
sensor_cols = [c for c in df.columns if 'R' in c or 'T' in c]
sensor_sel = st.sidebar.selectbox("Sensor", sensor_cols)

data_sel = df[df['batch'] == batch_sel][['batch', sensor_sel]]

# DASHBOARD LAYOUT
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Sensor Signal")
    fig_ts = px.line(data_sel, x=data_sel.index, y=sensor_sel, 
                    title=f"{sensor_sel} - Batch {batch_sel}")
    st.plotly_chart(fig_ts, use_container_width=True)

with col2:
    st.subheader("🔍 FFT Analysis")
    signal = data_sel[sensor_sel].fillna(0).values
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1)[:N//2]
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=xf, y=np.abs(yf[:N//2]), name='Spectrum'))
    fig_fft.update_layout(height=300)
    st.plotly_chart(fig_fft, use_container_width=True)

# AI ENGINE
st.subheader("🤖 Risk Assessment")
scaler = StandardScaler()
scaled = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
iso = IsolationForest(contamination=0.1, random_state=42)
anoms = iso.decision_function(scaled.reshape(-1, 1))
drift = 1 - np.mean(anoms)
risk = min(100, drift * 90)

col1, col2, col3 = st.columns(3)
col1.metric("🎯 Drift", f"{drift:.1%}")
col2.metric("🚨 Anomalies", f"{(anoms < 0).mean():.1%}")
col3.metric("⚠️ Risk Score", f"{risk:.0f}/100")

# ACTION RECOMMENDATIONS
st.subheader("✅ Recommended Action")
if risk > 80:
    st.error("🚨 **CALIBRATE NOW** - Critical drift detected")
elif risk > 60:
    st.warning("🧽 **CLEAN SENSOR** - Fouling detected") 
elif risk > 40:
    st.info("⚠️ **MONITOR** - Early warning")
else:
    st.success("✅ **OK** - Normal operation")

st.markdown("---")
st.caption("💡 Upload CSV files to GitHub for real Gas Sensor Array Drift Dataset")
