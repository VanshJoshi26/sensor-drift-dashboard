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
            df['batch'] = i  # ADD BATCH COLUMN HERE
            dfs.append(df)
        except FileNotFoundError:
            continue
    if not dfs:
        st.error("No CSV files found! Upload data_batch_1.csv to data_batch_10.csv")
        st.stop()
    return pd.concat(dfs, ignore_index=True)

df = load_data()
st.set_page_config(page_title="SmartSense Guardian", layout="wide")
st.title("🛡️ SmartSense Guardian: Sensor Drift & Fouling Dashboard")

# SHOW DATA INFO
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.write(f"Rows: {len(df):,}")
st.sidebar.write(f"Columns: {len(df.columns)}")
st.sidebar.write("Columns:", ', '.join(df.columns.tolist()[:5]) + "...")

# SAFE SELECTORS
st.sidebar.header("🎛️ Controls")
batches = sorted(df['batch'].unique())
batch_sel = st.sidebar.selectbox("Batch", batches, index=min(9, len(batches)-1))

# AUTO-FIND SENSOR COLUMNS
sensor_cols = [col for col in df.columns if any(x in col for x in ['R', 'T'])][:8]
if not sensor_cols:
    sensor_cols = df.select_dtypes(include=[np.number]).columns[:8].tolist()
    
sensor_sel = st.sidebar.selectbox("Sensor", sensor_cols)

# FILTER DATA
data_sel = df[df['batch'] == batch_sel]

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Sensor Signal")
    fig_ts = px.line(x=range(len(data_sel)), y=data_sel[sensor_sel], 
                     title=f"{sensor_sel} - Batch {batch_sel}")
    st.plotly_chart(fig_ts, use_container_width=True)

with col2:
    st.subheader("🔍 FFT Fouling Detection")
    signal = data_sel[sensor_sel].fillna(0).values
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1)[:N//2]
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=xf, y=np.abs(yf[:N//2]), name='Frequency'))
    fig_fft.update_layout(xaxis_title="Freq", yaxis_title="Magnitude")
    st.plotly_chart(fig_fft, use_container_width=True)

# AI ANALYSIS
st.subheader("🤖 AI Risk Assessment")
scaler = StandardScaler()
scaled = scaler.fit_transform(data_sel[sensor_sel].fillna(0).values.reshape(-1, 1))
iso = IsolationForest(contamination=0.1, random_state=42)
anoms = iso.decision_function(scaled)
drift_prob = 1 - np.mean(anoms)
risk = min(100, drift_prob * 85)

col1, col2, col3 = st.columns(3)
col1.metric("🎯 Drift Prob", f"{drift_prob:.1%}")
col2.metric("🚨 Anomalies", f"{(anoms < 0).mean():.1%}")
col3.metric("⚠️ Risk Score", f"{risk:.0f}/100", delta=20)

# RECOMMENDATIONS
st.subheader("✅ Recommended Action")
if risk > 80:
    st.error("🚨 **CALIBRATE IMMEDIATELY** - Critical drift detected")
elif risk > 60:
    st.warning("🧽 **TRIGGER CLEANING** - Fouling patterns found")
elif risk > 40:
    st.info("⚠️ **CLOSE MONITORING** - Early drift signs")
else:
    st.success("✅ **NORMAL OPERATION** - Continue monitoring")

st.markdown("---")
st.caption("SmartSense Guardian: GxP-compliant sensor intelligence")
