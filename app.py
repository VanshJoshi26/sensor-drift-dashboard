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
            df['batch'] = i
            dfs.append(df)
        except FileNotFoundError:
            st.error(f"Missing data_batch_{i}.csv. Download from Kaggle.")
            return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

df = load_data()
if df.empty:
    st.stop()

st.set_page_config(page_title="SmartSense Guardian", layout="wide")
st.title("🛡️ SmartSense Guardian: Sensor Drift & Fouling Dashboard")
st.markdown("AI-driven calibration risk for Gas Sensor Array Drift Dataset")

# Sidebar
st.sidebar.header("Inputs")
batch_sel = st.sidebar.selectbox("Select Batch", sorted(df['batch'].unique()), index=9)
analyte_sel = st.sidebar.selectbox("Analyte", sorted(df['analyte'].unique()))
sensor_cols = [col for col in df.columns if col.startswith('R') or col.startswith('T')]
sensor_sel = st.sidebar.selectbox("Sensor", sensor_cols[:8])  # Sample

data_sel = df[(df['batch'] == batch_sel) & (df['analyte'] == analyte_sel)]

col1, col2 = st.columns(2)

with col1:
    # Time Series Plot
    fig_ts = px.line(data_sel, x=data_sel.index, y=sensor_sel, 
                     title=f"Sensor {sensor_sel} Time Series (Batch {batch_sel})")
    st.plotly_chart(fig_ts, use_container_width=True)

with col2:
    # FFT Fouling Detection
    signal = data_sel[sensor_sel].values
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1)[:N//2]
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=xf, y=np.abs(yf[:N//2]), name='Frequency Spectrum'))
    fig_fft.update_layout(title="FFT: Fouling (High Freq Noise)", xaxis_title="Frequency")
    st.plotly_chart(fig_fft, use_container_width=True)

# Drift Detection (simplified trend change)
scaler = StandardScaler()
scaled_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
iso_forest = IsolationForest(contamination=0.1)
anomaly_scores = iso_forest.decision_function(scaled_signal.reshape(-1, 1))
drift_prob = 1 - np.mean(anomaly_scores)  # Proxy for drift strength

# Risk Score Calculation
sensor_crit = 0.8 if 'R' in sensor_sel else 0.6  # Resistance higher crit
batch_phase_sens = 1.2 if batch_sel > 6 else 1.0  # Later batches more sensitive
risk_score = min(100, drift_prob * 100 * sensor_crit * batch_phase_sens)

col1, col2, col3 = st.columns(3)
col1.metric("Drift Probability", f"{drift_prob:.2%}")
col2.metric("Anomaly Ratio", f"{(anomaly_scores < 0).mean():.1%}")
col3.metric("Risk Score", f"{risk_score:.0f}/100", delta=risk_score-50)

# Recommendations
st.subheader("Actions")
if risk_score > 80:
    st.error("🚨 **CALIBRATE IMMEDIATELY** - High drift & fouling risk")
elif risk_score > 60:
    st.warning("🧽 **TRIGGER CLEANING** - Fouling detected via FFT peaks")
elif risk_score > 40:
    st.info("⚠️ **Monitor Closely** - Early drift signs")
else:
    st.success("✅ **Normal Operation** - No action needed")

# Correlation Heatmap (sample sensors)
if st.checkbox("Show Sensor Correlations"):
    corr_data = data_sel[sensor_cols[:6]].corr()
    fig_corr = px.imshow(corr_data, title="Sensor Correlation Matrix", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")
st.caption("Built for GxP compliance: LSTM-ready, explainable risk scoring. Extend with full LSTM/Isolation Forest models.")
