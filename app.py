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

st.set_page_config(page_title="SmartSense Guardian", layout="wide")
st.title("🛡️ SmartSense Guardian: Pump Sensor Intelligence")

@st.cache_data
def load_pump_data():
    """Load Pump Sensor Data - single CSV file"""
    try:
        df = pd.read_csv('pump_sensor.csv')
        st.success(f"✅ Pump sensor data loaded: {len(df):,} readings")
        return df
    except:
        # DEMO DATA for pump sensors
        st.info("🎲 Demo pump data loaded")
        n = 5000
        timestamps = pd.date_range('2020-01-01', periods=n, freq='S')
        return pd.DataFrame({
            'timestamp': timestamps,
            'sensor_1': np.random.normal(100, 10, n) + np.linspace(0, 20, n),
            'sensor_2': np.random.normal(50, 5, n),
            'sensor_3': np.random.normal(25, 3, n),
            'machine_status': np.concatenate([np.zeros(4000), np.ones(1000)]),  # 0=OK, 1=FAIL
            'my_status': ['NORMAL']*4000 + ['BROKEN']*1000
        })

df = load_pump_data()

# DASHBOARD CONTROLS
st.sidebar.header("🎛️ Pump Controls")
status_options = df['machine_status'].unique() if 'machine_status' in df.columns else [0,1]
status_sel = st.sidebar.selectbox("Machine Status", status_options)

# Find numeric sensor columns
sensor_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'machine_status' in sensor_cols:
    sensor_cols.remove('machine_status')
sensor_sel = st.sidebar.selectbox("Sensor", sensor_cols[:12])

data_sel = df[df['machine_status'] == status_sel]

# MAIN DASHBOARD
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Pump Sensor Signal")
    fig_ts = px.line(data_sel, x=data_sel.index, y=sensor_sel, 
                     title=f"Sensor {sensor_sel} (Status: {status_sel})")
    st.plotly_chart(fig_ts, use_container_width=True)

with col2:
    st.subheader("🔍 FFT Vibration Analysis")
    signal = data_sel[sensor_sel].fillna(0).values
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1)[:N//2]
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=xf, y=np.abs(yf[:N//2]), name='Vibration Spectrum'))
    fig_fft.update_layout(height=350)
    st.plotly_chart(fig_fft, use_container_width=True)

# PUMP HEALTH ANALYSIS
st.subheader("🤖 Pump Health Monitor")
scaler = StandardScaler()
scaled_signal = scaler.fit_transform(signal.reshape(-1, 1))

iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(scaled_signal)
anomaly_scores = iso_forest.decision_function(scaled_signal)

drift_prob = max(0, 1 - np.mean(anomaly_scores))
risk_score = min(100, drift_prob * 95)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Drift Rate", f"{drift_prob:.1%}")
col2.metric("🚨 Anomalies", f"{(anomaly_scores < 0).mean():.1%}")
col3.metric("⚠️ Risk Score", f"{risk_score:.0f}/100")
col4.metric("📊 Samples", f"{len(data_sel):,}")

# MACHINE STATUS BREAKDOWN
st.subheader("🏭 Pump Status Overview")
status_counts = df['machine_status'].value_counts()
fig_pie = px.pie(values=status_counts.values, names=[f"Status {int(i)}" for i in status_counts.index],
                 title="Machine Status Distribution")
st.plotly_chart(fig_pie, use_container_width=True)

# CRITICAL RECOMMENDATIONS
st.subheader("✅ Maintenance Action")
if risk_score > 85:
    st.error("🚨 **EMERGENCY SHUTDOWN** - Pump failure imminent!")
elif risk_score > 70:
    st.warning("🛠️ **SCHEDULE MAINTENANCE** - High vibration detected")
elif risk_score > 50:
    st.warning("⚠️ **INSPECT PUMP** - Abnormal patterns")
else:
    st.success("✅ **PUMP HEALTHY** - Normal operation")

# SENSOR CORRELATION MATRIX
if st.checkbox("Show Sensor Correlation Heatmap"):
    corr_cols = sensor_cols[:8]
    corr_matrix = data_sel[corr_cols].corr()
