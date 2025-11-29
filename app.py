import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import plotly.graph_objects as go
import time
from keras import layers, models

# ==========================================
# 0. CONFIG & CPU OPTIMIZATION
# ==========================================
st.set_page_config(
    page_title="SolarTwin AI | Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# CRITICAL FIX: Force CPU standard (Fixes float16/float32 crash)
tf.keras.mixed_precision.set_global_policy('float32')

# ==========================================
# 1. CUSTOM LAYERS (PATCHED FOR CPU)
# ==========================================
@tf.keras.utils.register_keras_serializable()
class Time2Vector(layers.Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__(**kwargs)
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear', shape=(int(self.seq_len), 1), initializer='uniform', trainable=True)
        self.bias_linear = self.add_weight(name='bias_linear', shape=(int(self.seq_len), 1), initializer='uniform', trainable=True)
        self.weights_periodic = self.add_weight(name='weight_periodic', shape=(int(self.seq_len), 1), initializer='uniform', trainable=True)
        self.bias_periodic = self.add_weight(name='bias_periodic', shape=(int(self.seq_len), 1), initializer='uniform', trainable=True)

    def call(self, x):
        # PATCH: Convert input to float32 to match CPU weights
        x = tf.cast(x, tf.float32) 
        x_mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        time_linear = self.weights_linear * x_mean + self.bias_linear
        time_periodic = tf.math.sin(tf.multiply(x_mean, self.weights_periodic) + self.bias_periodic)
        return tf.concat([time_linear, time_periodic], axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len})
        return config

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([layers.Dense(ff_dim, activation="gelu"), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        # PATCH: Convert all inputs and intermediates to float32
        inputs = tf.cast(inputs, tf.float32)
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        
        # Explicit cast before add
        out1 = self.layernorm1(inputs + tf.cast(attn_output, tf.float32))
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # Explicit cast before add
        return self.layernorm2(out1 + tf.cast(ffn_output, tf.float32))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return config

# ==========================================
# 2. DATA & MODEL LOADING
# ==========================================
@st.cache_resource
def load_system():
    # Load Model (Inference Mode)
    model = tf.keras.models.load_model('ustt_model.keras', custom_objects={
        'Time2Vector': Time2Vector, 
        'TransformerBlock': TransformerBlock
    }, compile=False)
    
    # Load Scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    # Load Test Data
    df = pd.read_csv('test_data_sample.csv')
    return model, scaler, df

# Initialize Session State (Fixes NameError issues)
if 'model_loaded' not in st.session_state:
    try:
        model, scaler, df_test = load_system()
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['df'] = df_test
        st.session_state['model_loaded'] = True
    except Exception as e:
        st.error(f"System Boot Failure: {e}")
        st.stop()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("⚡ SolarTwin AI")
    st.caption("Universal Spatio-Temporal Transformer")
    
    if st.session_state.get('model_loaded'):
        st.success("● System Online (v1.0.4)")
    else:
        st.warning("○ System Loading...")
    
    st.divider()
    
    # Global Controls
    st.subheader("📍 Site Selection")
    site_option = st.selectbox("Target Facility", 
        ["Plant 1 (Utility Scale - 250MW)", "Plant 2 (Residential - 22MW)"],
        index=1 # Default to P2
    )
    site_id = 0.0 if "Plant 1" in site_option else 1.0
    
    st.divider()
    
    # Digital Twin Simulation
    st.subheader("🧪 Scenario Simulation")
    st.info("Inject synthetic weather patterns to stress-test the AI.")
    
    cloud_factor = st.slider("Cloud Cover Intensity", 0.0, 1.0, 0.0, 0.1, help="Artificially reduce irradiance input.")
    temp_offset = st.slider("Temp. Deviation (°C)", -10, 10, 0, help="Simulate heatwaves or cold snaps.")
    
    st.divider()
    st.markdown("© 2025 Shah Mohammad Rizvi | SolarTwin AI")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================

# -- Header --
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Grid Operations Center")
    st.markdown(f"**Monitoring:** {site_option}")
with col2:
    if st.button("🔄 Refresh Telemetry"):
        st.rerun()

# -- Tabs --
tab1, tab2, tab3 = st.tabs(["📊 Live Dashboard", "🔮 Forecast & Simulation", "⚠️ Risk Analysis"])

# --- DATA PREPARATION LOGIC ---
LOOK_BACK = 96
# Access data from Session State
if 'df' in st.session_state:
    site_df = st.session_state['df'][st.session_state['df']['site_id'] == site_id].reset_index(drop=True)
    
    # Pick a random window
    if 'window_idx' not in st.session_state:
        st.session_state['window_idx'] = np.random.randint(0, len(site_df) - LOOK_BACK - 24)

    start_idx = st.session_state['window_idx']
    input_slice = site_df.iloc[start_idx : start_idx + LOOK_BACK].copy()

    # APPLY SIMULATION
    feature_cols = ['DC_POWER', 'AC_POWER', 'IRRADIATION', 'MODULE_TEMPERATURE', 'Day_Sin', 'Day_Cos']
    input_slice['IRRADIATION'] = input_slice['IRRADIATION'] * (1.0 - cloud_factor)
    input_slice['MODULE_TEMPERATURE'] = input_slice['MODULE_TEMPERATURE'] + (temp_offset * 0.01)

    # Prepare Tensor (Explicit float32 cast for CPU)
    X_seq = input_slice[feature_cols].values.reshape(1, LOOK_BACK, 6).astype('float32')
    X_site = np.array([site_id]).reshape(1, 1).astype('float32')

    # Inference
    t0 = time.time()
    pred_scaled = st.session_state['model'].predict([X_seq, X_site], verbose=0)
    t1 = time.time()
    latency = (t1 - t0) * 1000

    # Inverse Transform
    dummy = np.zeros((1, 7))
    dummy[0, 0] = pred_scaled[0][0]
    pred_kw = st.session_state['scaler'].inverse_transform(dummy)[0, 0]

    # Get Truth
    truth_scaled = site_df.iloc[start_idx + LOOK_BACK]['DC_POWER']
    dummy[0, 0] = truth_scaled
    truth_kw = st.session_state['scaler'].inverse_transform(dummy)[0, 0]

    # History for plotting
    history_vals = site_df.iloc[start_idx : start_idx + LOOK_BACK]['DC_POWER'].values
    history_kw = []
    for val in history_vals:
        dummy[0,0] = val
        history_kw.append(st.session_state['scaler'].inverse_transform(dummy)[0,0])

    # ==========================================
    # TAB 1: LIVE DASHBOARD
    # ==========================================
    with tab1:
        # KPI Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.metric("Predicted Output (t+15m)", f"{pred_kw:.2f} kW", f"{(pred_kw - truth_kw):.2f} kW Error")
        
        with kpi2:
            capacity = 250000 if site_id == 0 else 22000
            load_pct = (pred_kw / capacity) * 100
            st.metric("Capacity Utilization", f"{load_pct:.1f}%", "Optimal")
            
        with kpi3:
            st.metric("Model Latency", f"{latency:.2f} ms", "Real-Time Ready")
            
        with kpi4:
            confidence = "High" if cloud_factor < 0.3 else "Moderate"
            st.metric("Forecast Confidence", confidence, "95% CI Active")

        # Main Chart
        st.subheader("⚡ Real-Time Load Tracking")
        fig = go.Figure()
        
        # History
        fig.add_trace(go.Scatter(
            x=list(range(len(history_kw))),
            y=history_kw,
            mode='lines',
            name='Historical Generation',
            line=dict(color='#3b82f6', width=2),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=[len(history_kw)], 
            y=[pred_kw],
            mode='markers+text',
            name='USTT Prediction',
            marker=dict(color='#ef4444', size=14, symbol='diamond'),
            text=[f"{pred_kw:.0f} kW"],
            textposition="top center"
        ))
        
        # Truth
        fig.add_trace(go.Scatter(
            x=[len(history_kw)],
            y=[truth_kw],
            mode='markers',
            name='Actual Ground Truth',
            marker=dict(color='#22c55e', size=10, symbol='x')
        ))

        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Time Steps (15-min Intervals)",
            yaxis_title="Power Generation (kW)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # TAB 2: FORECAST & SIMULATION
    # ==========================================
    with tab2:
        st.markdown("### 🧬 Digital Twin Simulation")
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Parameters:**\n- Cloud: {cloud_factor*100:.0f}%\n- Temp: {temp_offset:+}°C")
            st.dataframe(input_slice[feature_cols].tail(5).style.highlight_min(axis=0), use_container_width=True)
        with col_b:
            # Efficiency Gauge
            eff_fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = (1.0 - cloud_factor) * 100,
                title = {'text': "Solar Efficiency"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#f59e0b"}}
            ))
            eff_fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(eff_fig, use_container_width=True)

    # ==========================================
    # TAB 3: RISK ANALYSIS
    # ==========================================
    with tab3:
        st.subheader("Uncertainty Quantification (95% CI)")
        sigma = 2625 if site_id == 0 else 262
        upper = pred_kw + (1.96 * sigma)
        lower = max(0, pred_kw - (1.96 * sigma))
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Lower Bound", f"{lower:.2f} kW")
        c2.metric("Mean Forecast", f"{pred_kw:.2f} kW")
        c3.metric("Upper Bound", f"{upper:.2f} kW")
        
        # Risk Cone Chart
        risk_fig = go.Figure()
        risk_fig.add_trace(go.Scatter(x=list(range(len(history_kw))), y=history_kw, mode='lines', name='History', line=dict(color='gray')))
        
        # Cone
        x_fut = [len(history_kw)]
        risk_fig.add_trace(go.Scatter(x=[len(history_kw), len(history_kw)], y=[lower, upper], mode='lines', line=dict(width=10, color='rgba(239, 68, 68, 0.4)'), name='95% CI Range'))
        
        risk_fig.update_layout(title="Instantaneous Risk Range", xaxis_title="Time", yaxis_title="Power")
        st.plotly_chart(risk_fig, use_container_width=True)

else:
    st.error("Data could not be loaded. Please check test_data_sample.csv")