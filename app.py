
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="NeuroScope | EEG Lab",
    page_icon="🧠",
    layout="wide",
)

# ---------- DARK THEME ----------
st.markdown("""
<style>
.stApp {
    background-color: #080b12;
    color: #e8edf7;
}
h1, h2, h3 {
    color: #e8edf7;
}
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #263248;
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("🧠 NeuroScope")
st.caption("EEG Signal Processing & Research Lab")

st.info(
    "DEMO MODE — Signals below are synthetic, "
    "not real recordings of brain activity."
)

# ---------- SIDEBAR ----------
st.sidebar.title("Signal Controls")

sampling_rate = 250

duration = st.sidebar.selectbox(
    "Time window",
    [2, 4, 6, 8, 10],
    index=1,
)

channel_count = st.sidebar.slider(
    "Number of channels",
    min_value=1,
    max_value=8,
    value=4,
)

show_noise = st.sidebar.checkbox(
    "Simulate signal noise",
    value=True,
)

# ---------- DEMO SIGNALS ----------
rng = np.random.default_rng(42)
time = np.arange(
    duration * sampling_rate
) / sampling_rate

signals = {}

for channel in range(channel_count):
    frequency = 8 + channel * 2

    signal = (
        np.sin(2 * np.pi * frequency * time)
        + 0.35 * np.sin(2 * np.pi * 20 * time)
    )

    if show_noise:
        signal += rng.normal(0, 0.2, len(time))

    signals[f"Channel {channel + 1}"] = signal

# ---------- METRICS ----------
m1, m2, m3 = st.columns(3)

m1.metric("Sampling rate", f"{sampling_rate} Hz")
m2.metric("Channels", channel_count)
m3.metric(
    "Samples per channel",
    f"{len(time):,}",
)

# ---------- CHANNEL SELECTOR ----------
selected_channels = st.multiselect(
    "Select channels to display",
    list(signals.keys()),
    default=list(signals.keys()),
)

# ---------- EEG WAVEFORMS ----------
st.subheader("Multichannel Signal Viewer")

fig = go.Figure()

for index, channel_name in enumerate(selected_channels):
    fig.add_trace(
        go.Scatter(
            x=time,
            y=signals[channel_name] + index * 3,
            mode="lines",
            name=channel_name,
            line=dict(width=1.2),
        )
    )

fig.update_layout(
    template="plotly_dark",
    height=480,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Time (seconds)",
    yaxis_title="Amplitude (offset for display)",
    legend=dict(orientation="h"),
    paper_bgcolor="#080b12",
    plot_bgcolor="#080b12",
)

st.plotly_chart(fig, use_container_width=True)

# ---------- SIGNAL STATISTICS ----------
st.subheader("Signal Statistics")

if selected_channels:
    stats = []

    for channel_name in selected_channels:
        signal = signals[channel_name]

        stats.append({
            "Channel": channel_name,
            "Mean": round(float(np.mean(signal)), 4),
            "Standard deviation": round(
                float(np.std(signal)), 4
            ),
            "Minimum": round(float(np.min(signal)), 4),
            "Maximum": round(float(np.max(signal)), 4),
        })

    st.dataframe(
        stats,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("Select at least one channel.")

st.caption(
    "NeuroScope prototype · Synthetic data only · "
    "Not a medical device."
)
