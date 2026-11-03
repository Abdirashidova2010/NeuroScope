import tempfile
from pathlib import Path
import joblib
import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from scipy.signal import welch

st.set_page_config(
    page_title="NeuroScope | Real EEG",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NeuroScope — Real EEG Viewer")
st.caption("Explore EEG recordings using MNE-Python")

st.warning(
    "Research demonstration only. This tool does not provide medical diagnoses."
)


# --- AI Model Evaluation Dashboard ---
st.header("📊 AI Model Evaluation")
st.caption(
    "Leave-one-recording-out evaluation. Each recording is tested using a "
    "model trained on the other two recordings from the same participant."
)

results_path = Path(__file__).parent / "leave_one_recording_out_results.csv"

if results_path.exists():
    results_df = pd.read_csv(results_path)

    if not results_df.empty:
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Mean accuracy", f"{results_df['accuracy'].mean() * 100:.1f}%")
        metric_col2.metric(
            "Mean balanced accuracy",
            f"{results_df['balanced_accuracy'].mean() * 100:.1f}%"
        )

        st.subheader("Performance across held-out recordings")
        performance_fig = go.Figure()
        performance_fig.add_trace(go.Bar(
            x=results_df["held_out_recording"],
            y=results_df["accuracy"] * 100,
            name="Accuracy",
        ))
        performance_fig.add_trace(go.Bar(
            x=results_df["held_out_recording"],
            y=results_df["balanced_accuracy"] * 100,
            name="Balanced accuracy",
        ))
        performance_fig.update_layout(
            barmode="group",
            template="plotly_dark",
            yaxis_title="Score (%)",
            xaxis_title="Held-out recording",
            yaxis=dict(range=[0, 100]),
            height=380,
        )
        st.plotly_chart(performance_fig, width="stretch")

        st.subheader("Detailed classification metrics")
        metric_columns = [
            "held_out_recording", "T1_precision", "T1_recall", "T1_f1",
            "T2_precision", "T2_recall", "T2_f1",
        ]
        st.dataframe(
            results_df[[c for c in metric_columns if c in results_df.columns]].round(3),
            hide_index=True,
            width="stretch",
        )

        confusion_columns = ["T1_correct", "T1_as_T2", "T2_as_T1", "T2_correct"]
        if all(c in results_df.columns for c in confusion_columns):
            st.subheader("Confusion matrix by recording")
            selected = st.selectbox(
                "Choose a held-out recording",
                results_df["held_out_recording"].tolist(),
                key="evaluation_recording",
            )
            row = results_df[results_df["held_out_recording"] == selected].iloc[0]
            matrix = [
                [int(row["T1_correct"]), int(row["T1_as_T2"])],
                [int(row["T2_as_T1"]), int(row["T2_correct"])],
            ]
            matrix_fig = go.Figure(data=go.Heatmap(
                z=matrix,
                x=["Predicted T1 (left)", "Predicted T2 (right)"],
                y=["Actual T1 (left)", "Actual T2 (right)"],
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues",
            ))
            matrix_fig.update_layout(
                template="plotly_dark",
                height=350,
                xaxis_title="Predicted class",
                yaxis_title="Actual class",
            )
            st.plotly_chart(matrix_fig, width="stretch")

        st.download_button(
            "Download evaluation results (CSV)",
            data=results_path.read_bytes(),
            file_name="leave_one_recording_out_results.csv",
            mime="text/csv",
        )
        st.info(
            "These results evaluate held-out recordings from one participant. "
            "They do not demonstrate generalization to new participants or "
            "clinical diagnostic performance."
        )
    else:
        st.warning("The evaluation results CSV is empty. Run the evaluation script first.")
else:
    st.info(
        "Evaluation results have not been found yet. Run evaluate_all_recordings.py "
        "and keep its CSV output in the same folder as eeg_loader.py."
    )

st.divider()

st.write(
    "Upload an EEG recording in EDF or FIF format to inspect its channels "
    "and visualize the signal."
)

uploaded_file = st.file_uploader(
    "Upload an EEG recording",
    type=["edf", "fif"],
)

if uploaded_file is None:
    st.info("Please upload an EEG recording to begin.")
    st.stop()

# Load the uploaded EEG recording.
try:
    uploaded_file.seek(0)

    if uploaded_file.name.lower().endswith(".edf"):
        suffix = ".edf"
    else:
        suffix = ".fif"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    if suffix == ".edf":
        raw = mne.io.read_raw_edf(temp_path, preload=True, verbose="ERROR")
    else:
        raw = mne.io.read_raw_fif(temp_path, preload=True, verbose="ERROR")

except Exception as error:
    st.error(f"Could not read this recording: {error}")
    st.info("Please upload a valid EDF or FIF recording.")
    st.stop()

st.success("Recording loaded successfully!")

# --- NeuroScope AI Classification ---
st.header("🧠 Motor Imagery Classification")

try:
    saved_model = joblib.load("neuroscope_final_model.joblib")
    model = saved_model["model"]
    channel_names = saved_model["channel_names"]

    raw_ai = raw.copy()
    if float(raw_ai.info["sfreq"]) != float(saved_model["sampling_frequency"]):
        raise ValueError(
            f"AI classification requires {saved_model['sampling_frequency']} Hz. "
            f"This recording uses {raw_ai.info['sfreq']} Hz."
        )

    missing_channels = set(channel_names) - set(raw_ai.ch_names)

    if missing_channels:
        raise ValueError(
            f"Missing required EEG channels: {sorted(missing_channels)}"
        )
    else:
        raw_ai.pick("eeg")
        raw_ai.reorder_channels(channel_names)
        raw_ai.filter(8.0, 30.0, verbose="ERROR")

        events, _ = mne.events_from_annotations(
            raw_ai,
            event_id={"T1": 1, "T2": 2},
            verbose="ERROR",
        )

        epochs = mne.Epochs(
            raw_ai,
            events,
            event_id={"T1": 1, "T2": 2},
            tmin=1.0,
            tmax=3.5,
            baseline=None,
            preload=True,
            reject_by_annotation=True,
            verbose="ERROR",
        )

        if len(epochs) > 0:
            X_new = epochs.get_data(copy=True)
            predictions = model.predict(X_new)

            left_count = int(np.sum(predictions == 1))
            right_count = int(np.sum(predictions == 2))

            col1, col2 = st.columns(2)
            col1.metric("Left-hand imagery", left_count)
            col2.metric("Right-hand imagery", right_count)
            st.write(f"Total classified trials: {len(predictions)}")
            st.caption(
                "Experimental EEG classification. Predictions are not medical diagnoses."
            )
        else:
            st.info("No valid T1/T2 motor imagery trials found.")

except Exception as error:
    st.warning(f"Classification unavailable: {error}")

# --- EEG Signal Viewer ---
# This section is deliberately OUTSIDE the classification try/except block.
sampling_rate = float(raw.info["sfreq"])
duration = float(raw.times[-1]) if len(raw.times) else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Sampling rate", f"{sampling_rate:g} Hz")
col2.metric("Channels", len(raw.ch_names))
col3.metric("Duration", f"{duration:.1f} seconds")

selected_channels = st.multiselect(
    "Choose EEG channels to display",
    raw.ch_names,
    default=raw.ch_names[:5],
)

max_window = max(1, min(30, int(duration)))
window = st.slider(
    "Window length (seconds)",
    min_value=1,
    max_value=max_window,
    value=min(5, max_window),
)

max_start = max(0, int(duration) - window)

if "eeg_playing" not in st.session_state:
    st.session_state.eeg_playing = False

if "eeg_start_time_slider" not in st.session_state:
    st.session_state.eeg_start_time_slider = 0

st.session_state.eeg_start_time_slider = min(
    st.session_state.eeg_start_time_slider,
    max_start
)

play_col, pause_col, reset_col = st.columns(3)

with play_col:
    if st.button("▶ Play", use_container_width=True):
        st.session_state.eeg_playing = True

with pause_col:
    if st.button("⏸ Pause", use_container_width=True):
        st.session_state.eeg_playing = False

with reset_col:
    if st.button("↺ Reset", use_container_width=True):
        st.session_state.eeg_playing = False
        st.session_state.eeg_start_time_slider = 0

if st.session_state.eeg_playing and max_start > 0:
    st_autorefresh(interval=1000, key="eeg_playback_refresh")

    st.session_state.eeg_start_time_slider = min(
        st.session_state.eeg_start_time_slider + 1,
        max_start
    )

    if st.session_state.eeg_start_time_slider >= max_start:
        st.session_state.eeg_playing = False

start = st.slider(
    "Start time (seconds)",
    min_value=0,
    max_value=max_start,
    step=1,
    key="eeg_start_time_slider",
)

start_sample = int(start * sampling_rate)
stop_sample = int((start + window) * sampling_rate)

if not selected_channels:
    st.warning("Please select at least one EEG channel.")
    st.stop()

signal, times = raw.get_data(
    picks=selected_channels,
    start=start_sample,
    stop=stop_sample,
    return_times=True,
)

# Apply a 1–40 Hz band-pass filter to a copy of the recording.
raw_filtered = raw.copy().filter(
    l_freq=1.0,
    h_freq=40.0,
    picks=selected_channels,
    method="iir",
    iir_params={
        "order": 4,
        "ftype": "butter",
    },
    verbose="ERROR",
)

filtered_signals = raw_filtered.get_data(
    picks=selected_channels,
    start=start_sample,
    stop=stop_sample,
) * 1e6

raw_signals = signal * 1e6

# Use the first selected channel for the frequency analysis below.
signal_uv = filtered_signals[0]
signal_raw_uv = raw_signals[0]

signal_raw_uv = signal[0] * 1e6

fig = go.Figure()

spacing = max(
    1.0,
    np.max(np.abs(raw_signals)) * 2
)

for i, channel_name in enumerate(selected_channels):
    offset = i * spacing

    fig.add_trace(
        go.Scatter(
            x=times,
            y=raw_signals[i] + offset,
            mode="lines",
            name=f"{channel_name} - Raw",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=filtered_signals[i] + offset,
            mode="lines",
            name=f"{channel_name} - Filtered",
        )
    )

fig.update_layout(
    template="plotly_dark",
    title="Multichannel EEG Signals",
    xaxis_title="Time (seconds)",
    yaxis_title="Amplitude + channel offset (µV)",
    height=550,
)

st.plotly_chart(fig, width="stretch")

# --- EEG Signal Quality Indicators ---
st.subheader("EEG Signal Quality Indicators")
signal_for_quality = np.asarray(signal_uv, dtype=float)

if signal_for_quality.size > 0 and np.all(np.isfinite(signal_for_quality)):
    amplitude_range = float(np.max(signal_for_quality) - np.min(signal_for_quality))
    signal_std = float(np.std(signal_for_quality))
    threshold = 100.0
    large_value_percent = float(
        np.mean(np.abs(signal_for_quality) > threshold) * 100
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Amplitude range", f"{amplitude_range:.2f} µV")
    col2.metric("Standard deviation", f"{signal_std:.2f} µV")
    col3.metric("Values above 100 µV", f"{large_value_percent:.2f}%")

    st.caption(
        "These are descriptive signal metrics. Large amplitudes can reflect artifacts, "
        "but thresholds depend on the recording setup and do not establish signal "
        "quality by themselves."
    )
else:
    st.warning("No finite signal values are available for quality analysis.")

# --- EEG Frequency Band Analysis ---
st.subheader("EEG Frequency Band Analysis")

signal_centered = signal_uv - np.mean(signal_uv)

frequencies, psd = welch(
    signal_centered,
    fs=sampling_rate,
    nperseg=min(512, len(signal_centered)),
    noverlap=min(256, len(signal_centered) // 2)
)

fft_values = psd

bands = {
    "Delta (1–4 Hz)": (1, 4),
    "Theta (4–8 Hz)": (4, 8),
    "Alpha (8–13 Hz)": (8, 13),
    "Beta (13–30 Hz)": (13, 30),
    "Gamma (30–40 Hz)": (30, 40),
}

band_powers = {}
for band_name, (low, high) in bands.items():
    mask = (frequencies >= low) & (frequencies < high)
    band_powers[band_name] = float(fft_values[mask].sum())

total_power = sum(band_powers.values())

if total_power > 0:
    relative_power = {
        band: (power / total_power) * 100
        for band, power in band_powers.items()
    }

    st.subheader("Relative EEG Band Power (%)")

    st.bar_chart(relative_power)

    st.dataframe({
        "Frequency band": list(relative_power.keys()),
        "Relative power (%)": [
            round(value, 2)
            for value in relative_power.values()
        ]
    }, hide_index=True, width="stretch")

    st.subheader("🧠 EEG Band Power Interpretation")

    dominant_band = max(relative_power, key=relative_power.get)
    dominant_percentage = relative_power[dominant_band]

    st.metric(
        "Dominant frequency band",
        dominant_band,
        f"{dominant_percentage:.2f}% of total band power"
    )

    st.info(
        "The dominant band represents the largest relative power "
        "among the five frequency bands in this selected EEG segment. "
        "It does not independently establish a person's mental state "
        "or indicate a medical condition."
    )

else:
    st.warning("No measurable power in the selected frequency bands.")