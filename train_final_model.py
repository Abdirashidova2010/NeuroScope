import joblib
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from pathlib import Path
import mne
import numpy as np
import joblib

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

DATA_DIR = Path(__file__).parent

TRAIN_FILES = [
    "S001R04.edf",
    "S001R08.edf",
    "S001R12.edf",
]

EVENT_ID = {"T1": 1, "T2": 2}

X_parts = []
y_parts = []
channel_names = None

for filename in TRAIN_FILES:

    print(f"\nLoading {filename}...")

    raw = mne.io.read_raw_edf(
        str(DATA_DIR / filename),
        preload=True,
        verbose="ERROR"
    )

    events, _ = mne.events_from_annotations(
        raw,
        event_id=EVENT_ID,
        verbose="ERROR"
    )

    raw.pick("eeg")
    raw.filter(8.0, 30.0, verbose="ERROR")

    if channel_names is None:
        channel_names = raw.ch_names.copy()
    else:
        missing = set(channel_names) - set(raw.ch_names)

        if missing:
            raise ValueError(f"Missing channels: {missing}")

        raw.reorder_channels(channel_names)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_ID,
        tmin=1.0,
        tmax=3.5,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1]

    X_parts.append(X)
    y_parts.append(y)

    print(f"Extracted trials: {len(y)}")

X_train = np.concatenate(X_parts, axis=0)
y_train = np.concatenate(y_parts)

print("\nFinal dataset:")
print("Trials:", len(y_train))
print("EEG shape:", X_train.shape)
print("Class counts:", np.unique(y_train, return_counts=True))

model = Pipeline([
    ("csp", CSP(n_components=4, log=True)),
    ("lda", LinearDiscriminantAnalysis()),
])

print("\nTraining final model...")
model.fit(X_train, y_train)

output_path = DATA_DIR / "neuroscope_final_model.joblib"

joblib.dump({
    "model": model,
    "channel_names": channel_names,
    "event_id": EVENT_ID,
    "sampling_frequency": 160,
    "filter_band": [8.0, 30.0],
    "epoch_window": [1.0, 3.5],
}, output_path)

print("\nFinal model saved:", output_path)
print("Training completed successfully!")