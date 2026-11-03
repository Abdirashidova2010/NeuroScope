
from pathlib import Path

import mne
import numpy as np

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score

DATA_DIR = Path(__file__).parent

TRAIN_FILES = ["S001R04.edf", "S001R08.edf"]
TEST_FILE = "S001R12.edf"

EVENT_ID = {"T1": 1, "T2": 2}


def load_recording(filename, channel_names=None):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {filename}")

    print(f"\nLoading {filename}...")
    raw = mne.io.read_raw_edf(
        str(path), preload=True, verbose="ERROR"
    )

    events, _ = mne.events_from_annotations(
        raw, event_id=EVENT_ID, verbose="ERROR"
    )

    raw.pick("eeg")

    if channel_names is not None:
        if set(raw.ch_names) != set(channel_names):
            raise ValueError("EEG channels do not match.")
        raw.reorder_channels(channel_names)

    raw.filter(8.0, 30.0, verbose="ERROR")

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

    print(f"Extracted trials: {len(y)}")
    return X, y, epochs.ch_names


# Load training recordings.
X_parts = []
y_parts = []
channel_names = None

for filename in TRAIN_FILES:
    X, y, names = load_recording(filename, channel_names)

    if channel_names is None:
        channel_names = names

    X_parts.append(X)
    y_parts.append(y)

X_train = np.concatenate(X_parts, axis=0)
y_train = np.concatenate(y_parts)

# Load the unseen test recording.
X_test, y_test, _ = load_recording(TEST_FILE, channel_names)

# Baseline: always predict the most frequent training class.
classes, counts = np.unique(y_train, return_counts=True)
majority_class = classes[np.argmax(counts)]

baseline_predictions = np.full(
    len(y_test), majority_class
)

baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)
baseline_balanced = balanced_accuracy_score(
    y_test, baseline_predictions
)

# CSP + LDA model.
model = Pipeline([
    ("csp", CSP(n_components=4, log=True)),
    ("lda", LinearDiscriminantAnalysis()),
])

print("\nTraining CSP + LDA...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

model_accuracy = accuracy_score(y_test, predictions)
model_balanced = balanced_accuracy_score(y_test, predictions)

# Compare both approaches.
print("\n--- BASELINE VS CSP + LDA ---")

print("\nMajority-class baseline:")
print(f"Accuracy: {baseline_accuracy * 100:.1f}%")
print(f"Balanced accuracy: {baseline_balanced * 100:.1f}%")

print("\nCSP + LDA:")
print(f"Accuracy: {model_accuracy * 100:.1f}%")
print(f"Balanced accuracy: {model_balanced * 100:.1f}%")

print("\n--- COMPARISON ---")
print(
    f"Accuracy difference: "
    f"{(model_accuracy - baseline_accuracy) * 100:+.1f} percentage points"
)
print(
    f"Balanced accuracy difference: "
    f"{(model_balanced - baseline_balanced) * 100:+.1f} percentage points"
)

print("\nExperiment complete.")
