
from pathlib import Path

import mne
import numpy as np

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

DATA_DIR = Path(__file__).parent

TRAIN_FILES = ["S001R04.edf", "S001R12.edf"]
TEST_FILE = "S001R08.edf"

EVENT_ID = {"T1": 1, "T2": 2}


def load_recording(filename, expected_channels=None):
    """Load one EDF recording and extract motor-imagery trials."""

    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {filename}")

    print(f"\nLoading {filename}...")

    raw = mne.io.read_raw_edf(
        str(path), preload=True, verbose="ERROR"
    )

    descriptions = set(raw.annotations.description)
    print("Annotations:", sorted(descriptions))

    if not all(event in descriptions for event in EVENT_ID):
        raise ValueError(f"{filename} is missing T1 or T2 annotations.")

    events, _ = mne.events_from_annotations(
        raw, event_id=EVENT_ID, verbose="ERROR"
    )

    raw.pick("eeg")

    if expected_channels is not None:
        missing = set(expected_channels) - set(raw.ch_names)

        if missing:
            raise ValueError(
                f"{filename} is missing EEG channels: {missing}"
            )

        raw.reorder_channels(expected_channels)

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
    print("EEG channels:", len(epochs.ch_names))
    print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))

    return X, y, epochs.ch_names


# 1. Load training recordings.
X_train_parts = []
y_train_parts = []
channel_names = None

for filename in TRAIN_FILES:
    X, y, names = load_recording(
        filename, expected_channels=channel_names
    )

    if channel_names is None:
        channel_names = names

    X_train_parts.append(X)
    y_train_parts.append(y)

X_train = np.concatenate(X_train_parts, axis=0)
y_train = np.concatenate(y_train_parts)

# 2. Load the completely held-out test recording.
X_test, y_test, _ = load_recording(
    TEST_FILE, expected_channels=channel_names
)

if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
    raise ValueError("Both classes must be present in training and testing.")

print("\n--- Dataset summary ---")
print("Training trials:", len(y_train))
print("Testing trials:", len(y_test))
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# 3. Build and train the CSP + LDA pipeline.
model = Pipeline([
    ("csp", CSP(n_components=4, log=True)),
    ("lda", LinearDiscriminantAnalysis()),
])

print("\nTraining CSP + LDA...")
model.fit(X_train, y_train)

# 4. Evaluate only on the unseen recording.
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)

matrix = confusion_matrix(
    y_test, predictions, labels=[1, 2]
)

print("\n--- Held-out recording results ---")
print(f"Accuracy: {accuracy * 100:.1f}%")
print(f"Balanced accuracy: {balanced_accuracy * 100:.1f}%")

print("\nConfusion matrix:")
print("Rows = actual class; columns = predicted class")
print("             Predicted T1   Predicted T2")
print(f"Actual T1       {matrix[0, 0]:4d}             {matrix[0, 1]:4d}")
print(f"Actual T2       {matrix[1, 0]:4d}             {matrix[1, 1]:4d}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[1, 2],
        target_names=["Left fist (T1)", "Right fist (T2)"],
        zero_division=0,
    )
)

print(
    "\nEvaluation complete. This measures generalization to a "
    "separate run from the same participant, not to a new person."
)


import joblib

# Save the trained CSP + LDA pipeline
joblib.dump(model, "neuroscope_model.joblib")

print("Model saved successfully as neuroscope_model.joblib")
