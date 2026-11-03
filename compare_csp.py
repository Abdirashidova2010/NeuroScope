
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

ALL_FILES = [
    "S001R04.edf",
    "S001R08.edf",
    "S001R12.edf",
]

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


# 1. Run leave-one-recording-out experiments.

all_results = []

for test_file in ALL_FILES:

    print("\n" + "=" * 55)
    print(f"HELD-OUT TEST RECORDING: {test_file}")
    print("=" * 55)

    train_files = [f for f in ALL_FILES if f != test_file]

    X_train_parts = []
    y_train_parts = []
    channel_names = None

    # Load the two training recordings.
    for filename in train_files:
        X, y, names = load_recording(
            filename,
            expected_channels=channel_names
        )

        if channel_names is None:
            channel_names = names

        X_train_parts.append(X)
        y_train_parts.append(y)

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts)

    # Load the completely unseen test recording.
    X_test, y_test, _ = load_recording(
        test_file,
        expected_channels=channel_names
    )

    print(f"Training trials: {len(y_train)}")
    print(f"Testing trials: {len(y_test)}")

    # Compare 4, 6, and 8 CSP components.
    for n_components in [4, 6, 8]:

        print(f"\nTesting {n_components} CSP components...")

        model = Pipeline([
            ("csp", CSP(
                n_components=n_components,
                log=True
            )),
            ("lda", LinearDiscriminantAnalysis()),
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(
            y_test, predictions
        )

        result = {
            "Test_Recording": test_file,
            "CSP_components": n_components,
            "Accuracy": accuracy * 100,
            "Balanced_Accuracy": balanced_accuracy * 100,
        }

        all_results.append(result)

        print(f"Accuracy: {accuracy * 100:.1f}%")
        print(
            f"Balanced accuracy: "
            f"{balanced_accuracy * 100:.1f}%"
        )


# 2. Summarize results across all three recordings.

print("\n" + "=" * 55)
print("FINAL CSP COMPONENT COMPARISON")
print("=" * 55)

for n_components in [4, 6, 8]:

    component_results = [
        r for r in all_results
        if r["CSP_components"] == n_components
    ]

    mean_accuracy = np.mean([
        r["Accuracy"] for r in component_results
    ])

    mean_balanced = np.mean([
        r["Balanced_Accuracy"] for r in component_results
    ])

    print(f"\nCSP components: {n_components}")
    print(f"Mean accuracy: {mean_accuracy:.1f}%")
    print(f"Mean balanced accuracy: {mean_balanced:.1f}%")


# 3. Save all results to a CSV file.

import csv

with open("csp_component_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Test_Recording",
            "CSP_components",
            "Accuracy",
            "Balanced_Accuracy",
        ]
    )

    writer.writeheader()
    writer.writerows(all_results)

print("\nResults saved to csp_component_results.csv")