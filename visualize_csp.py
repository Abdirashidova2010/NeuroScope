
from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt

from mne.decoding import CSP

DATA_DIR = Path(__file__).parent
TRAIN_FILES = ["S001R04.edf", "S001R08.edf"]
EVENT_ID = {"T1": 1, "T2": 2}

all_epochs = []

# Load and preprocess the training recordings.
for filename in TRAIN_FILES:
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {filename}")

    print(f"Loading {filename}...")

    raw = mne.io.read_raw_edf(
        str(path), preload=True, verbose="ERROR"
    )

    events, _ = mne.events_from_annotations(
        raw, event_id=EVENT_ID, verbose="ERROR"
    )

    raw.pick("eeg")

# Remove trailing dots from EEG channel names.
raw.rename_channels(
    lambda name: name.rstrip(".")
)

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

all_epochs.append(epochs)

# Combine training trials.
combined = mne.concatenate_epochs(
    all_epochs, add_offset=True, verbose="ERROR"
)

X = combined.get_data(copy=True)
y = combined.events[:, -1]

print(f"\nTotal training trials: {len(y)}")
print(f"EEG channels: {len(combined.ch_names)}")

# Fit CSP using training data only.
csp = CSP(
    n_components=4,
    reg=None,
    log=True,
    transform_into="average_power",
)

csp.fit(X, y)

# Extract CSP features from the training trials.
features = csp.transform(X)

# Compare left-fist and right-fist imagery.
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for index, ax in enumerate(axes):
    left_features = features[y == 1, index]
    right_features = features[y == 2, index]

    ax.boxplot(
        [left_features, right_features],
        tick_labels=["Left fist", "Right fist"],
    )

    ax.set_title(f"CSP Component {index + 1}")
    ax.set_ylabel("Log-variance feature")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "NeuroScope: CSP Features by Motor Imagery Class",
    fontsize=15,
)

plt.tight_layout()

feature_path = DATA_DIR / "csp_class_features.png"
fig.savefig(feature_path, dpi=200, bbox_inches="tight")

print(f"Class-wise feature plot saved to: {feature_path}")

plt.show()

print("\nCSP fitted successfully.")
print("CSP component patterns shape:", csp.patterns_.shape)

# Visualize CSP spatial-pattern weights by EEG channel.
# This does not require assumed electrode coordinates.

fig, axes = plt.subplots(
    4, 1,
    figsize=(14, 16),
    constrained_layout=True
)
channel_names = combined.ch_names

for index, ax in enumerate(axes):
    weights = csp.patterns_[index]

    # Show the 15 channels with the largest absolute weights.
    strongest = np.argsort(np.abs(weights))[-15:]
    strongest = strongest[np.argsort(np.abs(weights[strongest]))]

    ax.barh(
        [channel_names[i] for i in strongest],
        weights[strongest],
    )

    ax.set_title(f"CSP Spatial Pattern {index + 1}")
    ax.set_xlabel("Spatial pattern weight")
    ax.set_ylabel("EEG channel")

fig.suptitle(
    "NeuroScope: CSP Spatial Pattern Analysis",
    fontsize=17,
)

output_path = DATA_DIR / "csp_patterns.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")

print(f"\nVisualization saved to: {output_path}")

plt.show()
