
from pathlib import Path

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# All recordings are stored beside this script.
DATA_DIR = Path(__file__).parent
RUNS = ["S001R04.edf", "S001R08.edf", "S001R12.edf"]

all_epochs = []

for filename in RUNS:
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing {filename}. Put it in the NeuroScope folder."
        )

    print(f"\nLoading {filename}...")
    raw = mne.io.read_raw_edf(
        str(file_path), preload=True, verbose="ERROR"
    )

    descriptions = set(raw.annotations.description)
    print("Annotations:", sorted(descriptions))

    # T1 = imagined left fist; T2 = imagined right fist.
    event_id = {
        name: code
        for name, code in [("T1", 1), ("T2", 2)]
        if name in descriptions
    }

    if len(event_id) != 2:
        raise ValueError(
            f"{filename}: expected T1 and T2 annotations; "
            f"found {sorted(descriptions)}"
        )

    events, _ = mne.events_from_annotations(
        raw, event_id=event_id, verbose="ERROR"
    )

    # Keep EEG channels and filter the 8–30 Hz frequency range.
    raw.pick("eeg")
    raw.filter(8.0, 30.0, verbose="ERROR")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=1.0,
        tmax=3.5,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    print(f"Trials extracted from {filename}: {len(epochs)}")
    all_epochs.append(epochs)

# Combine the trials from all three runs.
combined = mne.concatenate_epochs(
    all_epochs, add_offset=True, verbose="ERROR"
)

X = combined.get_data(copy=True)
y = combined.events[:, -1]

labels, counts = np.unique(y, return_counts=True)

print("\n--- Combined dataset ---")
print(f"Total trials: {len(y)}")
print(f"Left-fist trials (T1): {counts[labels == 1][0]}")
print(f"Right-fist trials (T2): {counts[labels == 2][0]}")
print(f"EEG data shape: {X.shape}")

if len(labels) != 2 or min(counts) < 3:
    raise ValueError("Not enough trials from both classes.")

# CSP extracts spatial features; LDA performs classification.
model = Pipeline([
    ("csp", CSP(n_components=4, log=True)),
    ("lda", LinearDiscriminantAnalysis()),
])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print("\n--- Combined CSP + LDA results ---")
print("Fold accuracies:", np.round(scores, 3))
print(f"Mean accuracy: {scores.mean() * 100:.1f}%")
print(
    f"Variation across folds: {scores.std() * 100:.1f} percentage points"
)
print(
    "\nInterpretation: preliminary within-subject evaluation. "
    "Random trial splits can include trials from the same recording "
    "in training and testing, so this does not establish "
    "generalization to a new session or person."
)
