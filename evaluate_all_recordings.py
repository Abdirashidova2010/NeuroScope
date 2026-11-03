from pathlib import Path
import mne
import numpy as np
import pandas as pd
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report

DATA_DIR = Path(__file__).parent
RECORDINGS = ["S001R04.edf", "S001R08.edf", "S001R12.edf"]
EVENT_ID = {"T1": 1, "T2": 2}

def load_recording(filename, expected_channels=None):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {filename}. Put all three EDF files beside this script.")
    print(f"\\nLoading {filename}...")
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    descriptions = set(raw.annotations.description)
    if not all(event in descriptions for event in EVENT_ID):
        raise ValueError(f"{filename} is missing T1 or T2 annotations. Found: {sorted(descriptions)}")
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose="ERROR")
    raw.pick("eeg")
    if expected_channels is not None:
        missing = set(expected_channels) - set(raw.ch_names)
        if missing:
            raise ValueError(f"{filename} is missing EEG channels: {missing}")
        raw.reorder_channels(expected_channels)
    raw.filter(8.0, 30.0, verbose="ERROR")
    epochs = mne.Epochs(raw, events, event_id=EVENT_ID, tmin=1.0, tmax=3.5,
                        baseline=None, preload=True, reject_by_annotation=True, verbose="ERROR")
    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1]
    print(f"Extracted trials: {len(y)}")
    print(f"Class counts: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, epochs.ch_names

def main():
    recordings = {}
    common_channels = None
    for filename in RECORDINGS:
        X, y, names = load_recording(filename, expected_channels=common_channels)
        if common_channels is None:
            common_channels = names
        recordings[filename] = (X, y)

    rows = []
    print("\\n========== LEAVE-ONE-RECORDING-OUT EVALUATION ==========")
    for test_filename in RECORDINGS:
        train_files = [name for name in RECORDINGS if name != test_filename]
        X_train = np.concatenate([recordings[name][0] for name in train_files], axis=0)
        y_train = np.concatenate([recordings[name][1] for name in train_files])
        X_test, y_test = recordings[test_filename]
        if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
            raise ValueError(f"Both T1 and T2 must be present for fold {test_filename}.")

        model = Pipeline([("csp", CSP(n_components=4, log=True)),
                          ("lda", LinearDiscriminantAnalysis())])
        print(f"\\n--- Held-out recording: {test_filename} ---")
        print("Training recordings:", ", ".join(train_files))
        print(f"Training trials: {len(y_train)} | Test trials: {len(y_test)}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        balanced = balanced_accuracy_score(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions, labels=[1, 2])
        report = classification_report(
            y_test, predictions, labels=[1, 2],
            target_names=["Left fist (T1)", "Right fist (T2)"],
            output_dict=True, zero_division=0
        )
        print(f"Accuracy: {accuracy * 100:.1f}%")
        print(f"Balanced accuracy: {balanced * 100:.1f}%")
        print("Confusion matrix (rows=actual, columns=predicted; order T1,T2):")
        print(matrix)
        print(classification_report(
            y_test, predictions, labels=[1, 2],
            target_names=["Left fist (T1)", "Right fist (T2)"], zero_division=0
        ))

        rows.append({
            "held_out_recording": test_filename,
            "training_recordings": "; ".join(train_files),
            "training_trials": len(y_train),
            "testing_trials": len(y_test),
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "T1_precision": report["Left fist (T1)"]["precision"],
            "T1_recall": report["Left fist (T1)"]["recall"],
            "T1_f1": report["Left fist (T1)"]["f1-score"],
            "T2_precision": report["Right fist (T2)"]["precision"],
            "T2_recall": report["Right fist (T2)"]["recall"],
            "T2_f1": report["Right fist (T2)"]["f1-score"],
            "T1_correct": int(matrix[0, 0]),
            "T1_as_T2": int(matrix[0, 1]),
            "T2_as_T1": int(matrix[1, 0]),
            "T2_correct": int(matrix[1, 1]),
        })

    results = pd.DataFrame(rows)
    output = DATA_DIR / "leave_one_recording_out_results.csv"
    results.to_csv(output, index=False)
    print("\\n========== SUMMARY ==========")
    print(results[["held_out_recording", "accuracy", "balanced_accuracy"]].to_string(index=False))
    print(f"\\nMean accuracy: {results['accuracy'].mean() * 100:.1f}%")
    print(f"Mean balanced accuracy: {results['balanced_accuracy'].mean() * 100:.1f}%")
    print(f"\\nDetailed results saved to: {output}")
    print("\\nThis tests held-out recordings from the same participant, not new people.")

if __name__ == "__main__":
    main()
