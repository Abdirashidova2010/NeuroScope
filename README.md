# NeuroScope

**An EEG Visualization and Motor Imagery Classification Dashboard**
NeuroScope/
├── eeg_loader.py
├── neuroscope_final_model.joblib
├── requirements.txt
├── README.md
├── TECHNICAL_VALIDATION.md
├── leave_one_recording_out_results.csv
├── evaluate_all_recordings.py
├── S001R04.edf
├── S001R08.edf
└── S001R12.edf

## Overview

NeuroScope is a Python-based application for exploring EEG recordings through interactive visualization, signal processing, and experimental machine learning.

It allows users to upload EEG recordings in EDF or FIF format, inspect EEG channels, visualize signals, explore frequency characteristics, and investigate motor imagery classification using a trained CSP-LDA model.

## Features
- Multichannel EEG visualization
- Interactive playback controls (Play, Pause, Reset)
- Raw and filtered signal comparison
- Fourth-order Butterworth IIR band-pass filtering
- Power Spectral Density estimation using Welch's method
- Frequency-band power analysis (Delta, Theta, Alpha, Beta, Gamma)
- Signal quality statistics
- EDF and FIF recording upload
- EEG channel selection
- Interactive EEG signal visualization
- Recording metadata: sampling rate, channel count, and duration
- Signal filtering and frequency analysis
- Motor imagery classification using CSP and LDA
- Model evaluation dashboard with accuracy, balanced accuracy, and confusion matrices

## Technology Stack

- Python 3.12
- Streamlit
- MNE-Python
- NumPy
- SciPy
- Pandas
- Scikit-learn
- Plotly
- Joblib

## Project Structure

```text
NeuroScope/
├── eeg_loader.py
├── neuroscope_final_model.joblib
├── requirements.txt
├── TECHNICAL_VALIDATION.md
├── leave_one_recording_out_results.csv
├── S001R04.edf
├── S001R08.edf
└── S001R12.edf
```

Additional training scripts, experiment results, visualizations, and backups are also included in the project folder.

## Installation

1. Install Python 3.12.
2. Open the project folder in VS Code.
3. Create a virtual environment:

```powershell
python -m venv .venv
```

4. Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running NeuroScope

From the project directory, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run eeg_loader.py
```

Streamlit will open the application in your browser.

## Dataset and Model

The initial AI experiment uses the PhysioNet EEG Motor Movement/Imagery dataset, specifically recordings S001R04, S001R08, and S001R12.

The trained classifier uses Common Spatial Patterns (CSP) for feature extraction and Linear Discriminant Analysis (LDA) for classification.

## Evaluation

The current leave-one-recording-out experiment achieved:

- Mean accuracy: 66.7%
- Mean balanced accuracy: 67.9%

These results are based on three recordings from one participant. They do not establish generalization to unseen participants or clinical validity.

See `TECHNICAL_VALIDATION.md` for the experimental methodology and limitations.

## Disclaimer

NeuroScope is an experimental educational and research project. Its predictions are not medical diagnoses and must not be used for clinical decision-making.

## Project Status
Functional EEG visualization dashboard, signal-processing pipeline, and initial CSP-LDA motor imagery classification system implemented. Recording-level evaluation has been completed using three recordings from one participant. Cross-participant validation and further model development remain future research directions.

Functional EEG dashboard and initial AI motor imagery classification implemented. Further validation and cross-subject evaluation remain future research directions.