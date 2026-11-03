# NeuroScope — Technical Validation Report

## 1. Project Overview

NeuroScope is an EEG analysis dashboard designed to load EEG recordings, visualize electrical brain activity, inspect signal characteristics, and explore AI-based classification of motor imagery signals.

The project combines EEG signal processing with machine learning to investigate how patterns in EEG recordings can be classified.

## 2. Dataset

**Dataset:** EEG Motor Movement/Imagery Dataset  
**Source:** PhysioNet  
**Participant:** S001  
**Recordings used:**
- S001R04.edf
- S001R08.edf
- S001R12.edf

Dataset characteristics:
- 64 EEG channels
- Sampling frequency: 160 Hz
- Recording duration: 125 seconds per recording
- Classification tasks: imagined left-hand and right-hand movements

The experiment uses recordings from one participant. It does not represent the entire PhysioNet dataset.

## 3. Signal Processing

The project uses two separate signal-processing pipelines.

### 3.1 Dashboard Signal Processing

The interactive EEG dashboard includes:

1. Loading EEG recordings using MNE-Python.
2. Selecting one or multiple EEG channels.
3. Applying a fourth-order Butterworth IIR band-pass filter between 1 and 40 Hz.
4. Displaying raw and filtered EEG signals for comparison.
5. Estimating Power Spectral Density (PSD) using Welch's method.
6. Analyzing power across five frequency bands:
   - Delta: 1–4 Hz
   - Theta: 4–8 Hz
   - Alpha: 8–13 Hz
   - Beta: 13–30 Hz
   - Gamma: 30–40 Hz

*Note: Frequency-band analysis currently uses the first selected EEG channel.*

### 3.2 AI Model Preprocessing

The AI classification pipeline includes:

1. Loading EEG recordings using MNE-Python.
2. Selecting EEG channels.
3. Applying a band-pass filter between 8 and 30 Hz.
4. Extracting epochs from 1.0 to 3.5 seconds after the relevant event.
5. Extracting spatial features using Common Spatial Patterns (CSP).
6. Classifying the extracted features using Linear Discriminant Analysis (LDA).

## 4. Machine Learning Model

**Feature extraction:** Common Spatial Patterns (CSP)  
**Classifier:** Linear Discriminant Analysis (LDA)  
**CSP components:** 4

The CSP-LDA pipeline learns spatial patterns associated with the two motor imagery classes and uses them to classify EEG trials.

## 5. Evaluation Methodology

A leave-one-recording-out evaluation was implemented.

Three experiments were conducted:

- Experiment 1: Train on R08 and R12; test on R04.
- Experiment 2: Train on R04 and R12; test on R08.
- Experiment 3: Train on R04 and R08; test on R12.

Each recording was held out once for testing.

This evaluates performance across separate recordings from the same participant. It does not establish generalization to unseen participants.

## 6. Experimental Results

| Held-out recording | Accuracy | Balanced accuracy |
|---|---:|---:|
| S001R04 | 80.0% | 81.25% |
| S001R08 | 60.0% | 59.82% |
| S001R12 | 60.0% | 62.50% |
| **Mean** | **66.7%** | **67.9%** |

The results show variation in classification performance across recordings.

The confusion matrices and detailed classification metrics are available in `leave_one_recording_out_results.csv`.

The following functionality was implemented and tested:

**EEG Dashboard**
- EDF recording upload.
- EEG metadata extraction.
- Single- and multichannel selection.
- Interactive EEG signal visualization.
- Playback controls (Play, Pause, Reset).
- Signal statistics.

**Signal Processing**
- Fourth-order Butterworth IIR band-pass filtering (1–40 Hz).
- Raw and filtered signal comparison.
- Power Spectral Density estimation using Welch's method.
- Frequency-band power analysis.

**AI Classification**
- Trained CSP + LDA model loading.
- Motor imagery classification (T1/T2).
- Sampling-frequency compatibility check.
- Required-channel compatibility check.
- Predictions on compatible EEG recordings.

**Model Evaluation**
- Leave-one-recording-out evaluation across three recordings.
- Accuracy and balanced accuracy calculations.
- Detailed classification metrics.
- Confusion matrices and results visualization.
## 8. Limitations

1. The current evaluation uses only participant S001.
2. Cross-subject generalization has not been evaluated.
3. The classifier is designed for the specific motor imagery task and is not a general-purpose EEG diagnostic model.
4. Predictions depend on compatible recording characteristics, including sampling frequency, channel configuration, and event annotations.
5. The reported accuracy does not establish clinical validity.

## 9. Conclusion

NeuroScope demonstrates an integrated workflow for EEG visualization, signal processing, and experimental motor imagery classification.

The CSP-LDA model achieved a mean accuracy of 66.7% and mean balanced accuracy of 67.9% in the three-fold leave-one-recording-out evaluation.

The next research direction is to investigate model generalization across different participants and evaluate the robustness of EEG classification under varying recording conditions.

**Project status:** Functional dashboard and initial AI evaluation completed. Further technical validation remains necessary.