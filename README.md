# Spoken Digit Classification

## Overview
This project builds a lightweight prototype to classify spoken digits (0–9) using audio files from the Free Spoken Digit Dataset (FSDD). The goal is to predict digits quickly and accurately.

## Approach
1. **Load and preprocess data**:  
   - Original parquet dataset contained audio as bytes.  
   - Converted bytes to WAV files and saved paths in `train_paths.parquet`.

2. **Feature extraction**:  
   - Extracted MFCCs, delta MFCCs, and chroma features using `librosa`.  
   - Concatenated features into fixed-length vectors for each audio.

3. **Modeling**:  
   - Trained a `RandomForestClassifier` using the extracted features.  
   - Evaluated on a hold-out test set, achieving ~96.3% accuracy.

4. **Prediction**:  
   - `predict.py` predicts the digit from any new WAV file.  
   - Optional `predict_live.py` can be used for real-time predictions (microphone input).

## Results
- Accuracy: 0.9630  
- Confusion matrix: <img width="408" height="289" alt="image" src="https://github.com/user-attachments/assets/1aca5776-0149-4c24-8e9c-ba4c8a319885" />
 

## Usage
```bash
# Train the model
python src/train.py --train data/train_paths.parquet --path-col path --label-col label

# Predict from a WAV file
python src/predict.py --audio data/train_wavs/0_0.wav
