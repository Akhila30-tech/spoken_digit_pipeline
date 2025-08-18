import numpy as np
import pandas as pd
import librosa
from typing import Optional, Tuple, List

# ---- Column guessing ----
CANDIDATE_AUDIO_ARRAY = ['audio', 'audio_array', 'samples', 'waveform', 'array']
CANDIDATE_AUDIO_PATH  = ['path', 'filepath', 'file', 'wav_path', 'wav', 'audio_path']
CANDIDATE_LABEL       = ['label', 'digit', 'class', 'target', 'y']

def _first_hit(cands: List[str], cols: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for c in cands:
        if c in cols_lower:
            return cols_lower[c]
    # partial match (e.g., 'audio.values')
    for name in cols:
        low = name.lower()
        for c in cands:
            if c in low:
                return name
    return None

def guess_columns(df: pd.DataFrame):
    cols = list(df.columns)
    audio_col = _first_hit(CANDIDATE_AUDIO_ARRAY, [c.lower() for c in cols])
    path_col  = _first_hit(CANDIDATE_AUDIO_PATH,  [c.lower() for c in cols])
    label_col = _first_hit(CANDIDATE_LABEL,       [c.lower() for c in cols])
    # Disambiguate if both audio_col and path_col found but one is obviously wrong
    if audio_col and isinstance(df.iloc[0][audio_col], str):
        audio_col = None
    if path_col and not isinstance(df.iloc[0][path_col], str):
        path_col = None
    return audio_col, path_col, label_col

# ---- Audio loading ----
def load_audio_from_row(row, audio_col: Optional[str], path_col: Optional[str], sr_target: Optional[int]=None) -> Tuple[np.ndarray, int]:
    if audio_col is not None and audio_col in row:
        # raw array stored; try to coerce
        data = row[audio_col]
        sig = np.asarray(data, dtype=np.float32).squeeze()
        # try to detect sr from row if exists
        sr = None
        for key in ['sr','sample_rate','rate','fs']:
            if key in row and isinstance(row[key], (int, float)):
                sr = int(row[key])
                break
        if sr is None:
            sr = 16000
        if sr_target is not None and sr != sr_target:
            sig = librosa.resample(sig, orig_sr=sr, target_sr=sr_target)
            sr = sr_target
        return sig, sr
    elif path_col is not None and path_col in row:
        path = str(row[path_col])
        sig, sr = librosa.load(path, sr=None, mono=True)
        if sr_target is not None and sr != sr_target:
            sig = librosa.resample(sig, orig_sr=sr, target_sr=sr_target)
            sr = sr_target
        return sig, sr
    else:
        raise ValueError("No valid audio column or path column found for this row.")

# ---- Feature extraction ----
def extract_features(signal: np.ndarray, sr: int) -> np.ndarray:
    # Ensure float32 mono
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    signal = signal.astype(np.float32)

    # Trim leading/trailing silence for stability
    yt, _ = librosa.effects.trim(signal, top_db=25)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)

    # Delta
    # mfcc_delta = librosa.feature.delta(mfcc)
 # mfcc_delta_mean = mfcc_delta.mean(axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=yt, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    feat = np.concatenate([mfcc_mean, chroma_mean], axis=0)
    return feat