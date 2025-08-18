import argparse, json, os, pandas as pd, numpy as np
from joblib import load
from tqdm import tqdm
from features import guess_columns, load_audio_from_row, extract_features

def predict_on_wav(model, sr, wav_path):
    import librosa
    sig, r = librosa.load(wav_path, sr=None, mono=True)
    if r != sr:
        sig = librosa.resample(sig, orig_sr=r, target_sr=sr)
        r = sr
    feat = extract_features(sig, r)
    pred = model.predict([feat])[0]
    return pred

def predict_on_parquet(model, sr, parquet_path, out_csv, audio_col=None, path_col=None):
    df = pd.read_parquet(parquet_path)
    a_col, p_col, _ = guess_columns(df)
    a_col = audio_col or a_col
    p_col = path_col or p_col

    preds = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc='Predicting'):
        row = row._asdict()
        sig, r = load_audio_from_row(row, a_col, p_col, sr_target=sr)
        feat = extract_features(sig, r)
        preds.append(model.predict([feat])[0])
    out = pd.DataFrame({'prediction': preds})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/digit_rf.joblib')
    ap.add_argument('--meta',  default='models/model_meta.json')
    ap.add_argument('--audio', default=None, help='Path to a single WAV file')
    ap.add_argument('--parquet', default=None, help='Path to parquet for batch predictions')
    ap.add_argument('--out', default='outputs/predictions.csv')
    ap.add_argument('--audio-col', default=None)
    ap.add_argument('--path-col', default=None)
    args = ap.parse_args()

    model = load(args.model)
    with open(args.meta) as f:
        meta = json.load(f)
    sr = meta.get('sr', 16000)

    if args.audio:
        pred = predict_on_wav(model, sr, args.audio)
        print(f"Prediction for {args.audio}: {pred}")
    elif args.parquet:
        out_csv = predict_on_parquet(model, sr, args.parquet, args.out, audio_col=args.audio_col, path_col=args.path_col)
        print(f"Wrote predictions -> {out_csv}")
    else:
        raise SystemExit("Provide --audio PATH or --parquet PATH")

if __name__ == '__main__':
    main()