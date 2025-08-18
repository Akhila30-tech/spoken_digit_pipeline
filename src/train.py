import argparse, json, os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

from features import guess_columns, load_audio_from_row, extract_features

def build_xy(df, audio_col, path_col, sr, have_labels, label_col):
    X, y = [], []
    it = tqdm(df.itertuples(index=False), total=len(df), desc='Featurizing')
    for row in it:
        row = row._asdict() if hasattr(row, '_asdict') else row._asdict()
        sig, r = load_audio_from_row(row, audio_col, path_col, sr_target=sr)
        feat = extract_features(sig, r)
        X.append(feat)
        if have_labels:
            y.append(row[label_col])
    X = np.stack(X)
    y = np.array(y) if have_labels else None
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True, help='Path to train parquet')
    ap.add_argument('--test', default=None, help='Optional path to test parquet (with labels)')
    ap.add_argument('--sr', type=int, default=16000)
    ap.add_argument('--audio-col', default=None)
    ap.add_argument('--path-col',  default=None)
    ap.add_argument('--label-col', default=None)
    ap.add_argument('--model-out', default='models/digit_rf.joblib')
    ap.add_argument('--meta-out',  default='models/model_meta.json')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    train_df = pd.read_parquet(args.train)
    a_col, p_col, l_col = guess_columns(train_df)
    a_col = args.audio_col or a_col
    p_col = args.path_col  or p_col
    l_col = args.label_col or l_col

    if l_col is None:
        raise ValueError("Could not find a label column. Pass --label-col explicitly.")

    print(f"Using columns -> audio_col={a_col}  path_col={p_col}  label_col={l_col}")
    X, y = build_xy(train_df, a_col, p_col, args.sr, have_labels=True, label_col=l_col)

    if args.test:
        test_df = pd.read_parquet(args.test)
        # Try to guess columns for test too (but reuse audio/path cols if available)
        a2, p2, l2 = guess_columns(test_df)
        a2 = a2 or a_col
        p2 = p2 or p_col
        l2 = l2 or l_col
        X_test, y_test = build_xy(test_df, a2, p2, args.sr, have_labels=(l2 is not None), label_col=l2)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X, y = X_train, y_train

    clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    clf.fit(X, y)

    # Evaluate
    if args.test:
        if 'y_test' not in locals():
            # If l2 was None, we can't score
            print("Test set has no labels. Skipping accuracy.")
            y_test = None
        else:
            pass
    if 'y_test' in locals() and y_test is not None:
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    else:
        print("Held-out split accuracy:")
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save artifacts
    dump(clf, args.model_out)
    meta = dict(sr=args.sr, audio_col=a_col, path_col=p_col, label_col=l_col, feature='mfcc+delta+chroma')
    with open(args.meta_out, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model -> {args.model_out}")
    print(f"Saved meta  -> {args.meta_out}")

if __name__ == '__main__':
    main()