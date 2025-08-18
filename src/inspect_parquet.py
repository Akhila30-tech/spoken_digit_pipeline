import argparse
import pandas as pd
from features import guess_columns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='Path to parquet file')
    ap.add_argument('--label-col', default=None)
    ap.add_argument('--audio-col', default=None)
    ap.add_argument('--path-col',  default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.file)
    print(f"Loaded: {args.file}  shape={df.shape}")
    print("\nColumns:")
    for c in df.columns:
        print(f" - {c}  ({df[c].dtype})")
    print("\nHead:")
    print(df.head())

    a_col, p_col, l_col = guess_columns(df)

    # Respect user overrides
    a_col = args.audio_col or a_col
    p_col = args.path_col  or p_col
    l_col = args.label_col or l_col

    print("\nGuessed/Final columns:")
    print(f"  audio_col: {a_col}")
    print(f"  path_col : {p_col}")
    print(f"  label_col: {l_col}")

if __name__ == '__main__':
    main()