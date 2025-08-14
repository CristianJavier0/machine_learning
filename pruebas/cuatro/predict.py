#!/usr/bin/env python3
import argparse
import pandas as pd
import joblib

def main():
    parser = argparse.ArgumentParser(description="Load a saved pipeline and predict on new CSV data.")
    parser.add_argument("--model", required=True, help="Path to saved pipeline (joblib)")
    parser.add_argument("--csv", required=True, help="CSV with the SAME feature columns used in training")
    parser.add_argument("--out", default="predicciones.csv", help="Where to save predictions CSV")
    parser.add_argument("--drop-cols", type=str, default="", help='Comma-separated cols to drop at predict time (must match training)')

    args = parser.parse_args()

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    pipe = joblib.load(args.model)

    df = pd.read_csv(args.csv)
    # If user wants to drop cols (ids, phone numbers) for inference consistency
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Predict
    y_pred = pipe.predict(df)
    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(df)[:,1]
        except Exception:
            proba = None

    out_df = df.copy()
    out_df["pred"] = y_pred
    if proba is not None:
        out_df["proba"] = proba

    out_df.to_csv(args.out, index=False)
    print("Predictions saved to:", args.out)

if __name__ == "__main__":
    main()
