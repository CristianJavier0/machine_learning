#!/usr/bin/env python3
import argparse
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def infer_column_types(df, target, drop_cols):
    used_cols = [c for c in df.columns if c not in ([target] if target in df.columns else []) and c not in drop_cols]
    # Numeric = those that can be coerced to numeric with minimal NaNs increase
    numeric_cols, categorical_cols = [], []
    for c in used_cols:
        # Try numeric
        coerced = pd.to_numeric(df[c], errors='coerce')
        # Heuristic: if > 80% values numeric (not NaN after coercion), treat as numeric
        ratio = 1.0 - (coerced.isna().sum() / len(coerced))
        if ratio >= 0.8:
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols, used_cols

def build_pipeline(numeric_cols, categorical_cols, rf_n_estimators, rf_max_depth, class_weight):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        n_jobs=-1,
        class_weight=class_weight if class_weight != "none" else None,
        random_state=42
    )

    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", clf)])
    return pipe

def main():
    parser = argparse.ArgumentParser(description="Train and save a fraud-detection style model from CSV.")
    parser.add_argument("--csv", required=True, help="Path to input CSV with features + target")
    parser.add_argument("--target", required=True, help="Target column name (e.g., es_fraude)")
    parser.add_argument("--drop-cols", type=str, default="", help='Comma-separated columns to drop (e.g., "id,telefono")')
    parser.add_argument("--model-out", default="model.pkl", help="Path to save trained pipeline (joblib)")
    parser.add_argument("--metrics-out", default="metrics.json", help="Path to save evaluation metrics (json)")
    parser.add_argument("--model-card-out", default="model_card.json", help="Path to save model metadata (json)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rf-n-estimators", type=int, default=250, help="RandomForest n_estimators")
    parser.add_argument("--rf-max-depth", type=int, default=None, help="RandomForest max_depth")
    parser.add_argument("--class-weight", type=str, default="none", choices=["none","balanced","balanced_subsample"],
                        help="Class weight strategy")

    args = parser.parse_args()

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    df = pd.read_csv(args.csv)

    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found in CSV. Available: {list(df.columns)}")

    y = df[args.target]
    numeric_cols, categorical_cols, used_cols = infer_column_types(df, args.target, drop_cols)

    X = df[numeric_cols + categorical_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y if len(np.unique(y))>1 else None
    )

    pipe = build_pipeline(numeric_cols, categorical_cols, args.rf_n_estimators, args.rf_max_depth, args.class_weight)
    pipe.fit(X_train, y_train)

    # Eval
    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    # AUC if available
    try:
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:,1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        pass

    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipe, args.model_out)

    # Model card
    model_card = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "library": "scikit-learn",
        "algorithm": "RandomForestClassifier",
        "params": {
            "n_estimators": args.rf_n_estimators,
            "max_depth": args.rf_max_depth,
            "class_weight": args.class_weight,
            "random_state": 42
        },
        "preprocessing": {
            "numeric_imputer": "median + StandardScaler(with_mean=False)",
            "categorical_imputer": "most_frequent + OneHotEncoder(handle_unknown='ignore')"
        },
        "feature_columns": {
            "numeric": numeric_cols,
            "categorical": categorical_cols
        },
        "dropped_columns": drop_cols,
        "target": args.target,
        "metrics_path": args.metrics_out,
        "model_path": args.model_out
    }
    with open(args.model_card_out, "w") as f:
        json.dump(model_card, f, indent=2)

    print("Training complete.")
    print("Saved pipeline to:", args.model_out)
    print("Saved metrics to:", args.metrics_out)
    print("Saved model card to:", args.model_card_out)

if __name__ == "__main__":
    main()
