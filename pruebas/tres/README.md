# ML Pipeline (Linux) — Train & Save Model, Then Predict

This project provides two CLI scripts:

- `train.py`: trains a model from a CSV and **saves** a ready-to-use pipeline (`model.pkl`)
- `predict.py`: loads the saved pipeline to make predictions on **new CSV data**

Both scripts are Linux-friendly and rely on common, stable packages.

---

## 1) Setup (Linux)

```bash
# 1) Create & activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2) Train a Model and Save It

Assume your CSV is `data.csv` and the target column is `es_fraude` (0/1).  
If your target column has a different name, just change `--target`.

```bash
python train.py --csv data.csv --target es_fraude --model-out model.pkl --metrics-out metrics.json
```

What the training script does:
- Splits train/test automatically (80/20)
- Builds a robust preprocessing + model **Pipeline**
  - Numeric: impute missing values, scale
  - Categorical: impute missing values, one-hot encode
- Trains a `RandomForestClassifier`
- Evaluates on the test split and writes `metrics.json`
- Saves the full pipeline to `model.pkl` (ready to be loaded for inference)
- Also writes a compact `model_card.json` with metadata

---

## 3) Predict with the Saved Model

Create a CSV with the **same feature columns** (without the target) called `nuevos.csv`. Then:

```bash
python predict.py --model model.pkl --csv nuevos.csv --out predicciones.csv
```

The script will write a CSV including the original rows plus:
- `pred`: the predicted class
- `proba`: the predicted probability of the positive class (if supported by the model)

---

## 4) Tips

- If your CSV has date/time columns, keep them as strings; the pipeline will treat them as categorical unless numeric-parsable.
- If you need to ignore certain columns (e.g., identifiers like phone number), use `--drop-cols` in `train.py` and `predict.py` to ensure train/predict use the same feature set.
- For imbalanced data, try `--class-weight balanced` or tune `--rf-n-estimators`, `--rf-max-depth`.

---

## 5) Reproducibility

- Set a fixed seed with `--seed` (default 42).

---

## 6) Example with typical columns

```bash
python train.py --csv llamadas.csv --target es_fraude \
  --drop-cols "id_llamada,numero_llamante" \
  --rf-n-estimators 300 --rf-max-depth 12
```

Then predict:

```bash
python predict.py --model model.pkl --csv llamadas_nuevas.csv --out predicciones.csv --drop-cols "id_llamada,numero_llamante"
```

---

Happy modeling!
