import re
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def normalize(name: str) -> str:
    """
    Minúsculas, elimina tildes y sustituye
    cualquier carácter no alfanumérico por '_'.
    """
    name = name.lower()
    accents = str.maketrans("áéíóúüñ", "aeiouun")
    name = name.translate(accents)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")

# Directorio base y artifacts
BASE_DIR  = Path(__file__).parent
ARTIFACTS = BASE_DIR / "artifacts"

# 1. Mostrar contenido de artifacts (solo para diagnóstico)
print("→ Archivos en artifacts/:")
for f in sorted(ARTIFACTS.iterdir()):
    print("  -", f.name)

# 2. Cargar modelo e imputador
model   = joblib.load(ARTIFACTS / "modelo_random_forest.pkl")
imputer = joblib.load(ARTIFACTS / "imputer.pkl")

# 3. Crear mapeo normalizado de stems a Path
stem_to_path = {
    normalize(p.stem): p
    for p in ARTIFACTS.glob("le_*.pkl")
}

categorical_cols = [
    "1.¿Has recibido una llamada sospechosa de fraude telefónico?",
    "2.¿Respondiste a la llamada?",
    "6.¿Desde qué país parecía provenir la llamada?",
    "7.¿Qué motivo alegaron en la llamada?",
    "9.¿La llamada usó alguna de estas palabras clave?",
    "10.¿Qué tipo de voz escuchaste?",
    "11.¿El tono fue…?",
    "12.¿Cuántas veces has recibido llamadas de ese mismo número?",
    "13.¿El número dejó buzón de voz o mensaje grabado?",
    "14.¿Intentaste devolver la llamada?",
    "15.¿Compartiste algún dato personal o financiero durante la llamada?",
    "16.¿Terminaste la llamada por sospecha?",
    "17.¿Reportaste el número a alguna plataforma o autoridad?",
    "18.¿Qué dispositivo usaste para recibir la llamada?",
    "19.¿Tenías conexión Wi-Fi o datos móviles al recibir la llamada?",
    "20.¿Tu número es personal o empresarial?"
]

# 5. Carga de encoders usando el mapeo
encoders = {}
for col in categorical_cols:
    key = normalize(f"le_{col}")
    pkl_path = stem_to_path.get(key)
    if not pkl_path:
        raise FileNotFoundError(
            f"Falta encoder para columna:\n  {col}\n"
            f"Buscando stem '{key}'. Disponibles:\n  {list(stem_to_path)}"
        )
    encoders[col] = joblib.load(pkl_path)

# 6. Definir y transformar el ejemplo
sample = {
    "1.¿Has recibido una llamada sospechosa de fraude telefónico?": "Si", 
    "2.¿Respondiste a la llamada?": "No", 
    "6.¿Desde qué país parecía provenir la llamada?": "México", 
    "7.¿Qué motivo alegaron en la llamada?": "Bloqueo de cuenta bancaria", 
    "9.¿La llamada usó alguna de estas palabras clave?": "Código SMS", 
    "10.¿Qué tipo de voz escuchaste?": "Persona real", 
    "11.¿El tono fue…?": "Amenazante", 
    "12.¿Cuántas veces has recibido llamadas de ese mismo número?": "1 vez", 
    "13.¿El número dejó buzón de voz o mensaje grabado?": "No", 
    "14.¿Intentaste devolver la llamada?": "No", 
    "15.¿Compartiste algún dato personal o financiero durante la llamada?": "No", 
    "16.¿Terminaste la llamada por sospecha?": "Sí", 
    "17.¿Reportaste el número a alguna plataforma o autoridad?": "No", 
    "18.¿Qué dispositivo usaste para recibir la llamada?": "iPhone", 
    "19.¿Tenías conexión Wi-Fi o datos móviles al recibir la llamada?": "Wi-Fi", 
    "20.¿Tu número es personal o empresarial?": "Personal"
}
df_sample = pd.DataFrame([sample])

def normalize_value(val):
    if isinstance(val, str):
        return val.strip().lower()
    return val

for col in encoders:
    if col in df_sample.columns:
        df_sample[col] = df_sample[col].apply(normalize_value)

# Codificar solo las columnas categóricas (no fecha/hora)
for col, le in encoders.items():
    df_sample[col] = le.transform(df_sample[col])

for col, le in encoders.items():
    print(f"{col}: {le.classes_}")

# Ajustar columnas para que coincidan con el entrenamiento
columns_fit = imputer.feature_names_in_
for col in columns_fit:
    if col not in df_sample.columns:
        df_sample[col] = pd.NA
df_sample = df_sample[[col for col in columns_fit]]

X_sample = pd.DataFrame(
    imputer.transform(df_sample),
    columns=df_sample.columns
)

# 7. Predecir y mostrar resultado
pred   = model.predict(X_sample)[0]
labels = {0: "No", 1: "Si", 2: "No estoy seguro"}
print(f"La predicción para el ejemplo es: {labels[pred]}")

# Solo retorna "Si" o "No"
# ...código existente...
valor = "Si" if pred == 1 else "No"
print(f"La predicción para el ejemplo es: {valor}")

# Guardar el resultado en un archivo de texto
with open("resultado_prediccion.txt", "w") as f:
    f.write(valor)

with open("resultado_prediccion.txt") as f:
    resultado = f.read().strip()
print(resultado)
