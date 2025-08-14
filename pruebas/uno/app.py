import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import re

def normalize(name: str) -> str:
    name = name.lower()
    accents = str.maketrans("áéíóúüñ", "aeiouun")
    name = name.translate(accents)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")

# 1. Cargar datos
data = pd.read_csv("datos1.csv")

# 2. Limpieza inicial
cols_to_drop = [
    "Marca temporal",
    "Nombre de usuario",
    "3. Ingresa el número de la llamada",
    "8. Cuéntanos de forma breve y con tus propias palabras lo que sucedió en esa llamada que recibiste"
]
data = data.drop(columns=cols_to_drop, errors="ignore")

# 3. Preprocesamiento

# 3.1 Codificar target
target_map = {"Si": 1, "No": 0, "No estoy seguro": 2}
data["target"] = data[
    "21.Según tu experiencia, ¿consideras esta llamada como un intento de fraude (vishing)?"
].map(target_map)
data = data.drop(
    columns=["21.Según tu experiencia, ¿consideras esta llamada como un intento de fraude (vishing)?"]
)

# 3.2 Procesar columnas de fecha y hora
fecha_col = "4. ¿En qué fecha recibiste la llamada?"
hora_col = "5. ¿A qué hora recibiste la llamada? (en formato de 24hrs)"

if fecha_col in data.columns:
    data[fecha_col] = pd.to_datetime(data[fecha_col], errors="coerce").dt.dayofyear

if hora_col in data.columns:
    data[hora_col] = pd.to_datetime(data[hora_col], format="%H:%M", errors="coerce").dt.hour

# 3.3 Identificar columnas categóricas y aplicar LabelEncoder (excepto fecha y hora)
categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
encoders = {}

for col in categorical_cols:
    if col not in [fecha_col, hora_col]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

# 3.4 Imputar valores faltantes con la moda
imputer = SimpleImputer(strategy="most_frequent")
data_imputed = pd.DataFrame(
    imputer.fit_transform(data),
    columns=data.columns
)

# 4. División de datos
X = data_imputed.drop(columns=["target"])
y = data_imputed["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 5. Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluar en el conjunto de prueba
y_pred = model.predict(X_test)
present_classes = set(y_test)
target_names_all = ["No", "Si", "No estoy seguro"]
target_names = [name for i, name in enumerate(target_names_all) if i in present_classes]

print("Clases presentes en y_test:", present_classes)
print(classification_report(y_test, y_pred, target_names=target_names))

# 7. Crear carpeta de artefactos si no existe
artifact_dir = "artifacts"
os.makedirs(artifact_dir, exist_ok=True)

# 8. Guardar modelo y preprocesadores
joblib.dump(model, os.path.join(artifact_dir, "modelo_random_forest.pkl"))
joblib.dump(imputer, os.path.join(artifact_dir, "imputer.pkl"))

# Guardar cada LabelEncoder con nombre normalizado
for col, le in encoders.items():
    safe_name = normalize(f"le_{col}")
    joblib.dump(le, os.path.join(artifact_dir, f"{safe_name}.pkl"))

print(f"Modelo, imputer y encoders guardados en carpeta `{artifact_dir}`.")