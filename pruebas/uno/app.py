# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Cargar datos
data = pd.read_csv("datos.csv")

# 2. Limpieza inicial
# Eliminar columnas redundantes o no útiles
cols_to_drop = [
    "Marca temporal", "Nombre de usuario", 
    "3. Ingresa el número de la llamada", "  3. Ingresa el número de la llamada",
    "8. Cuéntanos de forma breve y con tus propias palabras lo que sucedió en esa llamada que recibiste"  # (Opcional: NLP después)
]
data = data.drop(columns=cols_to_drop, errors="ignore")

# 3. Preprocesamiento
# Codificar target (Si=1, No=0, No estoy seguro=2)
target_map = {"Si": 1, "No": 0, "No estoy seguro": 2}
data["target"] = data["21.Según tu experiencia, ¿consideras esta llamada como un intento de fraude (vishing)?"].map(target_map)
data = data.drop(columns=["21.Según tu experiencia, ¿consideras esta llamada como un intento de fraude (vishing)?"])

# Codificar features categóricas
categorical_cols = data.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # Evita errores con NaN

# Imputar NaNs (si los hay)
imputer = SimpleImputer(strategy="most_frequent")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 4. Dividir datos
X = data_imputed.drop(columns=["target"])
y = data_imputed["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluar
y_pred = model.predict(X_test)

# Verificar clases presentes
present_classes = set(y_test)
target_names_all = ["No", "Si", "No estoy seguro"]
target_names = [name for i, name in enumerate(target_names_all) if i in present_classes]

print("Clases presentes en y_test:", present_classes)
print(classification_report(y_test, y_pred, target_names=target_names))