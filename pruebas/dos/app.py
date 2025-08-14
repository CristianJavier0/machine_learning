# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1. Cargar datos
data = pd.read_csv("datos.csv")

# 1. Normaliza nombres de columnas (elimina espacios y \n)
data.columns = data.columns.str.strip()

# 2. Limpieza inicial
cols_to_drop = [
    "Marca temporal", "Nombre de usuario", 
    "3. Ingresa el número de la llamada",
]
data = data.drop(columns=cols_to_drop, errors="ignore")

required_columns = [
    "4. ¿En qué fecha recibiste la llamada?",
    "5. ¿A qué hora recibiste la llamada?",
    # ... otras columnas necesarias
]

missing_cols = set(required_columns) - set(data.columns)
if missing_cols:
    print(f"¡Columnas faltantes! {missing_cols}")
    # Opción 1: Crear columnas faltantes con valores por defecto
    for col in missing_cols:
        data[col] = "Desconocido"  # o pd.NaT para fechas

if '5. ¿A qué hora recibiste la llamada?' not in data.columns:
    data['5. ¿A qué hora recibiste la llamada?'] = "00:00"
    
# 3. Definir columnas por tipo
categorical_cols = [
    "1. ¿Has recibido una llamada sospechosa de fraude telefónico?",
    "2. ¿Respondiste a la llamada?",
    "6. ¿Desde qué país parecía provenir la llamada?",
    "7. ¿Qué motivo alegaron en la llamada?",
    "9. ¿La llamada usó alguna de estas palabras clave?\n",  # Añade \n
    "10. ¿Qué tipo de voz escuchaste?",
    "11. ¿El tono fue…?",
    "12. ¿Cuántas veces has recibido llamadas de ese mismo número?",
    "13. ¿El número dejó buzón de voz o mensaje grabado?",
    "14. ¿Intentaste devolver la llamada?",
    "15.¿Compartiste algún dato personal o financiero durante la llamada?",
    "16.¿Terminaste la llamada por sospecha?",
    "17.¿Reportaste el número a alguna plataforma o autoridad?",
    "18.¿Qué dispositivo usaste para recibir la llamada?",
    "19.¿Tenías conexión Wi-Fi o datos móviles al recibir la llamada?",
    "20.  ¿Tu número es personal o empresarial?"
]

text_col = "8. Cuéntanos de forma breve y con tus propias palabras lo que sucedió en esa llamada que recibiste"

# Función para extraer características de fecha/hora
def extract_datetime_features(X):
    df = pd.DataFrame(X, columns=["4. ¿En qué fecha recibiste la llamada?", "5. ¿A qué hora recibiste la llamada?"])
    df["fecha_llamada"] = pd.to_datetime(df["4. ¿En qué fecha recibiste la llamada?"], errors='coerce')
    df["hora_llamada"] = pd.to_datetime(df["5. ¿A qué hora recibiste la llamada?"], format='%H:%M', errors='coerce').dt.hour
    df["dia_semana"] = df["fecha_llamada"].dt.dayofweek
    df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["es_madrugada"] = ((df["hora_llamada"] >= 0) & (df["hora_llamada"] <= 6)).astype(int)
    return df[["dia_semana", "es_fin_de_semana", "es_madrugada"]]

# 4. Preprocesamiento con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('text', TfidfVectorizer(max_features=50, stop_words='spanish'), text_col),
        ('datetime', FunctionTransformer(extract_datetime_features), ["4. ¿En qué fecha recibiste la llamada?", "5. ¿A qué hora recibiste la llamada?"])
    ],
    remainder='drop'
)

# 5. Definir target y features CORREGIDO
# 3. Definir X e y
y = data["21.Según tu experiencia, ¿consideras esta llamada como un intento de fraude (vishing)?"].map({"Si": 1, "No": 0, "No estoy seguro": 2})
X = data.drop(columns=["21.Según tu experiencia, ¿consideras esta llamada como un intento de fraude (vishing)?"])

# 4. Manejo de clases desbalanceadas
class_dist = y.value_counts()
print("Distribución de clases original:")
print(class_dist)

if class_dist.min() < 2:
    print("\nAplicando filtrado de clases minoritarias...")
    clases_validas = class_dist[class_dist >= 2].index
    mask = y.isin(clases_validas)
    X = X[mask]
    y = y[mask]
    print("Distribución después de filtrar:")
    print(y.value_counts())

# 5. División de datos
if len(y.unique()) > 1 and y.value_counts().min() >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    print("\nAdvertencia: No se usa stratify por desbalance de clases")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
# 7. Crear y entrenar pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# 8. Evaluar
y_pred = pipeline.predict(X_test)
present_classes = set(y_test)
target_names_all = ["No", "Si", "No estoy seguro"]
target_names = [name for i, name in enumerate(target_names_all) if i in present_classes]

print("Clases presentes en y_test:", present_classes)
print(classification_report(y_test, y_pred, target_names=target_names))

# 9. Guardar pipeline
joblib.dump(pipeline, 'pipeline_fraude_telefonico.pkl')
print("✅ Pipeline guardado como 'pipeline_fraude_telefonico.pkl'")