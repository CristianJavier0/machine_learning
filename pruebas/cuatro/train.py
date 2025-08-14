import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # opcional

# Carga y limpieza de columnas
df = pd.read_csv("datos.csv")
df.columns = [
    "timestamp", "email", "fraud_call", "responded", "phone_number",
    "date_received", "time_received", "country", "reason", "description",
    "keywords", "voice_type", "tone", "call_count", "voicemail",
    "returned_call", "shared_data", "hung_up", "reported", "device",
    "connection", "number_type", "vishing_attempt"
]

# Mapear vishing_attempt y call_count
df["vishing_attempt"] = df["vishing_attempt"].replace({"No estoy seguro": "No"})
y = df["vishing_attempt"].map(lambda x: 1 if x == "Si" else 0)

count_map = {"1 vez": 1, "De 2 a 5 veces": 3, "Más de 5 veces": 6}
df["call_count"] = df["call_count"].map(count_map)

# Selección de características
X = df[[
    "fraud_call", "responded", "country", "reason", "voice_type", "tone",
    "call_count", "voicemail", "returned_call", "shared_data", "hung_up",
    "reported", "device", "connection", "number_type"
]].copy()

print(y.value_counts())

# División
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# (Opcional) SMOTE para balancear la clase minoritaria
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Pipeline
numeric_features = ["call_count"]
categorical_features = [c for c in X.columns if c not in numeric_features]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=200))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["No", "Si"]))
