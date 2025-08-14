from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Cargar el pipeline guardado
pipeline = joblib.load('model_pipeline.pkl')

# Definir estructura de la solicitud
class CallFeatures(BaseModel):
    llamada_sospechosa: str  # Ej: "Si"
    respondio_llamada: str    # Ej: "No"
    pais_procedencia: str     # Ej: "México"
    motivo_llamada: str       # Ej: "Oferta de premios"

@app.post("/predict")
def predict(features: CallFeatures):
    # Convertir datos de entrada a DataFrame
    input_data = pd.DataFrame([dict(features)])
    
    # Predecir
    prediction = pipeline.predict(input_data)
    
    return {"prediccion": int(prediction[0])}  # 0: No, 1: Si, 2: No seguro