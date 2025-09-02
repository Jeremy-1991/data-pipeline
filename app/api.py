from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from src.utils import load_params

# Charger les paramètres (chemins du modèle, scaler, dataset, etc.)
params = load_params()
model_path = params["model"]["path"]
preprocessor_path = params["preprocessor"]["path"]

# Charger modèle et scaler
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

app = FastAPI(
    title="Prédiction de Churn",
    description="Application de prédiction de Churn <br>Une version par API pour faciliter la réutilisation du modèle",
)

# Définir le schéma d'entrée pour les données
class CustomerData(BaseModel):
    Surname: str
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


# Endpoint de test
@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de prédiction !"}

# Endpoint de prédiction
@app.post("/predict", tags=["Predict"])
def predict(data: CustomerData) -> str:
    df = pd.DataFrame([data.model_dump()]) # model_dump formatte les données pour pandas

    preprocessed_data = preprocessor.transform(df)

    prediction = (
        "Exited" if int(model.predict(preprocessed_data)) == 1 else "Not Exited"
    )

    return prediction