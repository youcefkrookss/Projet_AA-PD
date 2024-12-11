import shap
from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import time
import logging

# Charger le modèle sauvegardé
model = tf.keras.models.load_model("final_model/data/model.keras")

# Initialiser FastAPI
app = FastAPI()

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser SHAP Explainer avec un jeu de données réduit
background_data = np.random.rand(10, 500)  # Réduction à 10 échantillons
explainer = shap.KernelExplainer(model.predict, background_data)

# Définir le schéma des données d'entrée
class PredictionRequest(BaseModel):
    input_data: list  # Une liste contenant les caractéristiques

@app.get("/")
def home():
    return {"message": "Welcome to the CIFAR-100 prediction API"}

@app.post("/predict")
def predict(request: PredictionRequest):
    # Prétraitement des données
    input_data = np.array(request.input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    return {"predicted_class": predicted_class, "probabilities": prediction.tolist()}

@app.post("/explain", summary="Explain the prediction")
async def explain(request: PredictionRequest):
    """
    Endpoint pour expliquer une prédiction en utilisant SHAP.
    """
    input_data = np.array(request.input_data).reshape(1, -1)

    # Mesurer le temps de calcul
    start_time = time.time()
    shap_values = explainer.shap_values(input_data)
    execution_time = time.time() - start_time
    logging.info(f"SHAP computation took {execution_time:.2f} seconds.")

    return {
        "shap_values": shap_values,
        "base_values": explainer.expected_value.tolist(),
        "execution_time": execution_time
    }
