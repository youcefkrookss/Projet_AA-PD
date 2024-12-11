import requests
import json

# Charger le fichier JSON
with open("../Models/CNN/serving_input_example.json", "r") as file:
    data = json.load(file)

url = "http://127.0.0.1:5000/predict_label"

# Envoyer la requête POST avec les données de l'image
response = requests.post(url, json=data)

# Afficher la réponse
if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.json())
