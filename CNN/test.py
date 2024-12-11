from flask import Flask, request, jsonify
from prometheus_client import Counter, Summary, generate_latest, start_http_server
import numpy as np
from keras.api.models import load_model
import time

# Charger le modèle
model = load_model("../Models/CNN/Cnn_model.keras")

# Liste des labels pour CIFAR-100 (classes en clair)
cifar100_labels = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

# application Flask
app = Flask(__name__)

# Prometheus metrics
prediction_counter = Counter('total_predictions', 'Total number of predictions')
error_counter = Counter('total_errors', 'Total number of errors')
response_time_summary = Summary('response_time_seconds', 'Time spent processing request')

# Route pour les prédictions
@app.route('/predict_label', methods=['POST'])
@response_time_summary.time()  # Mesurer le temps de réponse
def predict_label():
    try:
        prediction_counter.inc()  # Incrémenter le compteur de prédictions
        data = request.get_json()
        image = np.array(data['inputs']).astype('float32') / 255.0
        image = np.squeeze(image)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        class_id = int(np.argmax(prediction))
        label = cifar100_labels[class_id]
        return jsonify({'class_id': class_id, 'label': label})
    except Exception as e:
        error_counter.inc()  # Incrémenter le compteur d'erreurs
        return jsonify({'error': str(e)}), 400

# Route pour exposer les métriques à Prometheus
@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest()

if __name__ == '__main__':
    start_http_server(8000)  # Expose également les métriques sur le port 8000
    app.run(host='0.0.0.0', port=5000)
