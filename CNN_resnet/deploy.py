from flask import Flask, request, jsonify
import numpy as np
# from tensorflow.keras.models import load_model

from keras.api.models import load_model

# Charge le modèle sauvegardé
model = load_model("../Models/final_model/data/model.keras")

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

# Route pour la prédiction
@app.route('/predict_class', methods=['POST'])
def predict_class():
    """
    Route pour prédire uniquement la classe la plus probable.
    """
    try:
        data = request.get_json()
        image = np.array(data['inputs']).astype('float32') / 255.0
        image = np.squeeze(image)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        class_id = int(np.argmax(prediction))
        return jsonify({'class_id': class_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_probabilities', methods=['POST'])
def predict_probabilities():
    """
    Route pour prédire les probabilités pour toutes les classes.
    """
    try:
        data = request.get_json()
        image = np.array(data['inputs']).astype('float32') / 255.0
        image = np.squeeze(image)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        probabilities = prediction.tolist()
        return jsonify({'probabilities': probabilities})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_label', methods=['POST'])
def predict_label():
    """
    Route pour prédire le label (nom de la classe) de l'image.
    """
    try:
        data = request.get_json()
        image = np.array(data['inputs']).astype('float32') / 255.0
        image = np.squeeze(image)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        class_id = int(np.argmax(prediction))
        label = cifar100_labels[class_id]
        return jsonify({'class_id': class_id, 'label': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Vérification du service (health check)
@app.route('/health', methods=['GET'])
def health():
    return "Service is up and running!"


# Démarrer le serveur Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


