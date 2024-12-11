from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.api.models import load_model
from PIL import Image

# Charge le modèle
model = load_model("../../Models/CNN/Cnn_model.keras")

# Liste des labels CIFAR-100
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

# Application Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupére l'image téléchargée
        file = request.files['image']
        if not file:
            return jsonify({'error': 'Aucune image téléchargée'}), 400

        img = Image.open(file)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Redimensionne et normalise l'image
        img = img.resize((32, 32))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prédit
        prediction = model.predict(img_array)
        class_id = int(np.argmax(prediction))
        label = cifar100_labels[class_id]

        return jsonify({'label': label, 'class_id': class_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
