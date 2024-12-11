import requests
import json
import numpy as np
# from tensorflow.keras.datasets import cifar100
import matplotlib.pyplot as plt
from keras.api.datasets import cifar100
import random

# Liste des labels pour CIFAR-100
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

# Charge les données CIFAR-100 (test set)
(_, _), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# URL de l'API Flask
url = "http://127.0.0.1:5000/predict_label"

# Nombre d'images à tester
num_images = 5

# Sélectionne `num_images` indices aléatoires
random_indices = random.sample(range(len(x_test)), num_images)

# Teste les premières `num_images` images de l'ensemble de test
for i, index in enumerate(random_indices):
    image = x_test[index]
    true_label_id = int(y_test[index])
    true_label = cifar100_labels[true_label_id]

    # Prépare les données pour l'API
    payload = {"inputs": image.tolist()}

    # Envois la requête POST
    response = requests.post(url, json=payload)

    # Vérifie la réponse
    if response.status_code == 200:
        response_data = response.json()
        predicted_label = response_data['label']
        predicted_class_id = response_data['class_id']

        # Affiche les résultats
        print(f"Image {i + 1}:")
        print(f"  True Label: {true_label} (class ID: {true_label_id})")
        print(f"  Predicted Label: {predicted_label} (class ID: {predicted_class_id})")

        # Affiche l'image avec les labels
        plt.imshow(image)
        plt.title(f"True: {true_label}, Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()
    else:
        print(f"Error with image {i + 1}: {response.json()}")
