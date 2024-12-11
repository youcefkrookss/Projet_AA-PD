import json
import os
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import cifar100

# Charger les données CIFAR-100
(_, _), (x_test, y_test) = cifar100.load_data()

# Créer un dossier pour stocker les fichiers JSON
output_dir = "json_test_data_pca"
os.makedirs(output_dir, exist_ok=True)

# Prétraitement : Normalisation (0-1) et aplatir les images
x_test_normalized = x_test.astype('float32') / 255.0
x_test_flattened = x_test_normalized.reshape(x_test_normalized.shape[0], -1)  # De (32, 32, 3) à (3072,)

# Charger ou ajuster PCA pour réduire à 500 dimensions
# Si vous avez déjà un modèle PCA entraîné, chargez-le ici
pca = PCA(n_components=500)
x_test_pca = pca.fit_transform(x_test_flattened)  # Appliquer PCA sur les données de test

# Nombre de fichiers à générer
num_files = 10  # Changez ce nombre pour générer plus ou moins de fichiers

for i in range(num_files):
    # Sélectionner une image de test aléatoire
    index = np.random.randint(0, len(x_test_pca))
    input_data = x_test_pca[index].tolist()  # Convertir en liste après PCA

    # Créer un fichier JSON
    file_name = f"test_data_pca_{i+1}.json"
    file_path = os.path.join(output_dir, file_name)

    # Sauvegarder l'image dans un fichier JSON
    with open(file_path, "w") as json_file:
        json.dump({"input_data": input_data}, json_file)

    print(f"Fichier JSON généré : {file_path}")

print(f"\nTous les fichiers JSON avec PCA ont été enregistrés dans le dossier '{output_dir}'")
