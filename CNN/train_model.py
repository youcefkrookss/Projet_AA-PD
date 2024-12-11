import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import argparse
import dagshub
dagshub.init(repo_owner='youcefkrookss', repo_name='test', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/youcefkrookss/test.mlflow')
mlflow.set_experiment(experiment_name="CNN_Cifar-100")


# Active l'autologging de MLflow
mlflow.tensorflow.autolog()

# Charge les données CIFAR-100
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Fonction pour créer le modèle CNN
def create_final_cnn_model():
    model = Sequential()

    # Première couche de convolution
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))  # CIFAR-100 input shape
    model.add(BatchNormalization())

    # Deuxième couche de convolution
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # Troisième couche de convolution
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Quatrième couche de convolution
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # Cinquième couche de convolution
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Sixième couche de convolution
    model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # Applatissement des caractéristiques pour les couches entièrement connectées
    model.add(Flatten())

    # Couche entièrement connectée
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Couche de sortie avec 100 classes (pour CIFAR-100)
    model.add(Dense(100, activation='softmax'))

    # Compilation du modèle avec l'optimiseur Adam
    model.compile(
        optimizer=Adam(learning_rate=0.0009472383604157623),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Fonction principale
def train_model(epochs, batch_size):
    model = create_final_cnn_model()
    x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, height_shift_range=0.05, horizontal_flip=True, shear_range=0.05)
    datagen.fit(x_train_split)

    # Callback pour réduire le taux d'apprentissage si la perte de validation ne diminue pas
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    # Callback pour arrêter l'entraînement si la précision de validation ne s'améliore pas
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    with mlflow.start_run():
        history = model.fit(datagen.flow(x_train_split, y_train_split, batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            callbacks=[early_stop, lr_reduction],
                            verbose=2)

        plt.figure()
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training and Validation Accuracy")
        plt.savefig("accuracy_curve.png")
        mlflow.log_artifact("accuracy_curve.png")

        plt.figure()
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.plot(history.history['loss'], label='Training Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.savefig("loss_curve.png")
        mlflow.log_artifact("loss_curve.png")

        # Évaluation du modèle sur les données de test
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = y_test.flatten()

        # Calcul des métriques : accuracy, precision, recall, AUC
        report = classification_report(y_test_classes, y_pred_classes, target_names=[str(i) for i in range(100)])
        print(report)

        # Binarise les labels pour calculer l'AUC
        y_test_binarized = label_binarize(y_test_classes, classes=range(100))
        auc = roc_auc_score(y_test_binarized, y_pred, average='macro', multi_class='ovr')
        print(f"AUC: {auc}")

        # Enregistre les métriques dans MLflow
        accuracy = np.mean(y_test_classes == y_pred_classes)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_auc", auc)
        mlflow.log_text(report, "classification_report.txt")

        # Trace les courbes AUC pour 10 classes dans un même graphe
        plt.figure(figsize=(10, 8))
        for i in range(10):
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred[:, i])
            plt.plot(fpr, tpr, label=f'Class {i} AUC = {roc_auc_score(y_test_binarized[:, i], y_pred[:, i]):.2f}')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for Classes 0 to 9")
        plt.legend(loc="lower right")
        plt.savefig("roc_curves_classes_0_to_9.png")
        mlflow.log_artifact("roc_curves_classes_0_to_9.png")

        # Enregistre le modèle avec un exemple d'entrée et des dépendances
        input_example = x_test[:1]
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            input_example=input_example,
            pip_requirements=["tensorflow==2.18.0", "cloudpickle==3.1.0"]
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-100.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size)
