import torch
from ultralytics import YOLO
import os
import cv2
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour charger les images et les étiquettes
def load_images_and_labels(folder_path):
    images = []
    labels = []
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            image_path = os.path.join(folder_path, file)
            label_path = os.path.join(folder_path, file[:-4] + '.txt')
            if os.path.isfile(label_path):
                images.append(cv2.imread(image_path))
                with open(label_path, 'r') as f:
                    labels.append(f.readlines())
    return images, labels

# Chargement du modèle YOLO
model = YOLO("train/weights/best.pt")

# Chargement des images et des étiquettes
test_folder = "datasets/val"
images, labels = load_images_and_labels(test_folder)

# Prédiction des objets dans chaque image
predictions = []
for image in images:
    prediction = model.predict(image)  # C'est une fonction fictive, vous devrez l'adapter à votre modèle YOLO spécifique
    predictions.append(prediction)

# Convertir les étiquettes en format compatible avec les prédictions
true_labels = []
for label in labels:
    true_labels.append([1 if len(l.split()) > 0 else 0 for l in label])

# Convertir les prédictions en format compatible avec les étiquettes
pred_labels = []
for prediction in predictions:
    pred_label = [1 if len(pred) > 0 else 0 for pred in prediction]
    pred_labels.append(pred_label)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(np.concatenate(true_labels), np.concatenate(pred_labels))

# Tracer la matrice de confusion
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de Confusion')
plt.colorbar()
plt.xlabel('Étiquettes Prédites')
plt.ylabel('Étiquettes Réelles')
plt.show()
