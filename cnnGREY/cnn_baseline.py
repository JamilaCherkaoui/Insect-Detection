import torch
import torch.nn as nn
import torch.optim as optim
from generator_pytorch import ImageLabelDataset
from efficientnet_pytorch import EfficientNet
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torchvision

# Préparer liste fichiers
image_dir = "/home/miashs2/données/base_train_complete"
all_images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
train_files, val_files = train_test_split(all_images, test_size=0.2, random_state=42)

# Albumentations (transformations en grayscale)
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ElasticTransform(p=0.2),
    A.GridDistortion(p=0.2),
    A.OpticalDistortion(p=0.2),
    A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.5),
    A.ColorJitter(p=0.3),
    A.ToGray(p=1.0),  # Convertir les images en grayscale
    A.Normalize(),  # Optionnel, mais normalisation nécessaire
    ToTensorV2()
])

val_transform = A.Compose([
    A.ToGray(p=1.0),  # Convertir les images en grayscale
    A.Normalize(),
    ToTensorV2()
])

# Datasets & Loaders
train_dataset = ImageLabelDataset(image_dir, train_files, transform=train_transform)
val_dataset = ImageLabelDataset(image_dir, val_files, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)
NUM_CLASSES = 9

# Charger le modèle EfficientNet avec 1 canal en entrée (modifié pour le grayscale)
model = EfficientNet.from_pretrained('efficientnet-b3')


# Modifier la couche de sortie pour correspondre au nombre de classes
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)


all_labels = []
for _, labels in train_loader:
    all_labels.extend(labels.tolist())

class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(all_labels),
                                     y=all_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

from tqdm import tqdm
import copy

best_model = None
best_loss = float('inf')
patience = 20
patience_counter = 0

EPOCHS = 100

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # === Entraînement ===
    model.train()
    running_loss = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
       # print("Avant duplication:", images.shape)  # Afficher la forme des images

        # Si l'image est en grayscale avec 1 canal, dupliquer les canaux pour obtenir [batch_size, 3, 224, 224]
        #if images.shape[1] == 1:
        #    images = images.repeat(1, 3, 1, 1)

        # Vérifier à nouveau la forme des images après duplication
       # print("Après duplication:", images.shape)  # Afficher la forme des images après duplication

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    print(f"Train Loss: {train_loss:.4f}")

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Charger le meilleur modèle
model.load_state_dict(best_model)
torch.save(model.state_dict(), "efficientnet_baseline_grayscale.pth")
