import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from efficientnet_pytorch import EfficientNet

import albumentations as A
from albumentations.pytorch import ToTensorV2

# --------------------------------------------------
# 📦 Dataset pour test (pas de labels dans les noms)
# --------------------------------------------------
class ImageDatasetWithoutLabels(Dataset):
    def __init__(self, image_dir, image_filenames, transform=None):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = np.array(Image.open(img_path).convert("RGB").resize((300, 300)))
        #image = np.array(Image.open(img_path).convert("L").resize((300, 300)))


        if self.transform:
            image = self.transform(image=image)["image"]

        return image, filename

# --------------------------------------------------
# 🔧 Préparation
# --------------------------------------------------
image_dir = "/home/miashs2/données/datatest"
test_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

val_transform = A.Compose([
    A.ToGray(p=1.0),  # Convertir les images en grayscale
    A.Normalize(),
    ToTensorV2()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Device:", device)

# --------------------------------------------------
# 🔍 Chargement du modèle entraîné
# --------------------------------------------------
model = EfficientNet.from_name('efficientnet-b3')
model._fc = nn.Linear(model._fc.in_features, 9)  # 9 classes
model.load_state_dict(torch.load("efficientnet_baseline_grayscale.pth"))
model.eval()
model.to(device)
print("✅ Modèle chargé")

# --------------------------------------------------
# 📤 Dataset et DataLoader
# --------------------------------------------------
test_dataset = ImageDatasetWithoutLabels(image_dir, test_files, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("✅ Test loader prêt")

# --------------------------------------------------
# 🧠 Inférence + création de la soumission
# --------------------------------------------------
submission_rows = []

with torch.no_grad():
    for images, filenames in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        for filename, pred in zip(filenames, preds):
            img_id = os.path.splitext(filename)[0]  # enlever .jpg
            submission_rows.append((img_id, pred))

# --------------------------------------------------
# 💾 Export CSV
# --------------------------------------------------
submission_df = pd.DataFrame(submission_rows, columns=["idx", "gt"])
submission_df.to_csv("submission_cnn__grey.csv", index=False)
print("✅ Fichier de soumission généré : submission.csv")