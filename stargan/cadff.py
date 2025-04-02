import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
from xplique.concepts import CraftTorch as Craft
from model import Discriminator

# ==== CONFIGURATION ====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = "/home/miashs2/StarGAN/ft_collemboles_projet/models/70000-D.ckpt"
DATA_DIR = "/home/miashs2/données/data/test"
IMAGE_SIZE = 300
C_DIM = 32  # nb classes de conditionnement
NUM_CLASSES = 9  # vraies classes biologiques
TARGET_CLASS = 0  # ex. classe pour CRAFT

# ==== TRANSFORM ====
transform = transforms.Compose([
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# ==== DATA ====
dataset = ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ==== MODÈLE ====
D = Discriminator(image_size=IMAGE_SIZE, conv_dim=64, c_dim=C_DIM, repeat_num=6).to(DEVICE)
D.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
D.eval()

# ==== DÉCOUPAGE DU MODÈLE POUR CRAFT ====
# g(x) = jusqu'à la couche latente
# h(z) = classification (output class)

# h est un wrapper qui prend les features et sort les logits de classe
class ClassifierHead(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.head = nn.Sequential(*list(D.main.children())[-4:])

    def forward(self, z):
        return self.head(z).reshape(z.size(0), -1)

import torch.nn.functional as F

class DiscriminatorG(nn.Module):  # input -> latent
    def __init__(self, D):
        super().__init__()
        self.features = nn.Sequential(*list(D.main.children())[:-4])

    def forward(self, x):
        z = self.features(x)
        return F.relu(z)  # 🔥 Ajoute un ReLU pour forcer les activations positives

g = DiscriminatorG(D)
h = ClassifierHead(D)

# ==== PRÉPARATION DES IMAGES POUR LA CLASSE CIBLE ====
images_selected = []

for img, label in tqdm(loader, desc="🔎 Sélection des images"):
    if label.item() == TARGET_CLASS:
        images_selected.append(img[0])  # img est un batch de 1
    if len(images_selected) >= 100:  # limite de taille
        break

images_tensor = torch.stack(images_selected).to(DEVICE)
print("✅ Images sélectionnées :", images_tensor.shape)

# ==== CRAFT ====
craft = Craft(
    input_to_latent_model=g,
    latent_to_logit_model=h,
    number_of_concepts=3,
    patch_size=100,
    batch_size=16,
    device=DEVICE
)

crops, crops_u, concept_bank_w = craft.fit(images_tensor, class_id=TARGET_CLASS)

print("✅ CRAFT terminé")
print("📦 crops:", crops.shape)
print("📦 crops_u (embedding):", crops_u.shape)
print("📦 concept_bank_w:", concept_bank_w.shape)

# === Sauvegarde des crops ===
os.makedirs("craft_crops", exist_ok=True)
for i, crop in enumerate(crops):
    crop_np = (np.transpose(crop, (1, 2, 0)) * 255).astype(np.uint8)
    Image.fromarray(crop_np).save(f"craft_crops/class{TARGET_CLASS}_crop{i}.png")

importances = craft.estimate_importance()
craft.plot_image_concepts(images_tensor[7].cpu())  # affiche
plt.savefig("craft_overlay_class3_img4.png")       # force la sauvegarde
craft.plot_concepts_importances()
plt.savefig("importanceconcept.png")  
plt.close()
print("✅ Crops sauvegardés dans craft_crops/")