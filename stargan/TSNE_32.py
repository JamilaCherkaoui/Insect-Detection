import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from model import Generator
import os

# ==== CONFIG ====
MODEL_PATH = '/home/miashs2/StarGAN/ft_collemboles_projet/models/20000-G.ckpt'
DATA_DIR = '/home/miashs2/données/data/test'  # Dossier avec les 9 classes (0-8)
C_DIM = 32            # Nombre total de classes d'entraînement (projets)
CLASS_DIM = 9         # Nombre de classes biologiques (0 à 8)
IMAGE_SIZE = 300
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== TRANSFORM ====
transform = transforms.Compose([
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ==== DATA ====
dataset = ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print("📁 Classes trouvées :", dataset.classes)
print("🔢 Mapping class_to_idx :", dataset.class_to_idx)

# ==== MODEL ====
class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.main.children())[:-1])

    def forward(self, x, c):
        # Expand et concatène les labels
        c_expanded = c.view(c.size(0), c.size(1), 1, 1).expand(-1, -1, IMAGE_SIZE, IMAGE_SIZE)
        x = torch.cat([x, c_expanded], dim=1)
        return self.features(x).view(x.size(0), -1)

# Charger le générateur entraîné sur 32 classes (projets)
G = Generator(conv_dim=64, c_dim=C_DIM, repeat_num=6).to(DEVICE)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
G.eval()

extractor = FeatureExtractor(G).to(DEVICE)

# ==== EXTRACTION ====
features = []
labels = []

with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE)
        lbls = lbls.to(DEVICE)

        # One-hot encoding sur les 9 premières dimensions
        one_hot = torch.zeros(lbls.size(0), C_DIM).to(DEVICE)
        one_hot[:, :CLASS_DIM].scatter_(1, lbls.view(-1, 1), 1)

        feats = extractor(imgs, one_hot)
        features.append(feats.cpu())
        labels.append(lbls.cpu())

features = torch.cat(features).numpy()
labels = torch.cat(labels).numpy()

# ==== T-SNE ====
print("⏳ t-SNE en cours...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(features)

# ==== PLOT ====
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=20)
plt.colorbar(scatter, ticks=range(CLASS_DIM), label="Classe biologique")
plt.title('t-SNE des représentations latentes (StarGAN entraîné sur projets, testé sur classes)')
plt.tight_layout()
plt.savefig('tsne_latents_stargan_32cond_vs_9class_20000.png')
plt.show()

print("✅ t-SNE terminé et sauvegardé sous tsne_latents_stargan_32cond_vs_9class.png")