import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from model import Generator  # Ton modèle
import os

# ==== CONFIG ====
MODEL_PATH = '/home/miashs2/StarGAN/ft_collemboles_projet/models/4000-G.ckpt'
DATA_DIR = '/home/miashs2/données/data/test'  # Répertoire de test
C_DIM = 9
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

print("Classes trouvées :", dataset.classes)
print("Index associé à chaque classe :", dataset.class_to_idx)

# ==== MODEL ====
class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.main.children())[:-1])  # Enlève dernière couche

    def forward(self, x, c):
        x = torch.cat([x, c.view(c.size(0), c.size(1), 1, 1).expand(-1, -1, IMAGE_SIZE, IMAGE_SIZE)], dim=1)
        return self.features(x).view(x.size(0), -1)  # Flatten

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
        one_hot = torch.zeros(lbls.size(0), C_DIM).to(DEVICE)
        one_hot[torch.arange(lbls.size(0)), lbls] = 1
        feats = extractor(imgs, one_hot)
        features.append(feats.cpu())
        labels.append(lbls.cpu())

features = torch.cat(features).numpy()
labels = torch.cat(labels).numpy()

# ==== T-SNE ====
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(features)

# ==== PLOT ====
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=20)
plt.colorbar(scatter, ticks=range(C_DIM), label="Classes")
plt.title('t-SNE des représentations latentes de StarGAN')
plt.tight_layout()
plt.savefig('tsne_latents_stargan_70000.png')
plt.show()