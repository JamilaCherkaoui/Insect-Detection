import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import Generator
import numpy as np
from tqdm import tqdm
import os

# ==== CONFIG ====
MODEL_PATH = '/home/miashs2/StarGAN/ft_collemboles_projet/models/20000-G.ckpt'
DATA_DIR = '/home/miashs2/données/data/test'  # <- dossier avec classes biologiques 0-8
C_DIM = 32  # Car le modèle a été entraîné avec 32 projets
IMAGE_SIZE = 300
BATCH_SIZE = 8
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

# ==== MODEL ====
G = Generator(conv_dim=64, c_dim=C_DIM, repeat_num=6).to(DEVICE)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
G.eval()

# ==== DISTANCES ====
all_l1 = []
all_labels = []

with torch.no_grad():
    for imgs, labels in tqdm(loader, desc="💡 Computing L1 distances"):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        batch_l1 = []

        for target_class in range(9):  # Seulement les 9 classes bio
            target = torch.full((imgs.size(0),), target_class, dtype=torch.long, device=DEVICE)
            c_trg = torch.zeros(imgs.size(0), C_DIM).to(DEVICE)
            c_trg[torch.arange(imgs.size(0)), target] = 1  # Positionner le 1 uniquement dans les 9 premières

            translated = G(imgs, c_trg)
            l1 = F.l1_loss(translated, imgs, reduction='none')  # (B, 3, H, W)
            l1 = l1.view(imgs.size(0), -1).mean(dim=1)  # Moyenne par image
            batch_l1.append(l1.cpu().numpy())

        batch_l1 = np.stack(batch_l1, axis=1)  # (B, 9)
        all_l1.append(batch_l1)
        all_labels.append(labels.cpu().numpy())

# ==== SAVE ====
l1_matrix = np.concatenate(all_l1, axis=0)
true_labels = np.concatenate(all_labels, axis=0)

np.save("l1_distances.npy", l1_matrix)
np.save("true_labels.npy", true_labels)

print("✅ Sauvegardé : l1_distances.npy & true_labels.npy")