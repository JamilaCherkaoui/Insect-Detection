import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import Generator  # Assure-toi que le chemin est correct
from tqdm import tqdm

# ==== CONFIGURATION ====
MODEL_PATH = '/home/miashs2/StarGAN/ft_collemboles_projet/models/20000-G.ckpt'
DATA_DIR = '/home/miashs2/données/datatest'
C_DIM_GAN = 32  # Le GAN est entraîné sur 32 projets
NUM_CLASSES = 9  # On veut faire l'inférence sur les 9 classes biologiques
IMAGE_SIZE = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== TRANSFORM ====
transform = transforms.Compose([
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ==== CHARGEMENT DU MODÈLE ====
G = Generator(conv_dim=64, c_dim=C_DIM_GAN, repeat_num=6).to(DEVICE)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
G.eval()

# ==== RÉCUPÉRATION DES IMAGES ====
image_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')])
print(f"🖼️ {len(image_files)} images trouvées dans {DATA_DIR}")

l1_distances = []

# ==== CALCUL DES DISTANCES L1 ====
with torch.no_grad():
    for fname in tqdm(image_files, desc="🔁 Traitement des images"):
        img_path = os.path.join(DATA_DIR, fname)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        distances = []
        for target_class in range(NUM_CLASSES):
            target_label = torch.zeros(1, C_DIM_GAN).to(DEVICE)
            target_label[0, target_class] = 1  # Conditionne uniquement sur les 9 premières

            fake_img = G(img_tensor, target_label)
            l1 = torch.mean(torch.abs(img_tensor - fake_img)).item()
            distances.append(l1)

        l1_distances.append(distances)

# ==== SAUVEGARDE ====
l1_array = np.array(l1_distances)
np.save("l1_distances_test.npy", l1_array)
print(f"✅ Distances L1 sauvegardées sous l1_distances_test.npy | shape = {l1_array.shape}")