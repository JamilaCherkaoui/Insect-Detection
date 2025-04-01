import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from model import Discriminator
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# === CONFIG === #10 000 good
CHECKPOINT = "/home/miashs2/StarGAN/ft_collemboles_projet/models/16000-D.ckpt"
IMAGE_PATH = "/home/miashs2/données/data/test/3/0.60408492170890440.94915661647088730.14426245282710803-3_3_3_3-TIDM_URBA_DIJON2021_2_3-0.580078125-0.593505859375-0.205078125-0.49560546875.jpg"
IMAGE_SIZE = 300
C_DIM = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === PRÉTRAITEMENT ===
transform = transforms.Compose([
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Chargement image
img_pil = Image.open(IMAGE_PATH).convert("RGB")
original_np = np.array(img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

# === MODÈLE ===
D = Discriminator(image_size=IMAGE_SIZE, conv_dim=64, c_dim=C_DIM, repeat_num=6).to(DEVICE)
D.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
D.eval()

# === LAYER DE STRIDE FIN ===
target_layers = [D.main[-4]]  # ou autre selon ta structure

# === WRAPPER pour CAM ===
class DiscriminatorWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, out_cls = self.model(x)
        return out_cls  # on veut appliquer GradCAM sur les classes (pas real/fake)

wrapped_D = DiscriminatorWrapper(D)

# === TARGET DE CLASSE ===
target_class = 3  # à adapter selon ce que tu veux voir
targets = [ClassifierOutputTarget(target_class)]

# === GRAD-CAM ===
with GradCAM(model=wrapped_D, target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

# === SUPERPOSITION HEATMAP ===
visualization = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
cv2.imwrite("gradcam_stride_fine_deuxtest.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

print("✅ Grad-CAM enregistrée : gradcam_stride_fine.jpg")