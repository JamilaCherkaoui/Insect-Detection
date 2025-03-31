from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torchvision


def extract_label_from_filename(filename):
    vote = filename.split('-')[1]
    label = int(vote.split('_')[0])
    return label

class ImageLabelDataset(Dataset):
    def __init__(self, image_dir, image_filenames, target_size=(224, 224), transform=None):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        label = extract_label_from_filename(filename)

        img_path = os.path.join(self.image_dir, filename)
        image = np.array(Image.open(img_path).convert("RGB").resize(self.target_size))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label