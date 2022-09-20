import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files_A = sorted(glob.glob(os.path.join(root, mode) + "/fake/*.*"))

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)])

        img_A = self.transform(img_A)

        return {"A": img_A}

    def __len__(self):
        return len(self.files_A)
