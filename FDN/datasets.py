import glob
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files_A = sorted(glob.glob(os.path.join(root, mode) + "/fake/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, mode) + "/realUSM_311/*.*"))

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_B = Image.open(self.files_B[index % len(self.files_B)])

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files_A)
