import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets_generate import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment="_pix2pix")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=190, help="epoch to start training from")
parser.add_argument("--dataset_name", type=str, default="Celeb_DF_v2", help="name of the dataset")
parser.add_argument("--n_cpu", type=int, default=64, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image_path_temp height")
parser.add_argument("--img_width", type=int, default=256, help="size of image_path_temp width")
opt = parser.parse_args()
print(opt)
output_dir = "fakeGenerateSharpen_G%s" % opt.epoch
os.makedirs("images/%s" % output_dir, exist_ok=True)
cuda = True if torch.cuda.is_available() else False

# Initialize generator
generator = GeneratorUNet()
if cuda:
    generator = generator.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    assert "no model wight"

# Configure dataloaders
transforms_ = [
    transforms.CenterCrop((opt.img_height, opt.img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

test_dataloader = DataLoader(
    ImageDataset("../DFData/%s" % opt.dataset_name, transforms_=transforms_, mode="test"),
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.epoch + 1):
    for i, batch in enumerate(test_dataloader):
        pristine_A = Variable(batch["A"].type(Tensor))
        batches_done = (opt.epoch - epoch) * len(test_dataloader) + i
        fake_B = generator(pristine_A)
        save_image(fake_B, "images/%s/%s.png" % (output_dir, batches_done), normalize=True)
