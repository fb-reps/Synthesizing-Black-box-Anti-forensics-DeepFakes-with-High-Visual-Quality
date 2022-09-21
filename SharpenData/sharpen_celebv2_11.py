############################
# add sharpen to images
###########################

import argparse
from datasets import *
import os
import torch
from torchvision.transforms import GaussianBlur
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="FFPP", help="name of the dataset")
parser.add_argument("--data_mode", type=str, default="valid", help="switch train data and test data")
parser.add_argument("--write_folder", type=str, default="fakeUSM_311", help="name of the input folder")
parser.add_argument("--read_folder", type=str, default="fake", help="name of the read folder")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start")
parser.add_argument("--n_epochs", type=int, default=1, help="epochs to stop")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--kernel_size", type=int, default=3, help="kernel size to gaussian blur")
parser.add_argument("--sigma", type=float, default=1, help="sigma to gaussian blur")
parser.add_argument("--amount", type=float, default=1, help="amount to gaussian blur")
opt = parser.parse_args()
print(opt)
os.makedirs("../DFData/%s/%s/%s" % (opt.dataset_name, opt.data_mode, opt.write_folder), exist_ok=True)


gaussian_blur = GaussianBlur(opt.kernel_size, opt.sigma)

if __name__ == '__main__':
    path_A = "../DFData/%s/%s/%s/" % (opt.dataset_name, opt.data_mode, opt.read_folder)
    A_files = os.listdir(path_A)
    for A_file in A_files:
        image_path = os.path.join(path_A, A_file)
        f_open = Image.open(image_path)
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(f_open)
        gaussian_blur_image = gaussian_blur(img)
        sub_image = img - gaussian_blur_image
        sharpen_image = img + sub_image * opt.amount
        save_image(sharpen_image,
                   "../DFData/%s/%s/%s/%s" % (opt.dataset_name, opt.data_mode, opt.write_folder, A_file))
