import argparse
import time
import datetime
import sys

import torchvision
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets_attention import *

import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

writer = SummaryWriter(comment="_MVUNet")

parser = argparse.ArgumentParser()
parser.add_argument("--generator_path", type=str, default='/home/ncubigdata1/Documents/fanbing_documents_own/SharpenPix2pix_celeb_F_RU311/saved_models/Celeb_DF_v2/generator_190.pth', help="")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="Celeb_DF_v2", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image_path_temp height")
parser.add_argument("--img_width", type=int, default=256, help="size of image_path_temp width")
parser.add_argument("--channels", type=int, default=3, help="number of image_path_temp channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between downsample checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixel_wise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image_path_temp and real image_path_temp
lambda_pixel = 100

# Calculate output of image_path_temp discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize visual_optimizer and discriminator
visual_optimizer = VisualOptimizer()
discriminator2 = Discriminator2(input_shape)


if cuda:
    visual_optimizer = visual_optimizer.cuda()
    discriminator2 = discriminator2.cuda()
    criterion_GAN.cuda()
    criterion_pixel_wise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    visual_optimizer.load_state_dict(
        torch.load("saved_models/%s/visual_optimizer_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator2.load_state_dict(
        torch.load("saved_models/%s/discriminator2_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    visual_optimizer.apply(weights_init_normal)
    discriminator2.apply(weights_init_normal)
    visual_optimizer.generator.load_state_dict(torch.load(opt.generator_path))

# Optimizers Adam include
optimizer_G = torch.optim.Adam(visual_optimizer.MV_UNet.parameters(), lr=opt.lr*2, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [
    transforms.CenterCrop((opt.img_height, opt.img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("../DFData/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset("../DFData/%s" % opt.dataset_name, transforms_=transforms_, mode="valid"),
    batch_size=10,
    shuffle=True,
    num_workers=1,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    images = next(iter(val_dataloader))
    fake_A = Variable(images["A"].type(Tensor))
    fake_B = visual_optimizer(fake_A)
    img_sample = torch.cat((fake_A.data, fake_B.data), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

    img_grid = torchvision.utils.make_grid(img_sample.mul(0.5).add_(0.5), nrow=5, range=(0, 255))
    writer.add_image('fake_A fake_B', img_grid, batches_done)


# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs
        fake_A = Variable(batch["A"].type(Tensor))
        fakeUSM_B = Variable(batch["B"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((fake_A.size(0), *discriminator2.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((fake_A.size(0), *discriminator2.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Pixel-wise loss
        generate_B = visual_optimizer(fake_A)
        loss_pixel = criterion_pixel_wise(generate_B, fakeUSM_B)

        # GAN loss
        score_fake = discriminator2(generate_B)
        loss_GAN = criterion_GAN(score_fake, valid)

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator2(fakeUSM_B)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        score_fake = discriminator2(generate_B.detach())
        loss_fake = criterion_GAN(score_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        writer.add_scalar('loss/D', loss_D.item(), batches_done)
        writer.add_scalar('loss/G', loss_G.item(), batches_done)
        writer.add_scalar('loss/loss_pixel', loss_pixel.item(), batches_done)
        writer.add_scalar('loss/loss_GAN', loss_GAN.item(), batches_done)
        # writer.add_graph(visual_optimizer, fake_A)

        # If at sample interval save image_path_temp
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save downsample checkpoints
        torch.save(visual_optimizer.state_dict(), "saved_models/%s/visual_optimizer_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator2.state_dict(), "saved_models/%s/discriminator2_%d.pth" % (opt.dataset_name, epoch))
