# coding: utf-8

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from IPython.display import HTML
from torch import optim
import torch.nn as nn
import torch

args = {
    "epoch": 1,
    "n_critic": 10,
    "sample_interval": 5,
    "fine_size": 64,
    "batch_size": 64,
    "lr": 0.0001,
    "checkpoint": "checkpoint",
    "img_dir": "./faces",
    # "data_post_dir": "./post_opc",
    "img_output_dir": "./fake",
    "checkpoint_dir": "./checkpoint",
    "img_csv": "./small_img_name.csv",
    "channel": 3,
    "latent": 100,
    "beta": 0.5,
    "gpu": 1,
    "load": True,
    "clip": 0.01
}


# define a dataset sub class
class myDataset(Dataset):
    def __init__(self, img_dir, img_csv, transform=None):
        self.img_dir = img_dir
        self.img_csv = pd.read_csv(img_csv)
        self.transform = transform

    def __len__(self):
        return len(self.img_csv)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_csv.iloc[idx, 0])
        image = cv2.imread(img_name)  # 96 96 3
        if self.transform:
            image = self.transform(image)
        return image


class Rescale(object):
    """Rescale the image in a sample to a given size."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        # H x W x C
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


class ToTensor(object):
    """Convert ndarrys in image to Tensors"""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        # image = torch.from_numpy(image)
        return torch.from_numpy(image).type(torch.FloatTensor)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, gpu):
        super(Generator, self).__init__()
        self.gpu = gpu
        self.nz = 100
        self.ngf = 64
        self.main = nn.Sequential(
            # input is latent vector, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*10) x 3 x 3
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 6 x 6
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*6) * 12 * 12
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf*2) * 48 * 48
            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, gpu):
        super(Discriminator, self).__init__()
        self.gpu = gpu
        self.nc = 3
        self.ndf = 64
        self.main = nn.Sequential(
            # input nc x 96 x 96
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # input ndf x 48 x 48
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input nc x 24 x 24
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # input nc x 12 x 12
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # input nc x 6 x 6
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    img_dataset = myDataset(img_dir=args["img_dir"],
                            img_csv=args["img_csv"],
                            transform=transforms.Compose([
                                Rescale(64),
                                ToTensor(),  # convert H W C to C H W in range[0.0, 1.0]
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
    # cv show images
    # for i in range(5):
    #     sample = img_dataset[i]
    #     cv2.imshow('we', np.uint8(sample.numpy()))
    #     cv2.waitKey

    # check sample data
    # for i in range(len(img_dataset)):
    #     sample = img_dataset[i]
    #     print(sample)
    #     if i == 3:
    #         break
    dataloader = DataLoader(img_dataset, batch_size=args["batch_size"],
                            shuffle=True, num_workers=2)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and args["gpu"] > 0) else "cpu")
    # iter function for generating iterator
    # real_batch = next(iter(dataloader))
    # show a batch of images
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("training images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=2, normalize=True).cpu(),
    #                         (1,2,0)))
    # plt.show()

    # to(device) move nn to the device
    netG = Generator(args["gpu"]).to(device)
    netG.apply(weights_init)
    print(netG)
    netD = Discriminator(args["gpu"]).to(device)
    netD.apply(weights_init)
    print(netD)
    #    if args["load"] == True:
    #        netG.load_state_dict(torch.load(os.path.join(args["checkpoint_dir"],"generator.pkl")))
    #        netD.load_state_dict(torch.load(os.path.join(args["checkpoint_dir"],"discriminator.pkl")))
    #        netG.eval()
    #        netD.eval()

    if (device.type == "cuda") and (args["gpu"] > 1):
        netD = nn.DataParallel(netD, list(range(args["gpu"])))

    # criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    # optimizerD = optim.Adam(netD.parameters(), lr=args["lr"], betas=(args["beta"], 0.999))
    # optimizerG = optim.Adam(netG.parameters(), lr=args["lr"], betas=(args["beta"], 0.999))
    optimizerD = optim.RMSprop(netD.parameters(), lr=args["lr"], alpha=0.9)
    optimizerG = optim.RMSprop(netG.parameters(), lr=args["lr"], alpha=0.9)

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    print("Starting training Loop...")
    for epoch in range(args["epoch"]):
        # enumerate(iteration, start_index)
        for i, data in enumerate(dataloader, 0):
            # update D maximize D(x) - D(G(z))
            netD.zero_grad()
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            # forward pass real batch through D
            output = netD(real_cpu)
            output = output.view(-1)
            # calculate loss on all-real batch
            errD_real = -output.mean()
            # calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
            # get tensor input of (b_size, 100, 1, 1)
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake = netG(noise)
            output = netD(fake.detach()).view(-1)
            errD_fake = output.mean()
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real - errD_fake
            optimizerD.step()
            for p in netD.parameters():
                p.data.clamp_(-args["clip"], args["clip"])
            # update G: maximize D(G(z))
            if i % args["n_critic"] == 0:
                netG.zero_grad()
                output = netD(netG(noise)).view(-1)
                errG = -output.mean()
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            if i % 100 == 0:
                print('[%d/%d] [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args["epoch"], i, len(dataloader), errD.item(),
                         errG.item(), D_x, D_G_z1, D_G_z2))

            # if ((iters % 199 == 0) and epoch % 5 == 0) or ((epoch == args["epoch"] - 1) and (i == len(dataloader)-1)):
            if (epoch == args["epoch"] - 1) and (i == len(dataloader) - 2):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    torch.save(netG.state_dict(), os.path.join(args["checkpoint_dir"], "generator_wgan.pkl"))
    torch.save(netD.state_dict(), os.path.join(args["checkpoint_dir"], "discriminator_wgan.pkl"))

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label='G')
    plt.plot(D_losses, label='D')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    #    plt.show()
    plt.savefig(os.path.join(args["img_output_dir"], 'g_d_loss_wgan.png'))

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow((np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu(),
                             (1, 2, 0)) * 255).type(torch.IntTensor))
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow((np.transpose(img_list[0], (1, 2, 0)) * 255).type(torch.IntTensor))
    # plt.show()
    plt.savefig(os.path.join(args["img_output_dir"], 'real_fake_wgan.png'))

#    fig = plt.figure(figsize=(8, 8))
#    plt.axis("off")
#    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
#    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=True)
#    HTML(ani.to_jshtml())
#    Writer = animation.writers['ffmpeg']
#    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
#    ani.save("im.mp4", writer=writer)

