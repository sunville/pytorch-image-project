# coding: utf-8

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
from torch import optim
import torch.nn as nn
import torch

args = {
    "epoch": 3000,
    "n_critic": 5,
    "save_step": 5000,
    "fine_size": 256,
    "L1_lambda": 50,
    "batch_size": 16,
    "lr": 0.0001,
    "checkpoint": "checkpoint",
    "post_dir": "./opc/post_opc",
    "pre_dir":"./opc/pre_opc",
    "img_output_dir": "./opc/plot",
    "checkpoint_dir": "./opc/checkpoint",
    "img_csv": "./opc/opc_name.csv",
    "beta": 0.5,
    "gpu": 1,
    "load": False,
    "leak": 0.1,
    "cuda":"cuda:0"
}


# define a dataset sub class
class myDataset(Dataset):
    def __init__(self, pre_dir, post_dir, img_csv, transform=None):
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.img_csv = pd.read_csv(img_csv)
        self.transform = transform

    def __len__(self):
        return len(self.img_csv)

    def __getitem__(self, idx):
        img_name = self.img_csv.iloc[idx, 0]
        pre_img_name = os.path.join(self.pre_dir, img_name)
        post_img_name = os.path.join(self.post_dir, img_name)
        
        pre_image = cv2.imread(pre_img_name, 0)  # 2048 2048
        pre_image = cv2.resize(pre_image, (256, 256), interpolation=cv2.INTER_AREA)
        
        post_image = cv2.imread(post_img_name, 0) # 2048 2048
        post_image = cv2.resize(post_image, (256, 256), interpolation=cv2.INTER_AREA)
        post_image = post_image / 255 # 256 256
        pre_image = pre_image / 255 # 256 256
        pre_image = pre_image[:, :, np.newaxis] # 256 256 1
        post_image = post_image[:, :, np.newaxis] # 256 256 1
        image = np.concatenate((pre_image, post_image), axis=2)# 256 256 2
        if self.transform:
            image = self.transform(image)
        return image


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


def save_img(img_list, epoch):
        # convert fake image from (b_size, 1, H, W) to (b_size, H, W, 1)
        cv_image = np.transpose(img_list[-1], (0, 2, 3, 1))

        # convert image to (b_size, H, W)
        cv_image = np.squeeze(cv_image, axis=3)
        
        for i in range(args["batch_size"]):
            image = cv_image[i,:,:] * 255
            filename = 'real_fake_gan_opc_' + str(epoch) + "_" + str(i) + '.png'
            filename = os.path.join(args["img_output_dir"], filename)
            cv2.imwrite(filename, image)


def plot_data(G_losses, D_losses, epoch):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label='G')
    plt.plot(D_losses, label='D')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt_name = 'g_d_loss_gan_opc_'+str(epoch)+'.png'
    plt.savefig(os.path.join(args["img_output_dir"], plt_name))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngf = 256
        self.dim = 8
        self.encoder = nn.Sequential(
            # input is image (b_size, 1, 256, 256)
            nn.Conv2d(1, self.dim*2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.dim * 2),
            nn.ReLU(True),
            # (b_size, 16, 128, 128)

            nn.Conv2d(self.dim*2, self.dim*8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.dim * 8),
            nn.ReLU(True),
            # (b_size, 64, 64, 64)
            
            nn.Conv2d(self.dim*8, self.dim*16, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.dim * 16),
            nn.ReLU(True),
            # (b_size, 128, 32, 32)

            nn.Conv2d(self.dim*16, self.dim*64, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.dim * 64),
            nn.ReLU(True),
            # (b_size, 512, 16, 16)

            nn.Conv2d(self.dim*64, self.dim*128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(self.dim * 128),
            nn.ReLU(True),
            # (b_size, 1024, 8, 8)
        )


        self.decoder = nn.Sequential(
            # (b_size, 1024, 8, 8)
            nn.ConvTranspose2d(self.dim*128, self.dim*64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim * 64),
            nn.ReLU(True),
            # (b_size, 512, 16, 16)

            nn.ConvTranspose2d(self.dim*64, self.dim*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim * 16),
            nn.ReLU(True),
            # (b_size, 128, 32, 32)

            nn.ConvTranspose2d(self.dim*16, self.dim*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim * 8),
            nn.ReLU(True),
            # (b_size, 64, 64, 64)

            nn.ConvTranspose2d(self.dim*8, self.dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim * 2),
            nn.ReLU(True),
            # (b_size, 16, 128, 128)
            nn.ConvTranspose2d(self.dim*2, 1, 4, 2, 1, bias=False),
            nn.Sigmoid() # range in (0.0~1.0)
            # (b_size, 1, 256, 256)
        )


    def forward(self, net):
        net = self.encoder(net)
        net = self.decoder(net)
        return net


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ngf = 256
        self.dim = 8
        self.Conv = nn.Sequential(
            # (b_size, 2, 256, 256)
            nn.Conv2d(2, self.dim*8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 64, 256, 256)
            nn.Conv2d(self.dim*8, self.dim*8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 64, 256, 256)

            nn.Conv2d(self.dim*8, self.dim*8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 64, 128, 128)

            nn.Conv2d(self.dim*8, self.dim*16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 128, 128, 128)
            nn.Conv2d(self.dim*16, self.dim*16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 128, 128, 128)

            nn.Conv2d(self.dim*16, self.dim*16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 128, 64, 64)

            nn.Conv2d(self.dim*16, self.dim*32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 256, 64, 64)
            nn.Conv2d(self.dim*32, self.dim*32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 256, 64, 64)
            nn.Conv2d(self.dim*32, self.dim*32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 256, 64, 64)

            nn.Conv2d(self.dim*32, self.dim*32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 256, 32, 32)

            nn.Conv2d(self.dim*32, self.dim*64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 512, 32, 32)
            nn.Conv2d(self.dim*64, self.dim*64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 512, 32, 32)
            nn.Conv2d(self.dim*64, self.dim*64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 512, 32, 32)

            nn.Conv2d(self.dim*64, self.dim*64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 512, 16, 16)

            nn.Conv2d(self.dim*64, self.dim*64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 512, 16, 16)
            nn.Conv2d(self.dim*64, self.dim*64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 512, 16, 16)
            nn.Conv2d(self.dim*64, self.dim*64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 512, 16, 16)

            nn.Conv2d(self.dim*64, self.dim*64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # (b_size, 512* 8 *8)
        )

        self.FullConnected = nn.Sequential(
            # (b_size, 512* 8 *8)

            nn.Linear(512*8*8, 2048),
            nn.Dropout(0.5),
            nn.ReLU(True),

            nn.Linear(2048, 512),
            nn.Dropout(0.5),
            nn.ReLU(True),

            nn.Linear(512, 1),
            nn.Sigmoid()
            # (b_size, 1)
        )

    def forward(self, net):
        net = self.Conv(net)
        net = net.view(net.size(0), -1)
        net = self.FullConnected(net)
        return net


if __name__ == '__main__':
    img_dataset = myDataset(pre_dir=args["pre_dir"],
                            post_dir = args["post_dir"],
                            img_csv=args["img_csv"],
                            transform=transforms.Compose([
                                ToTensor(),  # convert H W C to C H W in range[0.0, 1.0]
                            ]))
    dataloader = DataLoader(img_dataset, batch_size=args["batch_size"],
                            shuffle=True, num_workers=2)

    device = torch.device(args["cuda"] if (torch.cuda.is_available() and args["gpu"] > 0) else "cpu")

    netG = Generator()
    print(netG)
    netD = Discriminator()
    print(netD)
    checkpoint = torch.load(os.path.join(args["checkpoint_dir"],"generator_gan_opc.pkl"),
                            map_location = lambda storage, loc:storage)
    netG.load_state_dict(checkpoint)
    netG.to(device)
    del checkpoint
    checkpoint = torch.load(os.path.join(args["checkpoint_dir"],"discriminator_gan_opc.pkl"),
                            map_location = lambda storage, loc:storage)
    netD.load_state_dict(checkpoint)
    del checkpoint
    netD.to(device)

    if (device.type == "cuda") and (args["gpu"] > 1):
        netD = nn.DataParallel(netD, list(range(args["gpu"])))

    optimizerD = optim.RMSprop(netD.parameters(), lr=args["lr"], alpha=0.9)
    optimizerG = optim.RMSprop(netG.parameters(), lr=args["lr"], alpha=0.9)
    criterion_gan = nn.BCELoss()
    criterion_mse = nn.MSELoss()
    real_label = 1.0
    fake_label = 0.0

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    print("Starting training Loop...")
    for epoch in range(args["epoch"]):
        # enumerate(iteration, start_index)
        for i, data in enumerate(dataloader, 0):
            if data.size(0) < args["batch_size"]:
                break
            # update D maximize log(D(x)) + log(1-D(G(z)))
            netD.zero_grad()

            # data:(b_size, 2, H, W) 
            pre_post_image = data.to(device)
            b_size = pre_post_image.size(0)
            label = torch.full((b_size,), real_label, device=device)

            # get pre_image and post_image of (b_size, 1, H, W)
            pre_image, post_image = torch.split(pre_post_image,1,dim=1)

            # forward pass real batch through D
            output = netD(pre_post_image).view(-1)

            # calculate loss on all-real batch
            errD_real = criterion_gan(output, label)

            # calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # get fake image of (b_size, 1, H, W)
            fake_image = netG(pre_image)

            # get pre_fake_image of (b_size, 2, H, W)
            pre_fake_image = torch.cat((pre_image, fake_image), dim=1)
            label.fill_(fake_label)

            output = netD(pre_fake_image.detach()).view(-1)

            errD_fake = criterion_gan(output, label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()
            errD = errD_fake + errD_real
            
            optimizerD.step()

            if i % args["n_critic"] == 0:
                # update G: maximize log(D(G(z)))
                netG.zero_grad()
                label.fill_(real_label)
                fake_image = netG(pre_image)
                pre_fake_image = torch.cat((pre_image, fake_image), dim=1)
                output = netD(pre_fake_image).view(-1)
                errG_gan = criterion_gan(output, label)
                errG_loss_L2 = args["L1_lambda"]*criterion_mse(fake_image, post_image)
                errG = errG_gan + errG_loss_L2
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()
                if i % (args["n_critic"] * 16) == 0:
                    G_losses.append(errG.item())
                    D_losses.append(errD.item())

            if i % 20 == 0:
                print('[%d/%d] [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args["epoch"], i, len(dataloader), errD.item(),
                         -errG.item(), D_x, D_G_z1, D_G_z2))

            # if ((iters % 199 == 0) and epoch % 5 == 0) or ((epoch == args["epoch"] - 1) and (i == len(dataloader)-1)):
            if iters % (500*len(dataloader)) == 0 or ((epoch == args["epoch"] - 1) and (i == len(dataloader)-2)):
                with torch.no_grad():
                    fake = netG(fake_image).detach().cpu()
                    img_list.append(fake.numpy())
                    save_img(img_list, epoch)
                    plot_data(G_losses, D_losses, epoch)
                    torch.save(netG.state_dict(), os.path.join(args["checkpoint_dir"], "generator_gan_opc.pkl"))
                    torch.save(netD.state_dict(), os.path.join(args["checkpoint_dir"], "discriminator_gan_opc.pkl"))
            iters = iters+1




