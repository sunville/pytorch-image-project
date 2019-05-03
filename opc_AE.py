# coding: utf-8

import os
import sys
sys.path.append(os.getcwd()) 
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
    "epoch": 2000,
    "n_critic": 5,
    "fine_size": 256,
    "L1_lambda": 100,
    "batch_size": 16,
    "lr": 0.0001,
    "post_dir": "./opc/post_opc",
    "pre_dir":"./opc/pre_opc",
    "img_output_dir": "./opc/plot/AE",
    "checkpoint_dir": "./opc/checkpoint/AE",
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
            filename = 'fake_AE_opc_' + str(epoch+2600) +'_'+ str(i) + '.png'
            filename = os.path.join(args["img_output_dir"], filename)
            cv2.imwrite(filename, image)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
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


    def forward(self, net):
        net = self.encoder(net)
        return net


class DecoderPre(nn.Module):
    def __init__(self):
        super(DecoderPre, self).__init__()
        self.ngf = 256
        self.dim = 8

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
        net = self.decoder(net)
        return net


class DecoderPost(nn.Module):
    def __init__(self):
        super(DecoderPost, self).__init__()
        self.ngf = 256
        self.dim = 8

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
        net = self.decoder(net)
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

    device = torch.device("cuda" if (torch.cuda.is_available() and args["gpu"] > 0) else "cpu")

    netEncoder = Encoder()
    print(netEncoder)
    netDecoderPre = DecoderPre()
    print(netDecoderPre)
    netDecoderPost = DecoderPost()
    print(netDecoderPost)
    checkpoint = torch.load(os.path.join(args["checkpoint_dir"],"encoder_AE_opc.pkl"),
                            map_location = lambda storage, loc:storage)
    netEncoder.load_state_dict(checkpoint)
    netEncoder.to(device)
    del checkpoint
    checkpoint = torch.load(os.path.join(args["checkpoint_dir"],"decoder_pre_AE_opc.pkl"),
                            map_location = lambda storage, loc:storage)
    netDecoderPre.load_state_dict(checkpoint)
    del checkpoint
    netDecoderPre.to(device)
    checkpoint = torch.load(os.path.join(args["checkpoint_dir"],"decoder_post_AE_opc.pkl"),
                            map_location = lambda storage, loc:storage)
    netDecoderPost.load_state_dict(checkpoint)
    del checkpoint
    netDecoderPost.to(device)
    
    optimizerEncoder = optim.Adam(netEncoder.parameters(), lr=args["lr"])
    optimizerDecoderPre = optim.Adam(netDecoderPre.parameters(), lr=args["lr"])
    optimizerDecoderPost = optim.Adam(netDecoderPost.parameters(), lr=args["lr"])
    criterion_mae = nn.MSELoss()

    img_list = []
    pre_losses = []
    post_losses = []
    iters = 0
    print("Starting training Loop...")
    for epoch in range(args["epoch"]):
        # enumerate(iteration, start_index)
        for i, data in enumerate(dataloader, 0):
            if data.size(0) < args["batch_size"]:
                break
            netEncoder.zero_grad()
            netDecoderPre.zero_grad()
            netDecoderPost.zero_grad()
            # data:(b_size, 2, H, W) 
            pre_post_image = data.to(device)

            # get pre_image and post_image of (b_size, 1, H, W)
            pre_image, post_image = torch.split(pre_post_image,1,dim=1)
 
            pre_feature_map = netEncoder(pre_image)
            pre_reconstruction = netDecoderPre(pre_feature_map)
            errPre = args["L1_lambda"] * criterion_mae(pre_reconstruction, pre_image)
            errPre.backward()

            post_feature_map = netEncoder(post_image)   
            post_reconstruction = netDecoderPost(post_feature_map)
            errPost = args["L1_lambda"] * criterion_mae(post_reconstruction, post_image)
            errPost.backward()

            optimizerEncoder.step()
            optimizerDecoderPre.step()
            optimizerDecoderPost.step()


            if i % args["n_critic"] == 0:
                pre_losses.append(errPre.item())
                post_losses.append(errPost.item())

            if i % 20 == 0:
                print('[%d/%d] [%d/%d]\tLoss_pre: %.4f\tLoss_post: %.4f\t'
                      % (epoch, args["epoch"], i, len(dataloader),
                         errPre.item(), errPost.item() ))

            # if ((iters % 199 == 0) and epoch % 5 == 0) or ((epoch == args["epoch"] - 1) and (i == len(dataloader)-1)):
            
            if iters % (100*len(dataloader)) == 0 or ((epoch == args["epoch"] - 1) and (i == len(dataloader)-2)):
                with torch.no_grad():
                    fake = netDecoderPost(netEncoder(pre_image)).detach().cpu()
                    img_list.append(fake.numpy())
                    save_img(img_list, epoch)

                torch.save(netEncoder.state_dict(), os.path.join(args["checkpoint_dir"], "encoder_AE_opc.pkl"))
                torch.save(netDecoderPre.state_dict(), os.path.join(args["checkpoint_dir"], "decoder_pre_AE_opc.pkl"))
                torch.save(netDecoderPost.state_dict(), os.path.join(args["checkpoint_dir"], "decoder_post_AE_opc.pkl"))
            
            iters = iters + 1

    plt.figure(figsize=(10, 5))
    plt.title("Autoencoder Loss During Training")
    plt.plot(pre_losses, label='AE_pre')
    plt.plot(post_losses, label='AE_post')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(args["img_output_dir"], 'AE_loss_opc.png'))



