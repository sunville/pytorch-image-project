import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class Mydataset(Dataset):
	'''image data set'''
	def __init__(self, csv_file, img_dir, transform=None):
		'''
		csv_file (string): store image names
		img_dir (string): store images
		'''
		self.csv_file = pd.read_csv(csv_file)
		self.img_dir = img_dir
		self.transform = transform

	def __len__(self):
		return len(self.csv_file)

	def __getitem__(self, idx):
		img_name = os.path.join(self.img_dir,
                                self.csv_file.iloc[idx, 0])
        image = cv2.imread(img_name)
        if self.transform:
            image = self.transform(image)

        return image


class Rescale(object):
	'''Rescale the image in a sample to a given size'''

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, image):
		'''This is for shrink image
			If zoom, use cv2.INTER_CUBIC (slow) or cv2.INTER_LINEAR (fast)'''
		# H x W x C
		h,w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h>w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)
		# cv2.resize(image, (width, height))
		return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


class ToTensor(object):
	'''Convert ndarray to FloatTensor'''

	def __call__(self, image):
		# numpy image H x W x C
		# torch image C x H x W
		image = image.transpose((2,0,1))
		return torch.from_numpy(image).type(torch.FloatTensor)


if __name__ == '__main__':
	img_dataset = Mydataset(img_dir=args["img_dir"],
							img_csv=args["img_csv"],
							transform=transforms.Compose([
								Rescale(64),
								ToTensor(),
								]))			
	dataloader = DataLoader(img_dataset, batch_size=args["batch_size"],
							shuffle=True, num_workers=2)








