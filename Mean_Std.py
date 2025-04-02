# import PIL, os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import cv2
import os
from osgeo import gdal,gdalconst

# https://gist.github.com/spirosdim/79fc88231fffec347f1ad5d14a36b5a8

class ImgDataset(Dataset):
    def __init__(self, data_dir, extension='.tif', transform=None):
        self.data_dir = data_dir
        self.image_files = []
        self.image_files = glob.glob(os.path.join(data_dir, "*.tif"))


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        dataset = gdal.Open(img_path, gdalconst.GA_ReadOnly)
        img = dataset.ReadAsArray().transpose(1, 2, 0)[:,:,:3]

        image = torch.from_numpy(img).float()


        return image


def compute_mean_std(dataloader):
    '''
    We assume that the images of the dataloader have the same height and width
    source: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py
    '''
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
    for batch_images in tqdm(dataloader):  # (B,H,W,C)
        channels_sum += torch.mean(batch_images, dim=[0, 1, 2])
        channels_sqrd_sum += torch.mean(batch_images ** 2, dim=[0, 1, 2])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5
    print(num_batches)

    return mean, std


if __name__ == "__main__":
    dataset = ImgDataset('Y:\Training_Data\ESB')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                               num_workers=16, pin_memory=True)

    # output
    total_mean, total_std = compute_mean_std(train_loader)
    print('mean (RGB): ', str(total_mean))
    print('std (RGB):  ', str(total_std))