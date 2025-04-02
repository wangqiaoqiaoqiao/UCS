from sample.dataset.transform import *
import albumentations as A
from copy import deepcopy
from osgeo import gdal,gdalconst
import math
import numpy as np
import os
import random
import pandas as pd
from PIL import Image
import torch
import tifffile as tif
from torch.utils.data import Dataset
from torchvision import transforms
# imgsize for google
imgsize = 512
image_transform = A.Compose([
    A.CenterCrop(width=imgsize, height=imgsize, always_apply=True),
    A.Flip(p=0.5),
    A.Rotate(p=0.5),
]
)
from enum import IntEnum
class Resampling(IntEnum):
    NEAREST = 0
    BOX = 4
    BILINEAR = 2
    HAMMING = 5
    BICUBIC = 3
    LANCZOS = 1
class myImageFloder_labeled_neg(Dataset):
    def __init__(self, datalist, labelroot, channels=3, transform=None, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.size = 512
        self.scale = 0.5
        self.channels = channels
        self.labelroot = labelroot
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        # img = Image.open(img_path).convert('RGB')
        dataset = gdal.Open(img_path, gdalconst.GA_ReadOnly)
        cols = dataset.RasterXSize  # 列数
        rows = dataset.RasterYSize  # 行数
        if cols == 1024 and rows == 1024:
            colst = int(cols * self.scale)  # 计算新的行列数
            rowst = int(rows * self.scale)
            img = dataset.ReadAsArray(buf_xsize=colst, buf_ysize=rowst,
                                      resample_alg=gdalconst.GRIORA_CubicSpline).transpose(1, 2, 0)[:,:,:3]
        else:
            img = dataset.ReadAsArray().transpose(1, 2, 0)[:,:,:3]

        if 'negative' in img_path:
            mask = np.zeros(img.shape[:2], dtype='uint8')
        else:
            ibase = os.path.basename(img_path)[:-4]
            mask_path = os.path.join(self.labelroot, ibase + '.png')
            mask = Image.open(mask_path)
            if cols == 1024 and rows == 1024:
                mask = mask.resize((int(cols * self.scale), int(rows * self.scale)), resample=Resampling.NEAREST)
            mask = np.array(mask)

        transformed = image_transform(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        # width, height = img.size
        # img = img.resize((int(width * 2), int(height * 2)), resample=Resampling.BILINEAR)
        # mask = Image.fromarray(mask).resize((int(width * 2), int(height * 2)), resample=Resampling.NEAREST)

        # img, mask = resize(img, mask, (0.5, 2.0))
        # ignore_value = 255
        # img, mask = crop(img, mask, self.size, ignore_value)
        # img, mask = hflip(img, mask, p=0.5)

        return normalize(img, mask)


    def __len__(self):
        return len(self.datalist)
class myImageFloder_labeled(Dataset):
    def __init__(self, datalist, labelroot, channels=3, transform=None, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.scale = 0.5
        self.channels = channels
        self.labelroot = labelroot
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)[:3,:,:].transpose(1,2,0)
        # img = Image.open(img_path).convert('RGB')
        # img = gdal.Open(img_path).ReadAsArray()[:3,:,:].transpose(1,2,0)
        ibase = os.path.basename(img_path)[:-4]
        mask_path = os.path.join(self.labelroot, ibase + '.tif')
        mask = Image.open(mask_path)
        # dataset = gdal.Open(img_path, gdalconst.GA_ReadOnly)
        # cols = dataset.RasterXSize  # 列数
        # rows = dataset.RasterYSize  # 行数
        # ibase = os.path.basename(img_path)[:-4]
        # mask_path = os.path.join(self.labelroot, ibase + '.png')
        # mask = Image.open(mask_path)
        #
        # if cols == 1024 and rows == 1024:
        #     cols = int(cols * self.scale)  # 计算新的行列数
        #     rows = int(rows * self.scale)
        #     img = dataset.ReadAsArray(buf_xsize=cols, buf_ysize=rows,
        #                               resample_alg=gdalconst.GRIORA_CubicSpline)
        # else:
        #     img = dataset.ReadAsArray()

        transformed = image_transform(image=np.array(img), mask=np.array(mask))
        img = transformed["image"]
        mask = transformed["mask"]
        # Globel
        mask[mask != 2] = 0
        mask[mask == 2] = 1

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        # width, height = img.size
        # img = img.resize((int(width * 2), int(height * 2)), resample=Resampling.BILINEAR)
        # mask = Image.fromarray(mask).resize((int(width * 2), int(height * 2)), resample=Resampling.NEAREST)

        # img, mask = resize(img, mask, (0.5, 2.0))
        # ignore_value = 255
        # img, mask = crop(img, mask, self.size, ignore_value)
        # img, mask = hflip(img, mask, p=0.5)

        return normalize(img, mask)


    def __len__(self):
        return len(self.datalist)

class myImageFloder_unlabeled_fix(Dataset):
    def __init__(self, data_dir, channels=3, transform_weak=None,transform_strong=None, num_sample=0):
        self.datalist = []
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.size = 512
        self.scale = 0.5
        self.channels = channels
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        for root, dirs, files in os.walk(data_dir, topdown=False):
            for name in files:
                if (name != "Thumbs.db"):
                    self.datalist.append(os.path.join(data_dir, name))

    def __getitem__(self, index):
        img_path = self.datalist[index]
        # img = Image.open(img_path).convert('RGB')

        dataset = gdal.Open(img_path, gdalconst.GA_ReadOnly)
        cols = dataset.RasterXSize  # 列数
        rows = dataset.RasterYSize  # 行数
        cols = int(cols * self.scale)  # 计算新的行列数
        rows = int(rows * self.scale)
        img = dataset.ReadAsArray(buf_xsize=cols, buf_ysize=rows, resample_alg=gdalconst.GRIORA_CubicSpline).transpose(
            1, 2, 0)
        img = Image.fromarray(img)
        mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        # img, mask = resize(img, mask, (0.5, 2.0))
        # ignore_value = 254
        # img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)
        #
        # if random.random() < 0.8:
        #     img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        # img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        # img_s2 = blur(img_s2, p=0.5)
        # cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        # img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, ignore_mask, cutmix_box1

        # return normalize(img_w), img_s1, ignore_mask, cutmix_box1
    def __len__(self):
        return len(self.datalist)

class myImageFloder_unlabeled(Dataset):
    def __init__(self, data_dir, channels=3, transform_weak=None,transform_strong=None, num_sample=0):
        self.datalist = []
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.size = 512
        self.scale = 0.5
        self.channels = channels
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        for root, dirs, files in os.walk(data_dir, topdown=False):
            for name in files:
                if (name != "Thumbs.db"):
                    self.datalist.append(os.path.join(data_dir, name))

    def __getitem__(self, index):
        img_path = self.datalist[index]
        # img = Image.open(img_path).convert('RGB')

        dataset = gdal.Open(img_path, gdalconst.GA_ReadOnly)
        cols = dataset.RasterXSize  # 列数
        rows = dataset.RasterYSize  # 行数
        cols = int(cols * self.scale)  # 计算新的行列数
        rows = int(rows * self.scale)
        img = dataset.ReadAsArray(buf_xsize=cols, buf_ysize=rows, resample_alg=gdalconst.GRIORA_CubicSpline).transpose(
            1, 2, 0)
        img = Image.fromarray(img)
        mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        # img, mask = resize(img, mask, (0.5, 2.0))
        # ignore_value = 254
        # img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

        # return normalize(img_w), img_s1, ignore_mask, cutmix_box1
    def __len__(self):
        return len(self.datalist)

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
