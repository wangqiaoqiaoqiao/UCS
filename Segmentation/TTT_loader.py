import os
import torch
import numpy as np
import pandas as pd
import albumentations as A
from PIL import Image as Image
import torch.utils.data as data
from osgeo import gdal, gdalconst
imgsize = 1024
image_transform = A.Compose([
    A.CenterCrop(width=imgsize, height=imgsize, always_apply=True),
    A.Flip(p=0.5),
    A.Rotate(p=0.5),
]
)

image_transform_test = A.Compose([
    A.CenterCrop(width=imgsize, height=imgsize, always_apply=True),
]
)
IMG_MEAN_ALL_Ge = np.array([109.9142, 105.5509, 103.6405])
IMG_STD_ALL_Ge = np.array([57.0574, 50.5545, 48.2372])

class myImageFloder_IRN_pseudo_Ge_Pred_Two_unlabeled(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample=0, classes=3):
        self.datalist = []
        for root, dirs, files in os.walk(datalist, topdown=False):
            for name in files:
                if (name != "Thumbs.db"):
                    self.datalist.append(os.path.join(datalist,name))
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.classes = classes # lvwang, jianshe, background

    def __getitem__(self, index):

        img_path = self.datalist[index]

        img1 = Image.open(img_path).convert('L')  # Gray
        img1 = np.array(img1)

        img2 = Image.open(img_path).convert('RGB')  # avoid RGBA
        img2 = np.array(img2)  # convert to RGB

        # Augmentation
        # masks = []
        # masks.append(img1)
        # masks.append(mask)
        # if self.aug:
        #     transformed = image_transform(image=img2, masks=masks)
        #     img2 = transformed["image"]
        #     img1 = transformed["masks"][0]
        #     mask = transformed["masks"][1]
        # else:
        #     transformed = image_transform_test(image=img2, masks=masks)
        #     img2 = transformed["image"] # centercrop
        #     img1 = transformed["masks"][0]
        #     mask = transformed["masks"][1]

        IMG_MEAN_ALL_Get = np.array(
            [IMG_MEAN_ALL_Ge[0] * 299 / 1000 + IMG_MEAN_ALL_Ge[1] * 587 / 1000 + IMG_MEAN_ALL_Ge[2] * 114 / 1000])
        IMG_STD_ALL_Get = np.array(
            [IMG_STD_ALL_Ge[0] * 299 / 1000 + IMG_STD_ALL_Ge[1] * 587 / 1000 + IMG_STD_ALL_Ge[2] * 114 / 1000])
        img1 = (img1 - IMG_MEAN_ALL_Get) / IMG_STD_ALL_Get
        img1 = torch.from_numpy(img1).float()

        img2 = (img2 - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        return img_path, img1, img2
    def __len__(self):
        return len(self.datalist)

class myImageFloder_IRN_pseudo_Ge_Pred_Two(data.Dataset):
    def __init__(self, datalist, maskroot, channels=3, aug=False, num_sample=0, classes=3):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.maskroot = maskroot
        self.classes = classes # lvwang, jianshe, background

    def __getitem__(self, index):

        img_path = self.datalist.iloc[index, 0]

        img2 = Image.open(img_path).convert('RGB') # avoid RGBA
        img2 = np.array(img2) # convert to RGB

        ibase = os.path.basename(img_path)[:-4]
        mask_path = os.path.join(self.maskroot, ibase + '.png')  # mask
        mask = np.array(Image.open(mask_path)).astype('uint8')  # 0,1

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img2, mask=mask)
            img2 = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img2, mask=mask)
            img2 = transformed["image"]  # centercrop
            mask = transformed["mask"]
        img1 = np.array(Image.fromarray(img2).convert('L'))

        # Augmentation
        # masks = []
        # masks.append(img1)
        # masks.append(mask)
        # if self.aug:
        #     transformed = image_transform(image=img2, masks=masks)
        #     img2 = transformed["image"]
        #     img1 = transformed["masks"][0]
        #     mask = transformed["masks"][1]
        # else:
        #     transformed = image_transform_test(image=img2, masks=masks)
        #     img2 = transformed["image"] # centercrop
        #     img1 = transformed["masks"][0]
        #     mask = transformed["masks"][1]

        IMG_MEAN_ALL_Get = np.array(
            [IMG_MEAN_ALL_Ge[0] * 0.299 + IMG_MEAN_ALL_Ge[1] * 0.587 + IMG_MEAN_ALL_Ge[2] * 0.114])
        IMG_STD_ALL_Get = np.array([IMG_STD_ALL_Ge[0] * 0.299 + IMG_STD_ALL_Ge[1] * 0.587 + IMG_STD_ALL_Ge[2] * 0.114])

        img1 = (img1 - IMG_MEAN_ALL_Get) / IMG_STD_ALL_Get
        img1 = torch.from_numpy(img1).float()

        img2 = (img2 - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        # mask[mask==2] = 0
        # mask[mask==3] = 0
        mask = torch.from_numpy(mask)
        return img_path, img1, img2, mask
    def __len__(self):
        return len(self.datalist)

class myImageFloder_IRN_pseudo_Ge_Pred_No_Color(data.Dataset):
    def __init__(self, datalist, maskroot, channels=3, aug=False, num_sample=0, classes=3):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.maskroot = maskroot
        self.classes = classes # lvwang, jianshe, background

    def __getitem__(self, index):

        img_path = self.datalist.iloc[index, 0]

        img = Image.open(img_path).convert('L') # avoid RGBA
        img = np.array(img) # convert to RGB

        ibase = os.path.basename(img_path)[:-4]
        mask_path = os.path.join(self.maskroot, ibase + '.png')  # mask
        mask = np.array(Image.open(mask_path)).astype('uint8')  # 0,1
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=mask)
            img = transformed["image"] # centercrop
            mask = transformed["mask"]

        IMG_MEAN_ALL_Get = np.array(
            [IMG_MEAN_ALL_Ge[0] * 299 / 1000 + IMG_MEAN_ALL_Ge[1] * 587 / 1000 + IMG_MEAN_ALL_Ge[2] * 114 / 1000])
        IMG_STD_ALL_Get = np.array(
            [IMG_STD_ALL_Ge[0] * 299 / 1000 + IMG_STD_ALL_Ge[1] * 587 / 1000 + IMG_STD_ALL_Ge[2] * 114 / 1000])
        img = (img - IMG_MEAN_ALL_Get) / IMG_STD_ALL_Get
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask)
        return img_path, img, mask
    def __len__(self):
        return len(self.datalist)

class myImageFloder_IRN_pseudo_Ge_No_Color(data.Dataset):
    def __init__(self,  maskroot, datalist, channels=3, aug=False, num_sample=0, classes=3):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample > 0:  # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug  # augmentation for images
        self.channels = channels
        self.classes = classes  # lvwang, jianshe, background
        # add
        self.maskroot = maskroot

    def __getitem__(self, index):

        img_path = self.datalist.iloc[index, 0]
        # dataset = gdal.Open(img_path, gdalconst.GA_ReadOnly)
        # img = dataset.ReadAsArray().transpose(1, 2, 0)[:, :, :3]
        img = Image.open(img_path).convert('L')
        img = np.array(img) # convert to RGB

        ibase = os.path.basename(img_path)[:-4]
        mask_path = os.path.join(self.maskroot, ibase+'.png') # mask
        mask = np.array(Image.open(mask_path)).astype('uint8') # 0,1


        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask =mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=mask)
            img = transformed["image"] # centercrop
            mask = transformed["mask"]
        IMG_MEAN_ALL_Get = np.array([IMG_MEAN_ALL_Ge[0] * 0.299 + IMG_MEAN_ALL_Ge[1] * 0.587 + IMG_MEAN_ALL_Ge[2] * 0.114])
        IMG_STD_ALL_Get = np.array([IMG_STD_ALL_Ge[0] * 0.299 + IMG_STD_ALL_Ge[1] * 0.587 + IMG_STD_ALL_Ge[2] * 0.114])
        img = (img - IMG_MEAN_ALL_Get) / IMG_STD_ALL_Get
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask

    def __len__(self):
        return len(self.datalist)

class myImageFloder_IRN_pseudo_Ge(data.Dataset):
    def __init__(self,  maskroot, datalist, channels=3, aug=False, num_sample=0, classes=3):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.classes = classes # lvwang, jianshe, background
        # add
        self.maskroot = maskroot

    def __getitem__(self, index):

        img_path = self.datalist.iloc[index, 0]
        dataset = gdal.Open(img_path, gdalconst.GA_ReadOnly)
        img = dataset.ReadAsArray().transpose(1, 2, 0)[:, :, :3]

        # img = Image.open(img_path).convert('RGB') # avoid RGBA
        # img = np.array(img) # convert to RGB

        ibase = os.path.basename(img_path)[:-4]
        mask_path = os.path.join(self.maskroot, ibase+'.png') # mask
        mask = np.array(Image.open(mask_path)).astype('uint8') # 0,1

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=mask)
            img = transformed["image"] # centercrop
            mask = transformed["mask"]

        img = (img - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        return img, mask

    def __len__(self):
        return len(self.datalist)

class myImageFloder_IRN_pseudo_Ge_Test(data.Dataset):
    def __init__(self, datalist, labellist, channels=3, aug=False, num_sample=0, classes=3):
        self.aug = aug # augmentation for images
        self.channels = channels
        self.classes = classes # lvwang, jianshe, background
        self.datalist = []
        self.labellist = []
        for root, dirs, files in os.walk(datalist):
            for name in files:
                if (name != "Thumbs.db") and '.tif' in name:
                    self.datalist.append(os.path.join(datalist, name))
                    self.labellist.append(os.path.join(labellist, name[:-4] + '.png'))

    def __getitem__(self, index):

        img_path = self.datalist[index]

        img2 = Image.open(img_path).convert('RGB')  # avoid RGBA
        img2 = np.array(img2)  # convert to RGB

        mask_path = self.labellist[index]
        mask = np.array(Image.open(mask_path)).astype('uint8')  # 0,1

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img2, mask=mask)
            img2 = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img2, mask=mask)
            img2 = transformed["image"]  # centercrop
            mask = transformed["mask"]
        img1 = np.array(Image.fromarray(img2).convert('L'))

        IMG_MEAN_ALL_Get = np.array(
            [IMG_MEAN_ALL_Ge[0] * 0.299 + IMG_MEAN_ALL_Ge[1] * 0.587 + IMG_MEAN_ALL_Ge[2] * 0.114])
        IMG_STD_ALL_Get = np.array([IMG_STD_ALL_Ge[0] * 0.299 + IMG_STD_ALL_Ge[1] * 0.587 + IMG_STD_ALL_Ge[2] * 0.114])

        img1 = (img1 - IMG_MEAN_ALL_Get) / IMG_STD_ALL_Get
        img1 = torch.from_numpy(img1).float()

        img2 = (img2 - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        mask = torch.from_numpy(mask)
        return img_path, img1, img2, mask
    def __len__(self):
        return len(self.datalist)

class myImageFloder_IRN_pseudo_Ge_Pred(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample=0, classes=3):
        self.aug = aug # augmentation for images
        self.channels = channels
        self.classes = classes # lvwang, jianshe, background
        self.datalist = []
        self.labellist = []
        for root, dirs, files in os.walk(datalist):
            for name in files:
                if (name != "Thumbs.db") and name not in ['2000000000000140f_11_1d80_20.tif',
                                                          '200000000000004f9_29_30.tif', '60000000000000cd4_41_52.tif',
                                                          '20000000000000373_41_9df_50.tif']\
                        and '.tif' in name and 'pro' not in name and 'xml' not in name and 'ovr' not in name:
                    self.datalist.append(os.path.join(datalist, name))

    def __getitem__(self, index):

        img_path = self.datalist[index]

        img2 = Image.open(img_path).convert('RGB')  # avoid RGBA
        img2 = np.array(img2)  # convert to RGB


        # Augmentation
        if self.aug:
            transformed = image_transform(image=img2)
            img2 = transformed["image"]
        else:
            transformed = image_transform_test(image=img2)
            img2 = transformed["image"]  # centercrop
        img1 = np.array(Image.fromarray(img2).convert('L'))

        IMG_MEAN_ALL_Get = np.array(
            [IMG_MEAN_ALL_Ge[0] * 0.299 + IMG_MEAN_ALL_Ge[1] * 0.587 + IMG_MEAN_ALL_Ge[2] * 0.114])
        IMG_STD_ALL_Get = np.array([IMG_STD_ALL_Ge[0] * 0.299 + IMG_STD_ALL_Ge[1] * 0.587 + IMG_STD_ALL_Ge[2] * 0.114])

        img1 = (img1 - IMG_MEAN_ALL_Get) / IMG_STD_ALL_Get
        img1 = torch.from_numpy(img1).float()

        img2 = (img2 - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        return img_path, img1, img2
    def __len__(self):
        return len(self.datalist)



