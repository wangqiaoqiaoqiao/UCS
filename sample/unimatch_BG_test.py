from osgeo import gdal,gdalconst
import os
import tifffile as tif
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml
import albumentations as A
import numpy as np

# imgsize for google
imgsize = 1024
# for test
image_transform_test = A.Compose([
    A.CenterCrop(width=imgsize, height=imgsize, always_apply=True),
]
)
IMG_MEAN_ALL_Ge = np.array([109.9142, 105.5509, 103.6405])
IMG_STD_ALL_Ge = np.array([57.0574, 50.5545, 48.2372])
import torch.utils.data as data
from PIL import Image
import segmentation_models_pytorch as smp
from util.utils import init_log

class Segmentation_test_Ge_only(data.Dataset):
    def __init__(self, data_dir, channels=3, aug=False):
        self.images_path_list = []
        for root, dirs, files in os.walk(data_dir, topdown=False):
            for name in files:
                if (name != "Thumbs.db"):
                    self.images_path_list.append(os.path.join(data_dir, name))
        self.scale = 0.5
        self.aug = aug  # augmentation for images
        self.channels = channels

    def __getitem__(self, item):
        img_path = self.images_path_list[item]
        dataset = gdal.Open(img_path, gdalconst.GA_ReadOnly)
        img = dataset.ReadAsArray().transpose(1, 2, 0)[:, :, :3]

        img = (img - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
        return img_path, img

    def __len__(self):
        return len(self.images_path_list)

def main():

    nchannels = 3
    # model = DeepLabV3Plus(cfg)
    model = smp.Unet(encoder_name="timm-regnety_040", encoder_weights="imagenet",
                     in_channels=nchannels, classes=2).cuda()
    model.cuda()

    data_dir = r'Y:\Training_Data\ESB'
    logdir = r'D:\wwr\Second\sample\runs_ESB'
    # Setup parameters
    batch_size = 1
    classes = 2
    nchannels = 3

    testdataloader = torch.utils.data.DataLoader(
        Segmentation_test_Ge_only(data_dir, channels=nchannels, aug=False),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    resume = os.path.join(logdir, 'latest.pth')
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        weights_dict = {}
        for k, v in checkpoint['model'].items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        model.load_state_dict(weights_dict)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
    else:
        print("=> no checkpoint found at resume")
        print("=> Will stop.")

    device = 'cuda'
    # should be placed after weight loading
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))



    output_dir_seg = os.path.join(r'Y:\Training_Data\Negative_pred')
    Pred(model, testdataloader, device, output_dir_seg)


def Pred(model, dataloader, device, output_dir_seg):
    model.eval()
    os.makedirs(output_dir_seg, exist_ok=True)
    with torch.no_grad():
        for img_path, x in tqdm(dataloader):
            x = x.to(device, non_blocking=True)
            output_seg = model(x)
            for t in range(x.shape[0]):
                img = np.array(Image.open(img_path[t]).convert('RGB'))
                logits = torch.nn.functional.softmax(output_seg,dim=1)
                fg = logits[0, 0].cpu().numpy()
                ypred_seg = np.zeros(fg.shape,dtype='uint8')
                ypred_seg[fg > 0.999] = 1
                cv2.imwrite(os.path.join(output_dir_seg, img_path[0].split('\\')[-1][:-4] + '.png'),
                            ypred_seg)
                ypred_seg[ypred_seg == 1] = 255
                cv2.imwrite(os.path.join(output_dir_seg, img_path[0].split('\\')[-1][:-4] + '_c.png'),
                            ypred_seg)
                combine = cv2.addWeighted(img, 0.5,np.array(Image.fromarray(ypred_seg.astype('uint8')).convert('RGB')) , 0.5, 0)
                tif.imwrite(os.path.join(output_dir_seg, img_path[0].split('\\')[-1][:-4] + '.tif'), combine)



if __name__ == "__main__":
    main()