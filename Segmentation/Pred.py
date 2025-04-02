import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import gdal
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import tifffile as tif
import torch.utils.data as data
from models.Unet_mix_transformer import *

import albumentations as A
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
import pandas as pd
class myImageFloder_IRN_pseudo_Ge_Pred(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample=0, classes=3):
        self.aug = aug # augmentation for images
        self.channels = channels
        self.classes = classes # lvwang, jianshe, background
        self.datalist = []
        self.labellist = []
        for root, dirs, files in os.walk(datalist):
            for name in files:
                if (name != "Thumbs.db"):
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


class Unet_mit_gray(nn.Module):
    def __init__(self, encoder_name, decoder_key, pretrained, align_corners=False):
        super(Unet_mit_gray, self).__init__()
        self.align_corners = align_corners
        if encoder_name == 'mit_b0':
            self.encoder = mit_b0()
        elif encoder_name == 'mit_b1':
            self.encoder = mit_b1(inchannels=1)
        elif encoder_name == 'mit_b2':
            self.encoder = mit_b2()
        elif encoder_name == 'mit_b3':
            self.encoder = mit_b3()
        self.decoder = SegFormerHead(feature_strides=decoder_key['feature_strides'],
                                     in_channels=decoder_key['in_channels'],
                                     in_index=decoder_key['in_index'],
                                     channels=decoder_key['channels'],
                                     dropout_ratio=decoder_key['dropout_ratio'],
                                     num_classes=decoder_key['num_classes'],
                                     norm_cfg=decoder_key['norm_cfg'],
                                     align_corners=decoder_key['align_corners'],
                                     decoder_params=decoder_key['decoder_params'],
                                     loss_decode=decoder_key['loss_decode']
                                     )

        self.init_checkpoint(pretrained)

    def init_checkpoint(self, pretrained):
        checkpoint = torch.load(pretrained)
        checkpoint.popitem()
        checkpoint.popitem()
        if self.encoder.state_dict()['patch_embed1.proj.weight'].shape == torch.Size([64, 1, 7, 7]):
            checkpoint.pop('patch_embed1.proj.weight')
        self.encoder.load_state_dict(checkpoint, strict=False)

    def forward(self, x):
        origin_x = x
        x = self.encoder(x)

        output = self.decoder(x)
        output = resize(
            input=output,
            size=origin_x.shape[-1],
            mode='bilinear',
            align_corners=self.align_corners)
        return x, output

class Unet_mit_rgb(nn.Module):
    def __init__(self, encoder_name, decoder_key, pretrained, align_corners=False):
        super(Unet_mit_rgb, self).__init__()
        self.align_corners = align_corners
        if encoder_name == 'mit_b0':
            self.encoder = mit_b0()
        elif encoder_name == 'mit_b1':
            self.encoder = mit_b1(inchannels=3)
        elif encoder_name == 'mit_b2':
            self.encoder = mit_b2()
        elif encoder_name == 'mit_b3':
            self.encoder = mit_b3()
        self.decoder = SegFormerHead(feature_strides=decoder_key['feature_strides'],
                                     in_channels=decoder_key['in_channels'],
                                     in_index=decoder_key['in_index'],
                                     channels=decoder_key['channels'],
                                     dropout_ratio=decoder_key['dropout_ratio'],
                                     num_classes=decoder_key['num_classes'],
                                     norm_cfg=decoder_key['norm_cfg'],
                                     align_corners=decoder_key['align_corners'],
                                     decoder_params=decoder_key['decoder_params'],
                                     loss_decode=decoder_key['loss_decode']
                                     )

        self.init_checkpoint(pretrained)

    def init_checkpoint(self, pretrained):
        checkpoint = torch.load(pretrained)
        checkpoint.popitem()
        checkpoint.popitem()
        if self.encoder.state_dict()['patch_embed1.proj.weight'].shape == torch.Size([64, 1, 7, 7]):
            checkpoint.pop('patch_embed1.proj.weight')
        self.encoder.load_state_dict(checkpoint, strict=False)

    def forward(self, x):
        origin_x = x
        x = self.encoder(x)

        output = self.decoder(x)
        output = resize(
            input=output,
            size=origin_x.shape[-1],
            mode='bilinear',
            align_corners=self.align_corners)

        return x, output

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1).contiguous()

class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i + 1), nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor).contiguous()

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0',
                               nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i,
                                   nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,
                                             kernel_size=3, \
                                             padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor).contiguous()

class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, in_tensor):
        att = 1 + F.sigmoid(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor

class Decorder(nn.Module):
    def __init__(self, decoder_key, align_corners=False):
        super(Decorder, self).__init__()
        self.align_corners = align_corners

        self.decoder = SegFormerHead(feature_strides=decoder_key['feature_strides'],
                                     in_channels=decoder_key['in_channels'],
                                     in_index=decoder_key['in_index'],
                                     channels=decoder_key['channels'],
                                     dropout_ratio=decoder_key['dropout_ratio'],
                                     num_classes=decoder_key['num_classes'],
                                     norm_cfg=decoder_key['norm_cfg'],
                                     align_corners=decoder_key['align_corners'],
                                     decoder_params=decoder_key['decoder_params'],
                                     loss_decode=decoder_key['loss_decode']
                                     )
        self.attention = nn.ModuleList()
        self.channels = [64, 128, 320, 512]
        for i in range(len(self.channels)):
            self.attention.append(ChannelGate(self.channels[i]))
            self.attention.append(SpatialGate(self.channels[i]))

    def forward(self, x, feature_gray, feature_rgb):
        feature = []
        for i in range(len(feature_rgb)):
            feature_gray_ = F.sigmoid(
                self.attention[int(2 * i)](feature_gray[i]) * self.attention[int(2 * i + 1)](feature_gray[i]))
            # feature_rgb_ = F.sigmoid(self.attention[int(2*i)](feature_rgb[i]) * self.attention[int(2*i+1)](feature_rgb[i]))
            feature.append(feature_gray_ * feature_rgb[i])
        output = self.decoder(feature)
        output = resize(
            input=output,
            size=x.shape[-1],
            mode='bilinear',
            align_corners=self.align_corners)

        return output

def Write_img2array(tiff_file, im_proj, im_geotrans, data_array):
    if 'int8' in data_array.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data_array.dtype.name:
        datatype = gdal.GDT_Uint16
    else:
        datatype = gdal.GDT_Float32

    if len(data_array.shape) == 3:
        im_bands, im_height, im_width = data_array.shape
    else:
        im_bands, (im_height, im_width) = 1, data_array.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(tiff_file, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(data_array)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(data_array[i])

    del dataset

def main():
    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    image_dir = r'Z:\Training_Data\ESB'
    batch_size = 1
    nchannels = 3
    classes = 4
    device = 'cuda'
    testdataloader = torch.utils.data.DataLoader(
        # myImageFloder( testlablist, aug=False, channels=nchannels),
        myImageFloder_IRN_pseudo_Ge_Pred(image_dir, aug=False, channels=nchannels),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    resume = os.path.join(r'.\runs\pos_pseudov2_Merge_retrain', 'model_best.tar')
    norm_cfg = dict(type='BN', requires_grad=True)
    net_gray = Unet_mit_gray(encoder_name='mit_b1',
                             decoder_key=dict(
                                 type='SegFormerHead',
                                 in_channels=[64, 128, 320, 512],
                                 in_index=[0, 1, 2, 3],
                                 feature_strides=[4, 8, 16, 32],
                                 channels=128,
                                 dropout_ratio=0.1,
                                 num_classes=classes,
                                 norm_cfg=norm_cfg,
                                 align_corners=False,
                                 decoder_params=dict(embed_dim=256),
                                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
                             pretrained='./pretrained/mit_b1.pth').to(device)
    net_rgb = Unet_mit_rgb(encoder_name='mit_b1',
                           decoder_key=dict(
                               type='SegFormerHead',
                               in_channels=[64, 128, 320, 512],
                               in_index=[0, 1, 2, 3],
                               feature_strides=[4, 8, 16, 32],
                               channels=128,
                               dropout_ratio=0.1,
                               num_classes=3,
                               norm_cfg=norm_cfg,
                               align_corners=False,
                               decoder_params=dict(embed_dim=256),
                               loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
                           pretrained='./pretrained/mit_b1.pth').to(device)
    decorder = Decorder(decoder_key=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))).to(device)

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net_gray.load_state_dict(checkpoint['state_dict_gray'])
        net_rgb.load_state_dict(checkpoint['state_dict_rgb'])
        decorder.load_state_dict(checkpoint['state_dict_decoder'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
    else:
        print("=> no checkpoint found at resume")
        print("=> Will stop.")
    # should be placed after weight loading

    net_gray.eval()
    net_rgb.eval()
    decorder.eval()
    output_dir_seg = r'Z:\data_for_dataset\data\jianshe'
    Cut_Image_dir = r'Z:\Training_Data\ESB'
    os.makedirs(output_dir_seg, exist_ok=True)
    colors = [[0, 0, 0], [255, 255, 255], [38, 38, 205], [34, 139, 34], [255, 191, 0]]  # 黑 白 黄 绿

    with torch.no_grad():
        for (img_path, x_gray, x_rgb) in tqdm(testdataloader):
            if os.path.exists(os.path.join(output_dir_seg, img_path[0].split('\\')[-1][:-4] + '.png')):
                continue
            else:
                x_gray = x_gray.to(device, non_blocking=True).unsqueeze(1)
                x_rgb = x_rgb.to(device, non_blocking=True)  # N C H W

                feature_gray, _ = net_gray.forward(x_gray)
                feature_rgb, output_rgb = net_rgb.forward(x_rgb)
                ypred = decorder.forward(x_rgb, feature_gray, feature_rgb)

                ypred = ypred.argmax(1)
                ypred[output_rgb.argmax(1) == 1] = 2
                ypred[output_rgb.argmax(1) == 2] = 3

                for t in range(x_rgb.shape[0]):
                    pred = ypred.squeeze().detach().cpu().numpy().astype('uint8')
                    pred[pred == 2] = 0
                    pred[pred == 3] = 0
                    cv2.imwrite(os.path.join(output_dir_seg, img_path[t].split('\\')[-1][:-4] + '.png'), pred)
                    pred[pred == 1] = 255
                    cv2.imwrite(os.path.join(output_dir_seg, img_path[t].split('\\')[-1][:-4] + '_c.png'), pred)




if __name__ == "__main__":
    main()