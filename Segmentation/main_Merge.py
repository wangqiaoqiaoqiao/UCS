import os
import time
from visdom import Visdom
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import cv2
import torch
import shutil
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils import data
from tensorboardX import SummaryWriter #change tensorboardX
from TTT_loader import myImageFloder_IRN_pseudo_Ge_Pred_Two
from metrics import SegmentationMetric, AverageMeter, acc2file
from models.Unet_mix_transformer import *

class Unet_mit_gray(nn.Module):
    def __init__(self, encoder_name, decoder_key, pretrained,align_corners=False):
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
    def init_checkpoint(self,pretrained):
        checkpoint = torch.load(pretrained)
        checkpoint.popitem()
        checkpoint.popitem()
        if self.encoder.state_dict()['patch_embed1.proj.weight'].shape == torch.Size([64, 1, 7, 7]):
            checkpoint.pop('patch_embed1.proj.weight')
        self.encoder.load_state_dict(checkpoint, strict=False)
    def forward(self,x):
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
    def __init__(self, encoder_name, decoder_key, pretrained,align_corners =False):
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
    def init_checkpoint(self,pretrained):
        checkpoint = torch.load(pretrained)
        checkpoint.popitem()
        checkpoint.popitem()
        if self.encoder.state_dict()['patch_embed1.proj.weight'].shape == torch.Size([64, 1, 7, 7]):
            checkpoint.pop('patch_embed1.proj.weight')
        self.encoder.load_state_dict(checkpoint, strict=False)
    def forward(self,x):
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
        self.gate_c.add_module('flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module('gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]))
            self.gate_c.add_module('gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]))
            self.gate_c.add_module('gate_c_relu_%d'%(i+1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor).contiguous()
class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0',nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d'%i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1))
    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor).contiguous()
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor


class Decorder(nn.Module):
    def __init__(self, decoder_key,align_corners =False):
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
            feature_gray_ = F.sigmoid(self.attention[int(2*i)](feature_gray[i]) * self.attention[int(2*i+1)](feature_gray[i]))
            feature.append(feature_gray_*feature_rgb[i])

        output = self.decoder(feature)
        output = resize(
            input=output,
            size=x.shape[-1],
            mode='bilinear',
            align_corners=self.align_corners)

        return output

IMG_MEAN_ALL_Ge = np.array([109.9142, 105.5509, 103.6405])
IMG_STD_ALL_Ge = np.array([57.0574, 50.5545, 48.2372])

def main():
    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # Setup datalist
    trainlist_pos = r"D:/wwr/Second/Segmentation/data/train_0.9_0.1.txt"
    label_dir = r'Z:/Training_Data/Label'
    testlist = r'D:/wwr/Second/Segmentation/data/test_0.1_0.1.txt'

    # Setup parameters
    batch_size = 8
    epochs = 100
    classes = 4
    nchannels = 3
    device = 'cuda'
    logdir = r'.\runs\pos_pseudov2_Merge_retrain_0.1'
    global best_acc
    best_acc = 0

    writer = SummaryWriter(log_dir=logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # train & test dataloader
    traindataloader_pos = torch.utils.data.DataLoader(
        myImageFloder_IRN_pseudo_Ge_Pred_Two(trainlist_pos,label_dir, aug=True, channels=nchannels),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    testdataloader = torch.utils.data.DataLoader(
        myImageFloder_IRN_pseudo_Ge_Pred_Two(testlist,label_dir, aug=False, channels=nchannels),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
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



    resume1 = os.path.join(r'.\runs\pos_pseudov2_Gray_0.1', 'checkpoint20.tar')
    resume2 = os.path.join(r'.\runs\pos_pseudov2_RGB_0.1', 'checkpoint20.tar')
    if os.path.isfile(resume1):
        print("=> loading checkpoint '{}'".format(resume1))
        checkpoint = torch.load(resume1)
        net_gray.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume1, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

        print("=> loading checkpoint '{}'".format(resume2))
        checkpoint = torch.load(resume2)
        checkpoint['state_dict'].pop('decoder.linear_pred.weight')
        checkpoint['state_dict'].pop('decoder.linear_pred.bias')
        net_rgb.load_state_dict(checkpoint['state_dict'],strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume2, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at resume")
        print("=> Will stop.")



    # should be placed after weight loading
    if torch.cuda.device_count() > 1:
        net_gray = torch.nn.DataParallel(net_gray, device_ids=range(torch.cuda.device_count()))
        net_rgb = torch.nn.DataParallel(net_rgb, device_ids=range(torch.cuda.device_count()))
        decorder = torch.nn.DataParallel(decorder, device_ids=range(torch.cuda.device_count()))
    for param in net_gray.parameters():
        param.requires_grad = False
    for param in net_rgb.parameters():
        param.requires_grad = False
    for param in net_gray.module.encoder.block4.parameters():
        param.requires_grad = True
    for param in net_rgb.module.encoder.block4.parameters():
        param.requires_grad = True
    for param in net_gray.module.encoder.block3.parameters():
        param.requires_grad = True
    for param in net_rgb.module.encoder.block3.parameters():
        param.requires_grad = True

    for param in net_rgb.module.decoder.parameters():
        param.requires_grad = True

    print('RGB:', list(net_rgb.module.encoder.block3.parameters()) + list(net_rgb.module.encoder.block4.parameters()) +
          list(net_rgb.module.decoder.parameters()) == list(filter(lambda p: p.requires_grad, net_rgb.parameters())))
    print('Gray:',
          list(net_gray.module.encoder.block3.parameters()) + list(net_gray.module.encoder.block4.parameters())
          == list(filter(lambda p: p.requires_grad, net_gray.parameters())))
    print('Decorder:', list(decorder.parameters()) == list(filter(lambda p: p.requires_grad, decorder.parameters())))
    optimizer = torch.optim.Adam(list(decorder.parameters())
                                 + list(net_gray.module.encoder.block4.parameters()) + list(
        net_rgb.module.encoder.block4.parameters())
                                 + list(net_gray.module.encoder.block3.parameters()) + list(
        net_rgb.module.encoder.block3.parameters())
                                 + list(net_rgb.module.decoder.parameters()),
                                 lr=0.0001, betas=(0.9, 0.999))

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    start_epoch = 0
    # 3. train  with pseudo labels
    for epoch in range(epochs-start_epoch):
        epoch = start_epoch + epoch + 1 # current epochs
        # adjust_learning_rate(optimizer, epoch)
        lr = optimizer.param_groups[0]['lr']
        print('epoch %d, lr: %.6f'%(epoch, lr))
        since = time.time()
        train_loss, train_oa, train_iou = train_epoch(net_gray, net_rgb, decorder, criterion,
                                                        traindataloader_pos,
                                                          optimizer, device, epoch, classes)
        print('train', time.time()-since)
        val_oa, val_iou = vtest_epoch(net_gray, net_rgb, decorder, criterion, testdataloader, device, epoch, classes, logdir)

        # save every epoch
        savefilename = os.path.join(logdir, 'checkpoint'+str(epoch)+'.tar')
        is_best = val_oa > best_acc
        best_acc = max(val_oa , best_acc)  # update
        torch.save({
            'epoch': epoch,
            'state_dict_decoder': decorder.module.state_dict() if hasattr(decorder, "module") else decorder.state_dict(), # multiple GPUs
            'state_dict_gray': net_gray.module.state_dict() if hasattr(net_gray, "module") else net_gray.state_dict(),
            'state_dict_rgb': net_rgb.module.state_dict() if hasattr(net_rgb, "module") else net_rgb.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_oa': val_oa,
            'best_acc': best_acc,
        }, savefilename)
        if is_best:
            shutil.copy(savefilename, os.path.join(logdir, 'model_best.tar'))
        # write
        writer.add_scalar('lr', lr, epoch)
        writer.add_scalar('train/1.loss', train_loss,epoch)
        writer.add_scalar('train/2.oa', train_oa, epoch)
        writer.add_scalar('train/3.iou_lvwang',train_iou[1], epoch)
        writer.add_scalar('train/4.iou_fei', train_iou[0], epoch)
        writer.add_scalar('val/2.oa', val_oa, epoch)
        writer.add_scalar('val/3.iou_lvwang',val_iou[1], epoch)
        writer.add_scalar('val/4.iou_fei', val_iou[0], epoch)
    writer.close()

def train_epoch(net_gray, net_rgb, decorder, criterion, dataloader, optimizer, device, epoch, classes):
    net_gray.train()
    net_rgb.train()
    decorder.train()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    for idx, (_, x_gray, x_rgb, update) in enumerate(dataloader):
        x_gray = x_gray.to(device, non_blocking=True).unsqueeze(1)
        x_rgb = x_rgb.to(device, non_blocking=True)  # N C H W
        update = update.to(device, non_blocking=True)  # N H W

        feature_gray, output_gray = net_gray.forward(x_gray)
        feature_rgb, output_rgb = net_rgb.forward(x_rgb)
        output = decorder.forward(x_rgb, feature_gray, feature_rgb)
        update_con = update.clone()
        update_con[update_con == 2] = 0
        update_con[update_con == 3] = 0

        update_rgb = update.clone()
        update_rgb[update_rgb == 1] = 0
        update_rgb[update_rgb == 2] = 1
        update_rgb[update_rgb == 3] = 2

        loss = criterion(output, update_con.long()) + criterion(output_rgb, update_rgb.long())

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        output = output.argmax(1)
        output[output_rgb.argmax(1) == 1] = 2
        output[output_rgb.argmax(1) == 2] = 3

        acc_total.addBatch(output[update!=255], update[update!=255])
        losses.update(loss.item(), x_rgb.size(0))

        oa = acc_total.OverallAccuracy()
        miou = acc_total.meanIntersectionOverUnion()
        iou = acc_total.IntersectionOverUnion()
        pbar.set_description(
            'Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU {miou:.3f}, '
            'IOU: {bg:.3f}, {ESB:.3f},{GPC:.3f}, {build:.3f}'.format(
            epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou,
            bg=iou[0], ESB=iou[1], GPC=iou[2], build=iou[3]))
        pbar.update()
    pbar.close()
    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou

def vtest_epoch(net_gray, net_rgb, decorder, criterion, dataloader, device, epoch, classes, logdir):
    net_gray.eval()
    net_rgb.eval()
    decorder.eval()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx, (_, x_gray, x_rgb, y_true) in enumerate(dataloader):
            x_gray = x_gray.to(device, non_blocking=True).unsqueeze(1)
            x_rgb = x_rgb.to(device, non_blocking=True)  # N C H W
            y_true = y_true.to(device, non_blocking =True) # n c h w
            feature_gray, _ = net_gray.forward(x_gray)
            feature_rgb, output_rgb = net_rgb.forward(x_rgb)
            ypred = decorder.forward(x_rgb, feature_gray, feature_rgb)

            ypred = ypred.argmax(1)
            ypred[output_rgb.argmax(1) == 1] = 2
            ypred[output_rgb.argmax(1) == 2] = 3

            acc_total.addBatch(ypred[y_true!=255], y_true[y_true!=255])

            # losses.update(loss.item(), x_rgb.size(0))
            oa = acc_total.OverallAccuracy()
            iou = acc_total.IntersectionOverUnion()
            miou = acc_total.meanIntersectionOverUnion()
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. OA {oa:.3f}, MIOU {miou:.3f}, '
                    'IOU: {bg:.3f}, {ESB:.3f},{GPC:.3f}, {build:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, oa=oa, miou=miou,
                    bg=iou[0], ESB=iou[1], GPC=iou[2], build=iou[3]))
            pbar.update()
        pbar.close()

    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    savefilename = os.path.join(logdir, 'checkpoint' + str(epoch) + '.txt')
    acc2file(acc_total, savefilename)
    return oa, iou


if __name__ == "__main__":
    main()