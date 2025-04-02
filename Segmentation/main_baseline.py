import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import shutil
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from tensorboardX import SummaryWriter #change tensorboardX
from TTT_loader import myImageFloder_IRN_pseudo_Ge
from metrics import SegmentationMetric, AverageMeter
from models.Unet_mix_transformer import Unet_mit


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
    logdir = r'.\runs\pos_pseudov2_RGB_0.1'
    global best_acc
    best_acc = 0

    writer = SummaryWriter(log_dir=logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # train & test dataloader
    traindataloader_pos = torch.utils.data.DataLoader(
        myImageFloder_IRN_pseudo_Ge(label_dir, trainlist_pos, aug=True, channels=nchannels),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testdataloader = torch.utils.data.DataLoader(
        # myImageFloder( testlablist, aug=False, channels=nchannels),
        myImageFloder_IRN_pseudo_Ge(label_dir, testlist, aug=False, channels=nchannels),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    norm_cfg = dict(type='BN', requires_grad=True)
    net = Unet_mit(encoder_name='mit_b1',
                   inchannels=3,
                   decoder_key=dict(
                       type='SegFormerHead',
                       in_channels=[64, 128, 320, 512],
                       in_index=[0, 1, 2, 3],
                       feature_strides=[4, 8, 16, 32],
                       channels=128,
                       dropout_ratio=0.1,
                       num_classes=4,
                       norm_cfg=norm_cfg,
                       align_corners=False,
                       decoder_params=dict(embed_dim=256),
                       loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
                   pretrained='./pretrained/mit_b1.pth').to(device)


    # print the model
    start_epoch = 0
    resume = os.path.join(r'.\runs\pos_pseudov2_RGB_0.1', 'checkpoint20.tar')
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
    else:
        print("=> no checkpoint found at resume")
        print("=> Will stop.")


    # should be placed after weight loading
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    # get all parameters (model parameters + task dependent log variances)
    # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    # 3. train  with pseudo labels
    for epoch in range(epochs-start_epoch):
        epoch = start_epoch + epoch + 1 # current epochs
        # adjust_learning_rate(optimizer, epoch)
        lr = optimizer.param_groups[0]['lr']
        print('epoch %d, lr: %.6f'%(epoch, lr))
        since = time.time()
        train_loss, train_oa, train_iou = train_epoch(net, criterion,
                                                        traindataloader_pos,
                                                          optimizer, device, epoch, classes)
        print('train', time.time()-since)
        val_loss, val_oa, val_iou = vtest_epoch(net, criterion, testdataloader, device, epoch, classes)
        # save every epoch
        savefilename = os.path.join(logdir, 'checkpoint'+str(epoch)+'.tar')
        is_best = val_oa > best_acc
        best_acc = max(val_oa , best_acc)  # update
        torch.save({
            'epoch': epoch,
            'state_dict': net.module.state_dict() if hasattr(net, "module") else net.state_dict(),  # multiple GPUs
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
        writer.add_scalar('val/1.loss', val_loss, epoch)
        writer.add_scalar('val/2.oa', val_oa, epoch)
        writer.add_scalar('val/3.iou_lvwang',val_iou[1], epoch)
        writer.add_scalar('val/4.iou_fei', val_iou[0], epoch)
    writer.close()


def train_epoch(net, criterion, dataloader, optimizer, device, epoch, classes):
    net.train()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    for idx, (images, update) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        update = update.to(device, non_blocking=True) # N 1 H W

        output = net(images)
        loss = criterion(output, update.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = output.argmax(1)
        # output = (torch.sigmo1id(output)>0.5) # N C H W
        acc_total.addBatch(output[update!=255], update[update!=255])
        losses.update(loss.item(), images.size(0))

        oa = acc_total.OverallAccuracy()
        miou = acc_total.meanIntersectionOverUnion()
        iou = acc_total.IntersectionOverUnion()
        pbar.set_description(
            'Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU {miou:.3f}, '
            'IOU: {bg:.3f}, {ESB:.3f},{GPC:.3f}, {build:.3f}'.format(
            epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou,
            bg=iou[0], ESB=iou[1], GPC=iou[2],build=iou[3]))
        pbar.update()
    pbar.close()
    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou

def vtest_epoch(model, criterion, dataloader, device, epoch, classes):
    model.eval()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx, (x, y_true) in enumerate(dataloader):
            x = x.to(device, non_blocking =True)
            y_true = y_true.to(device, non_blocking =True) # n c h w
            ypred = model.forward(x)

            loss = criterion(ypred, y_true.long())

            ypred = ypred.argmax(axis=1)
            # ypred = (torch.sigmoid(ypred)>0.5)
            acc_total.addBatch(ypred[y_true!=255], y_true[y_true!=255])

            losses.update(loss.item(), x.size(0))
            oa = acc_total.OverallAccuracy()
            iou = acc_total.IntersectionOverUnion()
            miou = acc_total.meanIntersectionOverUnion()
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU {miou:.3f}, '
                    'IOU: {bg:.3f}, {ESB:.3f},{GPC:.3f}, {build:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou,
                    bg=iou[0], ESB=iou[1], GPC=iou[2],build=iou[3]))
            pbar.update()
        pbar.close()

    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou


if __name__ == "__main__":
    main()