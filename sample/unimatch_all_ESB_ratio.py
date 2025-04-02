import argparse
import logging
import os
import math
import pprint
import cv2
import torch
from torch import nn
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4"
from torch.utils.data import DataLoader
IMG_MEAN_ALL_Ge = np.array([109.9142, 105.5509, 103.6405])
IMG_STD_ALL_Ge = np.array([57.0574, 50.5545, 48.2372])
import yaml
from tqdm import tqdm
from model.Unet import Unet
from dataset.semi import SemiDataset, myImageFloder_labeled_neg, myImageFloder_unlabeled
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, default='./configs/jianshe.yaml')
parser.add_argument('--labeled-id-path', type=str, default=r"./data/list_jianshe.txt")
parser.add_argument('--label_jianshe',
                    default=r"Z:\data_for_dataset\data\jianshe", type=str,
                    help='label root')
parser.add_argument('--unlabeled-id-path', type=str, default=r"Y:\Training_Data\ESB")
parser.add_argument('--save-path', type=str, default='./runs_ESB_50')
parser.add_argument('--unlabeled-ratio', default=8, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--warmup', default=0, type=float,
                    help='warmup epochs (unlabeled data based)')
parser.add_argument('--wdecay', default=5e-4, type=float,
                    help='weight decay')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov momentum')

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    os.makedirs(args.save_path,exist_ok=True)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    cudnn.enabled = True
    cudnn.benchmark = True

    nchannels = 3
    # model = DeepLabV3Plus(cfg)
    model = Unet(encoder_name="timm-regnety_040", encoder_weights="imagenet",
                     in_channels=nchannels, classes=cfg['nclass']).cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.SGD(grouped_parameters, lr=cfg['lr'],
                          momentum=0.9, nesterov=args.nesterov)

    model.cuda()
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))


    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()


    labeled_dataset = myImageFloder_labeled_neg(datalist=args.labeled_id_path, labelroot=args.label_jianshe, )
    unlabeled_dataset = myImageFloder_unlabeled(data_dir=args.unlabeled_id_path, )

    trainloader_l = DataLoader(
        labeled_dataset,
        batch_size=cfg['batch_size'] * torch.cuda.device_count(),
        num_workers=4,
        shuffle=True,
        drop_last=True)

    trainloader_u = DataLoader(
        unlabeled_dataset,
        batch_size=cfg['batch_size'] * args.unlabeled_ratio * torch.cuda.device_count(),
        num_workers=4,
        shuffle=True,
        drop_last=True)
    args.total_steps = cfg['epochs']*len(trainloader_l)/cfg['batch_size']
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    loader_iter1 = iter(trainloader_u)
    loader_iter2 = iter(trainloader_u)
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()
        for i, (img_x, mask_x) in enumerate(trainloader_l):
            try:
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2) = loader_iter1.next()
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _) = loader_iter2.next()
            except:
                loader_iter1 = iter(trainloader_u)
                loader_iter2 = iter(trainloader_u)
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2) = loader_iter1.next()
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _) = loader_iter2.next()
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())
            
            logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}, Lr:{:}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                            total_loss_w_fp.avg, total_mask_ratio.avg, scheduler.get_last_lr()[0]))

        checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
        torch.save(checkpoint, os.path.join(args.save_path, str(epoch)+'_latest.pth'))


if __name__ == '__main__':
    main()
