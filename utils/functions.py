import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import AverageMeter
import numpy as np


def iou(iou_track, pred, mask, num_classes=1, smooth=1e-5, thres=0.5):
    assert (pred.shape[-1] == mask.shape[-1]) & (
            pred.shape[-2] == mask.shape[-2]), "Predicted mask and Ground Truth must be at the same size"
    if num_classes > 1:
        pred = torch.argmax(pred, dim=1, keepdim=True)
        mask = torch.argmax(mask, dim=1, keepdim=True)
        for i in range(len(pred)):
            p = pred[i]
            m = mask[i]
            for c in range(num_classes):
                p_c = torch.where(p == c, 1, 0)
                m_c = torch.where(m == c, 1, 0)
                inter = torch.sum((p_c > 0) & (m_c > 0)).item()
                uni = torch.sum((p_c > 0) | (m_c > 0)).item()
                iou_track[c].append(inter / (uni + smooth))
    else:
        pred = torch.sigmoid(pred)
        pred = torch.where(pred > thres, 1, 0)
        for i, seg in enumerate(pred):
            intersection = torch.sum((seg > 0) & (mask[i] > 0))
            union = torch.sum((seg > 0) | (mask[i] > 0))
            x = intersection / (union + smooth)
            iou_track.append(x.item())

    return iou_track


def dice(dice_track, pred, mask, num_classes=1, smooth=1e-5, thres=0.5):
    assert (pred.shape[-1] == mask.shape[-1]) & (
            pred.shape[-2] == mask.shape[-2]), "Predicted mask and Ground Truth must be at the same size"
    if num_classes > 1:
        pred = torch.argmax(pred, dim=1, keepdim=True)
        mask = torch.argmax(mask, dim=1, keepdim=True)
        for i in range(len(pred)):
            p = pred[i]
            m = mask[i]
            for c in range(num_classes):
                p_c = torch.where(p == c, 1, 0)
                m_c = torch.where(m == c, 1, 0)
                inter = torch.sum((p_c > 0) & (m_c > 0)).item()
                uni = torch.sum((p_c > 0) | (m_c > 0)).item()
                dice_track[c].append(2 * inter / (uni + inter + smooth))
    else:
        pred = torch.sigmoid(pred)
        pred = torch.where(pred > thres, 1, 0)
        for i, seg in enumerate(pred):
            intersection = torch.sum((seg > 0) & (mask[i] > 0))
            union = torch.sum((seg > 0) | (mask[i] > 0))
            x = 2 * intersection / (intersection + union + smooth)
            dice_track.append(x.item())

    return dice_track


def train_one_epoch(
        model,
        current_epoch,
        max_epoch,
        train_loader,
        num_classes,
        criterion,
        optimizer,
        scheduler,
        writer_dict,
        device,
        weight_loss=[0.5, 0.5]
):
    pdar = tqdm(train_loader, desc=f"Training {current_epoch + 1}/{max_epoch}", leave=True)

    bce_meter = AverageMeter()
    tversky_meter = AverageMeter()
    iou_track = [[] for _ in range(num_classes)]
    dice_track = [[] for _ in range(num_classes)]
    if num_classes == 1:
        iou_track = []
        dice_track = []
    model.train()
    writer = writer_dict['writer']
    steps = writer_dict['steps']
    running_iou = 0
    running_dice = 0
    for batch in pdar:
        image, mask = batch
        image, mask = image.to(device), mask.to(device)

        pred = model(image)
        bce, tversky = criterion(pred, mask)

        # loss = weight_loss[0] * bce + weight_loss[1] * tversky
        loss = bce
        optimizer.zero_grad()


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        if writer:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], steps)
        if scheduler:
            scheduler.step()

        iou_track = iou(iou_track, pred, mask, num_classes, thres=0.6)
        dice_track = dice(dice_track, pred, mask, num_classes, thres=0.6)
        bce_meter.update(bce.item())
        tversky_meter.update(tversky.item())

        steps += 1
        writer_dict['steps'] = steps
        if num_classes > 1:
            running_iou = np.min(np.mean(iou_track, axis=1))
            running_dice = np.min(np.mean(dice_track, axis=1))
        else:
            running_iou = np.mean(iou_track)
            running_dice = np.mean(dice_track)
        pdar.set_postfix({
            'struct_loss': bce_meter.avg,
            'Tversky': tversky_meter.avg,
            'iou': running_iou,
            'dice': running_dice
        })

    return bce_meter.avg, tversky_meter.avg, running_iou, running_dice
