import os

import torch
from torch.utils.tensorboard import SummaryWriter
from models.segformer import SegFormer
from dataset.feet_dataset import FeetDataset
from utils.criterion import MyCriterion
from utils.functions import train_one_epoch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
import numpy as np


def main():
    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.enabled = True
    checkpoint_path = 'logs/run2'
    writer_dict = {
        'writer': SummaryWriter(checkpoint_path),
        'steps': 0
    }

    model = SegFormer(backbone='MiT-B1', num_classes=3)

    train_set = FeetDataset(root_path='Images',
                            img_path='images',
                            mask_path='label_images',
                            img_size=(512, 512),
                            num_classes=3,
                            use_aug=True,
                            train=True)

    train_loader = DataLoader(train_set,
                              batch_size=16,
                              num_workers=8,
                              shuffle=True,
                              drop_last=False,
                              pin_memory=True)

    criterion = MyCriterion(0.5, 0.5)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = DataParallel(model).to(device)
    optimizer = AdamW(model.parameters(),
                      lr=1e-4,
                      weight_decay=1e-5,
                      # betas = (0.9, 0.9)
                      )
    # optimizer = SGD(model.parameters(),
    #                 lr = 1e-3,
    #                 momentum=0.9,
    #                 weight_decay=1e-5,
    #                 nesterov=True)
    num_epoch = 100
    scheduler = CosineAnnealingLR(optimizer=optimizer,
                                  T_max=(num_epoch-10) * len(train_loader),
                                  eta_min=1e-6)
    best_score = 0

    for epoch in range(num_epoch):
        train_bce, train_tversky, train_iou, train_dice = train_one_epoch(model=model,
                                                                          current_epoch=epoch,
                                                                          max_epoch=num_epoch,
                                                                          train_loader=train_loader,
                                                                          num_classes=3,
                                                                          criterion=criterion,
                                                                          optimizer=optimizer,
                                                                          scheduler=scheduler,
                                                                          writer_dict=writer_dict,
                                                                          device=device,
                                                                          weight_loss=[2/3, 1/3]
                                                                          )
        torch.cuda.empty_cache()

        checkpoint = {
            "epoch": epoch,
            "iou": train_iou,
            "dice": train_dice,
            "state_dict": model.module.state_dict()
        }

        if (np.mean(train_iou) + np.mean(train_dice)) / 2 > best_score:
            best_score = (np.mean(train_iou) + np.mean(train_dice)) / 2
            torch.save(checkpoint, os.path.join(checkpoint_path, 'checkpoint_best.pt'))
            print("-----> save new best")

        tags = ['train/BCE', 'train/Tversky', 'train/IoU', 'train/dice']
        values = [train_bce, train_tversky, np.mean(train_iou), np.mean(train_dice)]

        for tag, value in zip(tags, values):
            writer_dict['writer'].add_scalar(tag, value, epoch)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    main()
