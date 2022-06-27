import os

import albumentations.augmentations
import cv2
import numpy as np

from torch.utils.data import Dataset
from albumentations.augmentations import transforms, crops
import albumentations.augmentations as A
from albumentations.augmentations.geometric import rotate, resize
from albumentations.core.composition import Compose, OneOf

import torchvision.transforms as tf

image_dir = r'C:\Users\chau\Desktop\ShoeSizeProject\Images\train'
mask_dir = r'C:\Users\chau\Desktop\ShoeSizeProject\Images\trainannot'


class FeetDataset(Dataset):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self,
                 root_path,
                 img_path,
                 mask_path,
                 img_size,
                 train,
                 num_classes,
                 use_aug):
        image_dir = os.path.join(root_path, img_path)
        mask_dir = os.path.join(root_path, mask_path)

        images = sorted(os.listdir(image_dir))
        masks = sorted(os.listdir(mask_dir))

        image_name = [i.split('.')[0] for i in images]
        mask_name = [i.split('.')[0] for i in masks]

        image_format = [i.split('.')[1] for i in images]
        mask_format = [i.split('.')[1] for i in masks]

        inter = [i for i in image_name if i in mask_name]
        self.num_classes = num_classes
        self.items = []
        for i in range(len(inter)):
            img = os.path.join(image_dir, inter[i] + f".{image_format[i]}")
            mask = os.path.join(mask_dir, inter[i] + f".{mask_format[i]}")
            m = cv2.imread(mask)
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            if len(np.unique(m)) == self.num_classes:
                self.items.append((img, mask))

        self.img_size = img_size
        self.use_aug = use_aug
        self.train = train

    def transform(self, image, mask, h, w, use_aug, train):
        train_transform = Compose([
            rotate.RandomRotate90(p=0.5),
            # transforms.Flip(p=0.5),
            transforms.HorizontalFlip(p=0.5),
            # transforms.HueSaturationValue(p=0.5),
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            transforms.GaussianBlur(p=0.5),
            transforms.GaussNoise(p=0.5),
            # transforms.RandomSnow(p=0.5),
            transforms.RandomShadow(shadow_roi=(0, 0.3, 1, 1), num_shadows_lower=1, num_shadows_upper=3,
                                    shadow_dimension=8, p=1),
            transforms.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            # transforms.ColorJitter(p=0.5),
            # transforms.Transpose(p=0.5),
            # transforms.ChannelShuffle(p=0.5)
        ], p=0.6)
        trans_resize = resize.Resize(h, w)
        img_transform = tf.Compose([
            tf.ToTensor(),
            tf.Normalize(FeetDataset.mean, FeetDataset.std)
        ])
        label_transform = tf.Compose([
            tf.ToTensor()
        ])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if use_aug:
            out = train_transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        if train:
            out = trans_resize(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        classes = sorted(np.unique(mask))

        assert len(classes) == self.num_classes, "num classes doesn't match"

        mask = np.stack([np.where(mask == c, 1, 0) for c in classes])

        image = img_transform(image)
        mask = label_transform(mask)
        return image, mask.float()

    def __getitem__(self, item):
        image, mask = self.items[item]
        image = cv2.imread(image)
        mask = cv2.imread(mask)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_NEAREST)
        # mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.uint8)
        mask = mask.astype(np.uint8)
        image, mask = self.transform(image, mask, self.img_size[0], self.img_size[1], self.use_aug,
                                     train=self.train)

        return image, mask.permute(1, 2, 0)[-1].unsqueeze(0)

    @staticmethod
    def denormalize(image):
        image *= FeetDataset.std
        image += FeetDataset.mean
        return image

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import torch

    dataset = FeetDataset(root_path=r'C:\Users\chau\Desktop\ShoeSizeProject\Images',
                          img_path='images',
                          mask_path='label_images',
                          img_size=(512, 512),
                          train=True,
                          num_classes=3,
                          use_aug=True)
    for i in range(10):
        img, mask = dataset[i]
        print(torch.unique(mask))

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(FeetDataset.denormalize(img.permute(1, 2, 0)))
        ax[1].imshow(mask.permute(1, 2, 0), cmap='gray')
        plt.show()
