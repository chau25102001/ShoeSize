import os

import cv2

from models.segformer import SegFormer
from dataset.feet_dataset import FeetDataset
import torch
from torch.nn import DataParallel, functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.nn.functional import one_hot
from albumentations.augmentations import transforms, crops
from albumentations.augmentations.geometric import rotate, resize
import torchvision.transforms as tf
from post_process import remove_noise1, remove_noise2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = SegFormer('MiT-B1', num_classes=3)

trans_resize = resize.Resize(512, 512)
img_transform = tf.Compose([
    tf.ToTensor(),
    tf.Normalize(FeetDataset.mean, FeetDataset.std)
])

checkpoint = torch.load('logs/run2/checkpoint_best.pt', map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model = DataParallel(model).to(device)

model.eval()

img = cv2.imread(r"C:\Users\chau\Desktop\ShoeSizeProject\Images\chan3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = trans_resize(image=img)['image']
img = img_transform(img).unsqueeze(0).to(device)

pred = model(img)
pred = one_hot(torch.argmax(pred, dim=1)).float().squeeze(0).permute(2, 0, 1)
pred = pred * 255.0 / torch.max(pred)
pred = pred.permute(2, 1, 0).detach().cpu().numpy()
pred = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
fig, ax = plt.subplots(1,2)
ax[0].imshow(pred)
pred = remove_noise2(pred, kernel_size=(2, 5), iterations=10)
pred = remove_noise1(pred, kernel_size=(2, 5), iterations=10)
ax[1].imshow(pred)
fig.show()

# prediction.save(os.path.join(dir, f"prediction{i}.png"))
