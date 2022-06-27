import os

from models.segformer import SegFormer
from dataset.feet_dataset import FeetDataset
import torch
from torch.nn import DataParallel, functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.nn.functional import one_hot
import warnings

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = SegFormer('MiT-B1', num_classes=1)

    test_set = FeetDataset(
        root_path='Images',
        img_path='train',
        mask_path='trainannot',
        num_classes=2,
        img_size=(512, 512),
        use_aug=False,
        train=True
    )
    print(f"reading {len(test_set)} images")

    checkpoint = torch.load('logs/run3/checkpoint_best.pt', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = DataParallel(model).to(device)

    model.eval()
    dir = 'inference'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    for i in range(len(test_set)):
        img, _ = test_set[i]
        img_ = img.unsqueeze(0).to(device)

        pred = model(img_)
        pred = torch.sigmoid(pred)
        pred = torch.where(pred > 0.7, 1, 0)
        pred = pred * 255.0 / torch.max(pred)
        fig, ax = plt.subplots(1, 2)
        img = FeetDataset.denormalize(img.permute(1, 2, 0)).permute(2, 0, 1)
        # prediction = Image.fromarray(pred)
        save_image(img, os.path.join(dir, f"img{i}.png"))
        save_image(pred, os.path.join(dir, f"mask{i}.png"))

        # prediction.save(os.path.join(dir, f"prediction{i}.png"))
