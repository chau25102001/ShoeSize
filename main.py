import streamlit as st
import math
import os

import cv2
import imutils

from models.segformer import SegFormer
from dataset.feet_dataset import FeetDataset
import torch
from torch.nn import DataParallel
import matplotlib.pyplot as plt
from albumentations.augmentations.geometric import rotate, resize
import torchvision.transforms as tf
from post_process import *


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = SegFormer('MiT-B0', num_classes=1)
    trans_resize = resize.Resize(512, 512)
    img_transform = tf.Compose([
        tf.ToTensor(),
        tf.Normalize(FeetDataset.mean, FeetDataset.std)
    ])

    # B0: run1, run4
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def load_model():
        return torch.load('logs/run4/checkpoint_best.pt', map_location=device)

    checkpoint = load_model()
    model.load_state_dict(checkpoint['state_dict'])
    model = DataParallel(model).to(device)

    model.eval()
    papper_size = [21.0, 29.7]
    st.title("Shoe size measuring project")
    st.header("Take a front view picture of one of your feet stepping on an A4 paper.")
    uploaded_file = st.file_uploader('Upload your foot picture here')
    print(uploaded_file)
    if not (uploaded_file is None):
        fig, ax = plt.subplots(2, 4, figsize=(10, 5))
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        img_h = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_h = cv2.cvtColor(img_h, cv2.COLOR_BGR2RGB)
        ax[0, 0].imshow(img_h)

        # img = hist_match(img, template).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = remove_shadow(img, 7, 7)
        ax[0, 1].imshow(cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR))
        img = apply_brightness_contrast(img, 15, -25)
        ax[0, 2].imshow(cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR))
        img = sharpen(img)
        ax[0, 3].imshow(cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR))

        img = trans_resize(image=img)['image']
        img = img_transform(img).unsqueeze(0).to(device)

        pred = model(img)
        pred = torch.sigmoid(pred)
        pred = torch.where(pred > 0.7, 1, 0)
        pred = pred * 255.0 / torch.max(pred)
        pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, 0]
        ax[1, 0].imshow(pred, cmap='gray')
        pred = refine_edge(pred)

        pred = remove_noise2(pred, kernel_size=(3, 3), iterations=2).astype(np.uint8)
        contours = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = imutils.grab_contours(contours)
        cnt = max(cnt, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        paper = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
        hull = cv2.convexHull(approx)
        hull = refine_hull(hull, True)
        hull = np.array([[i] for i in hull]).astype(np.int32)
        if len(hull.shape) == 4:
            hull = hull[:, 0, :, :]
        ax[1, 1].imshow(cv2.drawContours(paper.copy(), [hull], -1, (0, 255, 0), 3))
        warp = four_point_transform(pred, hull[:, 0, :])
        foot = 255 - warp
        if foot.shape[0] < foot.shape[1]:
            foot = np.transpose(foot)
        foot_contour = cv2.findContours(
            foot[int(foot.shape[0] * 0.1):int(foot.shape[0] * 0.9), int(foot.shape[1] * 0.1):int(foot.shape[1] * 0.9)],
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnt = imutils.grab_contours(foot_contour)
        cnt = max(cnt, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        rect = cv2.minAreaRect(approx)
        box = np.int0(cv2.boxPoints(rect))
        for p in box:
            p[1] += int(foot.shape[0] * 0.1)
            p[0] += int(foot.shape[1] * 0.1)

        ax[1, 2].imshow(cv2.drawContours(cv2.cvtColor(warp.copy(), cv2.COLOR_GRAY2RGB), [box], -1, (0, 255, 0), 3))

        center = ((box[0] + box[2]) // 2).astype(np.uint8)
        angle = math.radians(max(rect[-1], 90 - rect[-1]))
        edge1, edge2 = calulate_edges(box)
        theta = rect[-1] if rect[-1] < 45 else rect[-1] - 90
        st.text(foot.shape)
        foot_box = subimage(foot, center, theta, int(edge1), int(edge2))
        foot_box = np.where(foot_box > 127, 1, 0).astype(np.uint8)
        foot_box = remove_noise1(foot_box, (3, 3), 7)
        ax[1, 3].imshow(foot_box, cmap='gray')
        if foot_box.shape[0] > foot_box.shape[1]:
            edge1 = np.sum(foot_box[int(foot_box.shape[0] * 0.5), :])
        else:
            edge1 = np.sum(foot_box[:, int(foot_box.shape[1] * 0.5)])

        horizontal_proj1 = edge1 * math.sin(angle)
        vertical_proj1 = edge1 * math.cos(angle)

        horizontal_proj2 = edge2 * math.cos(angle)
        vertical_proj2 = edge2 * math.sin(angle)
        scale_size = warp.shape[-1], warp.shape[-2]
        scale_horizontal = papper_size[0] / scale_size[0]
        scale_vertical = papper_size[1] / scale_size[1]
        width = math.sqrt((horizontal_proj1 * scale_horizontal) ** 2 + (vertical_proj1 * scale_vertical) ** 2)
        length = math.sqrt((horizontal_proj2 * scale_horizontal) ** 2 + (vertical_proj2 * scale_vertical) ** 2)
        factor = 1.02
        length = length * factor + 1.5
        st.pyplot(fig)

        st.text(f"Width: {round(width, 2)}, Length: {round(length, 2)}, Angle: {round(rect[-1], 2)}")
        size_length = convert(width, length, mode='length')
        size_width = convert(width, length, mode='width')

        if size_length:
            st.text(f"Shoe size is: US {size_length[0]}, UK {size_length[1]}, EU {size_length[2]}")
        elif size_width:
            st.text(f"Shoe size is: US {size_width[0]}, UK {size_width[1]}, EU {size_width[2]}")
        else:
            st.text("Cannot measure, please take another picture!")


if __name__ == "__main__":
    main()
