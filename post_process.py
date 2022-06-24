import os
import cv2
import matplotlib.pyplot as plt
import numpy as np



def remove_noise1(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size)
    image = cv2.dilate(image, kernel, iterations=iterations)
    image = cv2.erode(image, kernel, iterations=iterations)

    return image


def remove_noise2(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size)
    image = cv2.erode(image, kernel, iterations=iterations)
    image = cv2.dilate(image, kernel, iterations=iterations)

    return image


def near(p1, p2, radius):
    dis = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    if dis < radius ** 2:
        return True
    return False


def refine_hull(hull, radius):
    result = []
    elem = [hull[0]]
    for i in range(1, len(hull)):
        curr_point = hull[i]
        prev_point = hull[i - 1]
        if near(curr_point[0], prev_point[0], radius):
            elem.append(curr_point)
        else:
            elem = np.mean(np.array(elem), axis=0, dtype=np.int32)
            result.append(elem)
            elem = [curr_point]
    return np.array(result)

if __name__ == '__main__':
    dir = 'inference/mask36.png'
    mask = cv2.imread(dir)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask = remove_noise1(mask, (10, 10), iterations=5)
    # mask = remove_noise2(mask, (10, 10), iterations=5)

    mask = remove_noise1(mask, (5, 2), iterations=5)
    mask = remove_noise2(mask, (5, 2), iterations=5)

    classes = sorted(np.unique(mask))
    paper = np.where(mask == classes[0], 1, 0).astype(np.uint8)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(paper, cmap='gray')
    # print(paper.shape)
    contours, hierarchy = cv2.findContours(paper, 1, 2)
    pred_len = 0
    cnt = contours[0]
    for c in contours:
        if len(c) > pred_len:
            pred_len = len(c)
            cnt = c
    epsilon = 0.015 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    paper = cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)
    hull = cv2.convexHull(approx)
    print(hull)
    approx_ed = cv2.drawContours(paper, [hull], -1, (0, 255, 0), 3)
    ax[1].imshow(approx_ed)

    approx_ed = cv2.approxPolyDP(hull, 0.08 * cv2.arcLength(hull, True), True)
    ax[2].imshow(cv2.drawContours(np.zeros_like(paper), [approx_ed], -1, (0, 255, 0), 3))
    fig.show()
    # hull = refine_hull(hull, 50)
    # print(hull)
    # ax[2].imshow(cv2.drawContours(np.zeros_like(paper), [hull], -1, (0, 255, 0), 3))
    # plt.show()
