import cv2
import numpy as np
import math
from sklearn.cluster import KMeans


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def remove_spice_noise(image, kernel_size=3):
    image = cv2.medianBlur(image, kernel_size)
    return image


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


def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_ = cv2.filter2D(img, -1, kernel)
    return img_


def remove_shadow(img, dilate_kernel, median_blur_kernel):
    rgb_planes = cv2.split(img)

    result_planes = []
    # result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((dilate_kernel, dilate_kernel), np.uint8), 5)
        bg_img = cv2.medianBlur(dilated_img, median_blur_kernel)
        # diff_img = 255 - cv2.absdiff(plane, bg_img)
        # norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(bg_img)
        # result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    # result_norm = cv2.merge(result_norm_planes)

    return result


def hist_match(sources, templates):
    rgb_source = cv2.split(sources)
    rgb_template = cv2.split(templates)
    result = []
    for i, source in enumerate(rgb_source):
        oldshape = source.shape
        source = source.ravel()
        template = rgb_template[i]
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        result.append(interp_t_values[bin_idx].reshape(oldshape))

    return cv2.merge(result)


def kmeans(points, n_cluster=4):
    cluster = KMeans(n_clusters=n_cluster, random_state=0).fit(points)
    return cluster.labels_


def refine_hull(hull_points, front=True):
    if len(hull_points) == 4:
        return hull_points
    elif front:
        clusters = kmeans(hull_points[:, 0, :], n_cluster=4)
        dict = {}
        for i in np.unique(clusters):
            dict[i] = []
        for i, p in enumerate(hull_points):
            label = clusters[i]
            dict[label].append(p)
        keep = []
        for k in dict.keys():
            curr_cluster = dict[k]
            if len(curr_cluster) == 1:
                keep.append(curr_cluster[0])
                continue
            else:
                max_dis = 0
                other_points = [v for key, v in dict.items() if key != k]
                candidate = None
                for p1 in curr_cluster:
                    curr_dis = 0
                    for p2 in other_points:
                        curr_dis += calculate_distance(p1[0], p2[0][0])

                    if curr_dis > max_dis:
                        max_dis = curr_dis
                        candidate = p1
                keep.append(candidate)
        return order_points(np.array(keep)[:, 0, :])



def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def refine_edge(img):
    h, w = img.shape[-2], img.shape[-1]
    max_pix = np.max(img)
    small = (np.sum(img) / (h * w) / max_pix * 100) < 30
    if small:
        y_min = int(512 * 0.25)
        y_max = int(512 * 0.75)
        x_min = int(512 * 0.25)
        x_max = int(512 * 0.75)
    else:
        y_min = int(512 * 0.1)
        y_max = int(512 * 0.9)
        x_min = int(512 * 0.1)
        x_max = int(512 * 0.9)

    img[:, :y_min] = remove_noise2(img[:, :y_min], kernel_size=(5, 1), iterations=5)
    img[:, y_max:] = remove_noise2(img[:, y_max:], kernel_size=(5, 1), iterations=5)

    img[:x_min, :] = remove_noise2(img[:x_min, :], kernel_size=(1, 5), iterations=5)
    img[x_max:, :] = remove_noise2(img[x_max:, :], kernel_size=(1, 5), iterations=5)

    return img


def calculate_distance(p1, p2):
    distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    return distance


def calulate_edges(rect):
    assert len(rect) == 4, "Invalid number of points for input rectangle"
    edge1 = calculate_distance(rect[0], rect[1])
    edge2 = calculate_distance(rect[1], rect[2])

    return min(edge1, edge2), max(edge1, edge2)


def subimage(image, center, theta, width, height):
    '''
    Rotates OpenCV image around center with angle theta (in deg)
    then crops the image according to width and height.
    '''

    # Uncomment for theta in radians
    # theta *= 180/np.pi

    shape = (image.shape[1], image.shape[0])  # cv2.warpAffine expects shape in (length, height)

    matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

    x = int(center[0] - width / 2)
    y = int(center[1] - height / 2)

    image = image[y:y + height, x:x + width]

    return image


def convert(width, length, mode='length'):
    result = None
    if mode == 'length':
        if length <= 23.5:
            result = ['6', '5.5', '39']
        elif length <= 24.1:
            result = ['6.5', '6', '39']
        elif length <= 24.4:
            result = ['7', '6.5', '40']
        elif length <= 24.8:
            result = ['7.5', '7', '40 - 41']
        elif length <= 25.4:
            result = ['8', '7.5', '41']
        elif length <= 25.7:
            result = ['8.5', '8', '41 - 42']
        elif length <= 26.0:
            result = ['9', '8.5', '42']
        elif length <= 26.7:
            result = ['9.5', '9', '42 - 43']
        elif length <= 27.0:
            result = ['10', '9.5', '43 - 44']
        elif length <= 27.3:
            result = ['10.5', '10', '44 - 44.5']
        elif length <= 27.9:
            result = ['11', '10.5', '44 - 45']
        elif length <= 28.3:
            result = ['11.5', '11', '44 - 45']
        elif length <= 28.6:
            result = ['12', '11.5', '45 - 46']
        elif length <= 29.4:
            result = ['13', '12.5', '46 - 47']
        elif length <= 30.2:
            result = ['14', '13.5', '47 - 48']
        elif length <= 31:
            result = ['15', '14.5', '48 - 49']

    else:
        if width <= 9.8:
            result = ['5.5', '5', '38']
        elif width <= 10:
            result = ['6', '5.5', '39']
        elif width <= 10.2:
            result = ['7', '6.5', '40']
        elif width <= 10.4:
            result = ['8', '7.5', '41']
        elif width <= 10.6:
            result = ['9', '8.5', '42']
        elif width <= 10.8:
            result = ['10', '9.5', '43']
        elif width <= 11:
            result = ['11', '10.5', '44']

    return result
