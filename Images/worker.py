from PIL import Image, ImageDraw
from sys import argv
from os import mkdir
from os.path import isdir, join
import json


IMAGES_DIRECTORY = 'images'
METADATA_DIRECTORY = 'metadata'
MASK_IMAGES_DIRECTORY = 'mask_images'
LABEL_IMAGES_DIRECTORY = 'label_images'


def prepare_directories():
    if (not isdir(MASK_IMAGES_DIRECTORY)):
        mkdir(MASK_IMAGES_DIRECTORY)
    if (not isdir(LABEL_IMAGES_DIRECTORY)):
        mkdir(LABEL_IMAGES_DIRECTORY)


def handle_image(image_id):
    image = Image.open(join(IMAGES_DIRECTORY, image_id + '.jpeg'))
    image_size = image.size
    width, height = image_size
    metadata = json.load(
        open(join(METADATA_DIRECTORY, image_id + '.json'), encoding='UTF-8'))

    mask_image = Image.new('1', image_size)
    mask_draw = ImageDraw.Draw(mask_image)
    label_image = Image.new('P', image_size, '#000')
    label_draw = ImageDraw.Draw(label_image)

    colors_to_label_names = dict()

    for item in metadata['region_list']:
        vertices = []
        for v in item['border']['vertices']:
            vertices.append((v['x'] * width, v['y'] * height))
        mask_draw.polygon(vertices, fill=1, outline=1)
        if (item['label'] is not None):
            item_color = item['label']['color']
            item_label_name = item['label']['display_name']
            if item_label_name == 'Paper':
                label_draw.polygon(vertices, fill='#FFFFFF', outline=1)
                colors_to_label_names[item_color] = item_label_name
            if item_label_name == 'Foot':
                label_draw.polygon(vertices, fill='#888888', outline=1)
                colors_to_label_names[item_color] = item_label_name

    mask_image.save(join(MASK_IMAGES_DIRECTORY, image_id + '.png'))
    label_image.save(join(LABEL_IMAGES_DIRECTORY, image_id + '.png'))
    return colors_to_label_names


if __name__ == '__main__':
    prepare_directories()
    image_id = argv[1]
    print(handle_image(image_id))
