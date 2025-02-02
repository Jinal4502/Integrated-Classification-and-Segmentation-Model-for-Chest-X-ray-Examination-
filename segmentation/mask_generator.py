import json
import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def create_mask(image_path, boxes, polygons, save_dir, file_name):
    image = Image.open(image_path)
    width, height = image.size
    mask = np.zeros((height, width), dtype=np.uint8)

    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 1
    
    #will for polygon in tqdm(polygons):
    #    polygon = np.array(polygon, dtype=np.int32)
    #    cv2.fillPoly(mask, [polygon], color=200)
    os.makedirs(save_dir, exist_ok=True)
    mask_path = os.path.join(save_dir, file_name.replace(".png", "_mask.png"))
    mask_image = Image.fromarray(mask)
    mask_image.save(mask_path)
    return mask

def load_data(annotation_file, image_folder, mask_save_dir):
    dict_labels = {}
    with open(annotation_file) as f:
        annotations = json.load(f)
    images = []
    masks = []
    for annotation in tqdm(annotations):
        image_path = os.path.join(image_folder, annotation['file_name'])
        mask = create_mask(image_path, annotation['boxes'], annotation['polygons'], mask_save_dir, annotation['file_name'])
        images.append(image_path)
        dict_labels[annotation['file_name']] = annotation["syms"]
        masks.append(os.path.join(mask_save_dir, annotation['file_name'].replace(".png", "_mask.png")))
    return images, masks, dict_labels
    #return dict_labels

annotation_file = "/scratch/jjvyas1/segmentation/chestxdet/ChestX_Det_train.json"
image_folder = "/scratch/jjvyas1/segmentation/chestxdet/train/"
mask_save_dir = "/scratch/jjvyas1/segmentation/chestxdet/train/mask"

images, masks, dict_labels = load_data(annotation_file, image_folder, mask_save_dir)
#print(len(images))

#dict_labels = load_data(annotation_file, image_folder, mask_save_dir)
#print(dict_labels)

with open("dict_labels.txt", "w") as file:
    json.dump(dict_labels, file)
