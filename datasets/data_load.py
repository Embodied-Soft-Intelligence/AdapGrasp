import json
from pathlib import Path
import random
import os

import torch
import torch.utils.data
import torchvision
from datasets.data_util import preparing_dataset
import datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from PIL import Image

class CustomDetection(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_folder, transforms=None, return_masks=False):
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.transforms = transforms
        self.return_masks = return_masks
        self.img_ids = sorted(os.listdir(img_folder))  
        self.image_files = [f for f in os.listdir(img_folder) if f.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_folder, img_id)
        json_path = os.path.join(self.ann_folder, img_id.replace('.png', '.json'))

        img = Image.open(img_path).convert("RGB")

        with open(json_path, 'r') as f:
            annotations = json.load(f)
        image_id = torch.tensor(int(img_id[:4]))
        labels = []
        boxes = []
        width_es = []
        levels = []
        angles = []
        for obj_id, obj_annotations in annotations.items():
            for annotation in obj_annotations:
                x, y, width_i, width_e, height, angle, rank = annotation
                index = int(obj_id[:-1]) if  obj_id.endswith('a') else int(obj_id) 
                label = index
                level = rank / 4.0 - 0.125
                labels.append(label)
                boxes.append([x, y, width_i, height]) 
                width_es.append(width_e)  
                levels.append(level)
                angles.append(angle)

        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        width_es = torch.tensor(width_es, dtype=torch.float32)
        levels = torch.tensor(levels, dtype=torch.float32)
        angles = torch.tensor(angles, dtype=torch.float32)

        target = {
            "image_id":image_id,
            "labels": labels,
            "boxes": boxes,
            "width_es": width_es,
            "levels": levels,
            "angles": angles,
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


def make_data_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        
        return T.Compose([
            T.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')



def build(image_set, args):
    root = Path(args.data_path)
    mode = 'instances'
    PATHS = {
        "train": (root / "train_img", root / "train_label"),
        "val": (root / "val_img", root / "val_label"),
    }

    img_folder, ann_file = PATHS[image_set]
    transforms = make_data_transforms(image_set)

    dataset = CustomDetection(
            img_folder, 
            ann_file, 
            transforms=transforms, 
            return_masks=args.masks
        )

    return dataset