import json
import os
from collections import defaultdict

import cv2
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from data.preprocessing import get_train_transform, get_val_transform
from utils.channel_convert import make_pseudo_rgb



class COCODetectionDataset(Dataset):
    def __init__(self, images_dir: str, annotation_file: str, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform

        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco = json.load(f)

        self.image_id_to_filename = {
            img['id']: img['file_name'] for img in coco['images']}
        self.image_id_to_size = {img['id']: (
            img['width'], img['height']) for img in coco['images']}

        self.annotations = defaultdict(list)
        for ann in coco['annotations']:
            self.annotations[ann['image_id']].append(ann)

        self.image_ids = list(self.image_id_to_filename.keys())
        self.category_id_to_name = {
            cat['id']: cat['name'] for cat in coco['categories']}

        self._filter_invalid_samples()

    def __len__(self):
        return len(self.image_ids)

    def _filter_invalid_samples(self):
        valid_image_ids = []
        dropped_image_ids = []

        for image_id in self.image_ids:
            anns = self.annotations.get(image_id, None)
            if anns is None:
                dropped_image_ids.append(image_id)
                continue

            has_invalid_annotation = False
            image_size = self.image_id_to_size.get(image_id)
            img_width, img_height = image_size if image_size else (None, None)

            for ann in anns:
                bbox = ann.get('bbox')
                label = ann.get('category_id')

                if bbox is None or label is None:
                    has_invalid_annotation = True
                    break

                if len(bbox) != 4:
                    has_invalid_annotation = True
                    break

                if any(coord is None for coord in bbox):
                    has_invalid_annotation = True
                    break

                x, y, w, h = bbox

                if w <= 0 or h <= 0:
                    has_invalid_annotation = True
                    break

                if x is None or y is None or x < 0 or y < 0:
                    has_invalid_annotation = True
                    break

                if img_width is not None and x + w > img_width:
                    has_invalid_annotation = True
                    break

                if img_height is not None and y + h > img_height:
                    has_invalid_annotation = True
                    break

            if has_invalid_annotation:
                dropped_image_ids.append(image_id)
                del self.annotations[image_id]
                continue

            valid_image_ids.append(image_id)

        if dropped_image_ids:
            print(
                f"Filtered out {len(dropped_image_ids)} samples due to invalid bbox or labels.")

        self.image_ids = valid_image_ids

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        file_name = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.images_dir, file_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = make_pseudo_rgb(image)
        original_size = image.shape[:2]  # (H, W)

        annotations = self.annotations[image_id]
        bboxes = []
        category_ids = []

        for ann in annotations:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x+w, y+h])
            category_ids.append(ann['category_id'])

        if self.transform:
            transformed = self.transform(
                image=image, bboxes=bboxes, class_labels=category_ids)
            image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            if transformed_bboxes:
                bboxes = torch.tensor(transformed_bboxes, dtype=torch.float32)
            else:
                bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.tensor(
                transformed['class_labels'], dtype=torch.long)
        else:
            image = ToTensorV2()(image=image)['image']
            if bboxes:
                bboxes = torch.tensor(bboxes, dtype=torch.float32)
            else:
                bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.tensor(category_ids, dtype=torch.long)

        target = {
            'boxes': bboxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'orig_size': torch.tensor(original_size)
        }
        return image, target


class CocoDataModule(LightningDataModule):
    def __init__(self, train_img, train_ann, val_img, val_ann,
                 batch_size=4, num_workers=0, num_classes=3):
        super().__init__()
        self.train_img = train_img
        self.train_ann = train_ann
        self.val_img = val_img
        self.val_ann = val_ann
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes

    def setup(self, stage=None):
            self.train_dataset = COCODetectionDataset(
                images_dir=self.train_img, annotation_file=self.train_ann,
                transform=get_train_transform()
            )
            self.val_dataset = COCODetectionDataset(
                images_dir=self.val_img, annotation_file=self.val_ann,
                transform=get_val_transform()
            )

    def train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn
            )

    def val_dataloader(self):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn
            )

    def collate_fn(self, batch):
        images, targets = list(zip(*batch))
        images = torch.stack(images, dim=0)
        return images, targets

