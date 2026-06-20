import os
import os.path as osp
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FGVCAircraft(Dataset):
    """
    FGVC-Aircraft Dataset modified to match CUBDataset structure.
    """
    def __init__(self, data_root: str, is_train: bool = True, transformer: Optional[Callable] = None) -> None:
        super().__init__()
        self.data_root = os.path.expanduser(data_root)
        self.is_train = is_train
        self.transformer = transformer
        
        # Aircraft specific settings
        self.class_type = 'variant' 
        split = 'trainval' if is_train else 'test'
        self.classes_file = osp.join(self.data_root, 'data', f'images_{self.class_type}_{split}.txt')
        
        self.prepare_data()

    def prepare_data(self):
        # Read the labels file
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.strip().split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # Map string class names to integers
        self.classes = np.unique(targets)
        print (self.classes)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        
        self.samples = []
        for i in range(len(image_ids)):
            img_path = osp.join(self.data_root, 'data', 'images', f'{image_ids[i]}.jpg')
            self.samples.append((img_path, self.class_to_idx[targets[i]]))

        # Unique indices for tracking (as seen in your original AIR code)
        self.uq_idxs = np.array(range(len(self.samples)))

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        
        # Use cv2 to match CUBDataset / Albumentations workflow
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transformer is not None:
            # Albumentations takes 'image' as keyword arg
            result = self.transformer(image=img)
            img = result["image"]

        # Returning img, target, and unique index to match original AIR behavior
        return img, target

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return f"FGVCAircraft(root={self.data_root}, is_train={self.is_train}, Samples={len(self)})"

# --- Utility functions for Loading ---

def data_transformer(is_train: bool = True, scale_size=256, crop_size=224):
    if is_train:
        transformer = A.Compose([
            A.Resize(scale_size, scale_size),
            A.RandomCrop(crop_size, crop_size),
            A.HorizontalFlip(),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        transformer = A.Compose([
            A.Resize(scale_size, scale_size),
            A.CenterCrop(crop_size, crop_size),
            A.Normalize(),
            ToTensorV2()
        ])

    return transformer


def get_train_val_loader(
    data_root, batch_size=32, scale_size=256, crop_size=224,
    num_workers=8, pin_memory=True
):
    train_dataset = FGVCAircraft(
        data_root=data_root,
        is_train=True,
        transformer=data_transformer(
            is_train=True, scale_size=scale_size, crop_size=crop_size
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_dataset = FGVCAircraft(
        data_root=data_root,
        is_train=False,
        transformer=data_transformer(
            is_train=False, scale_size=scale_size, crop_size=crop_size
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def get_test_loader(
    data_root, batch_size=32, scale_size=256, crop_size=224,
    num_workers=8, pin_memory=True
):
    test_dataset = FGVCAircraft(
        data_root=data_root,
        is_train=False,
        transformer=data_transformer(
            is_train=False, scale_size=scale_size, crop_size=crop_size
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return test_loader
