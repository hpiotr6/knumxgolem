import torch
import pandas as pd
import os
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2 as cv


class ReferenceDataset(Dataset):
    def __init__(self, img_dir, annotations_file_path, transform=None):
        self.transform = transform
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotations_file_path)
        self.labels_group_ind = self.__init_labels_group()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotations_row = self.annotations.iloc[idx]
        img_path = os.path.join(self.img_dir, annotations_row["crop_file_name"])
        label = annotations_row["category_id"]
        image = cv.imread(img_path)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # label = self.img_labels.iloc[idx, 1]
        # # if label == 0 and self.transform_background:
        # #     image = self.transform_background(image=image)["image"]
        # # if label != 0 and self.transform_positives:
        # #     image = self.transform_positives(image=image)["image"]
        if self.transform:
            image = self.transform(image=image)["image"]
        # # if self.target_transform:
        # #     label = self.target_transform(label)
        return image, label

    def __init_labels_group(self):
        return (
            self.annotations.reset_index()
            .groupby(by="category_id")["index"]
            .apply(list)
            .reset_index(name="category_indices")["category_indices"]
            .to_dict()
        )


class ValDataset(Dataset):
    def __init__(self, img_dir, annotations_file_path, ref_group_label, transform=None):
        self.ref_group_label = ref_group_label
        self.transform = transform
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotations_file_path)
        self._filter_annotations()
        self.annotations[transform]
        self.labels_group_ind = self.__init_labels_group()

    def _filter_annotations(self):
        self.annotations = self.annotations[self.annotations["area"] > 1000]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotations_row = self.annotations.iloc[idx]
        img_path = os.path.join(self.img_dir, annotations_row["crop_file_name"])
        label = annotations_row["category_id"]
        image = cv.imread(img_path)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

    def __init_labels_group(self):
        return (
            self.annotations.reset_index()
            .groupby("category_id")["index"]
            .apply(list)
            .to_dict()
        )


class Tripplet(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.triplets = self._gen_triplets(dataset.labels_group_ind)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return (self.dataset[i] for i in self.triplets[idx])

    @staticmethod
    def _gen_triplets(labels_group_ind):
        triplets = []
        # x = {10:[1,2,3,5],11:[1,2,3,5], 12:[1,2,3,5]}
        for class_idx, indices in labels_group_ind.items():
            class_inds_without_current = list(labels_group_ind.keys()).copy()
            class_inds_without_current.remove(class_idx)
            # print(class_inds_without_current,indices)
            for idx in indices:
                inds_cp = indices.copy()
                inds_cp.remove(idx)
                same_class_idx = random.choice(inds_cp)
                sample_class = random.choice(class_inds_without_current)
                other_class_idx = random.choice(labels_group_ind[sample_class])
                triplets.append([idx, same_class_idx, other_class_idx])

        # triplets
        flatten_vals = [
            item for sublist in list(labels_group_ind.values()) for item in sublist
        ]
        assert len(triplets) == len(flatten_vals)  # utrata danych mordko

        return triplets
