import torch
import pandas as pd
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader

ref1_merged = pd.read_csv("ref1_merged.csv")


class ReferenceDataset(Dataset):
    # TODO transforms
    def __init__(self, img_dir, annotations_file_path):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotations_file_path)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx]["file_name"])
        image = Image.open(img_path)
        # image = cv2.imread(img_path)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.img_labels.iloc[idx, 1]
        # if label == 0 and self.transform_background:
        #     image = self.transform_background(image=image)["image"]
        # if label != 0 and self.transform_positives:
        #     image = self.transform_positives(image=image)["image"]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    ref_dataset = ReferenceDataset(
        img_dir="public_dataset/reference_images_part1",
        annotations_file_path="ref1_merged.csv",
    )
    print(ref_dataset[0])
