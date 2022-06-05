import os

path = 'datas'
K = 3

import albumentations as A
import torchvision
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import numpy as np

from skimage import io

import torch
import pandas as pd
import os
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
from skimage import io

class ReferenceDataset(Dataset):
    def __init__(self, img_dir, annotations_file_path, transform=None):
        self.transform = transform
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotations_file_path)

        self.delete_invalid()

        self.labels_group_ind = self.__init_labels_group()

    def delete_invalid(self):
        valid_indices = []
        for i in range(len(self.annotations)):
            annotations_row = self.annotations.iloc[i]
            img_path = os.path.join(self.img_dir, annotations_row["crop_file_name"])
            try:
                image = io.imread(img_path)
                valid_indices.append(True)
            except:
                valid_indices.append(False)
                print('Path does not work:', img_path)
        print('Deleted images:', len(valid_indices) - sum(valid_indices))
        self.annotations = self.annotations[pd.Series(valid_indices)]


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotations_row = self.annotations.iloc[idx]
        img_path = os.path.join(self.img_dir, annotations_row["crop_file_name"])
        label = annotations_row["category_id"]

        #image = cv.imread(img_path)
        #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = io.imread(img_path)

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


class Tripplet(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.triplets = self._gen_triplets(dataset.labels_group_ind)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        img1, img2, img3 = (self.dataset[i][0] for i in self.triplets[idx])
        lab1, lab2, lab3 = (self.dataset[i][1] for i in self.triplets[idx])
        return (img1, img2, img3),(lab1, lab2, lab3)

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

print('Ref dataset n_samples:', len(os.listdir(os.path.join(path, 'cropped_ref'))))
print('Valid dataset n_samples:', len(os.listdir(os.path.join(path, 'cropped_val'))))

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
transform_Gauss = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(p=0.5),
        A.Transpose(p=0.5),
        A.OpticalDistortion(p=0.5),
        A.Lambda(p=0.5),
        A.Blur(blur_limit=7, p=0.5),
        # A.Perspective(p=1)
        A.Normalize(),
        ToTensorV2(),
    ]
)

ref_dataset = ReferenceDataset(
    img_dir=os.path.join(path, "cropped_ref"),
    annotations_file_path=os.path.join(path, "ref1_merged_with_crops.csv"),
    transform=transform_Gauss,
)

triplet_dataset = Tripplet(ref_dataset)
x, labels = triplet_dataset[0]
# matplotlib_imshow(x[0])
#(img, label)

dataloader = DataLoader(triplet_dataset, batch_size=4, shuffle=True)

import torch
import torch.nn as nn
import numpy as np
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

"""<h1>Training model</h1>"""

model = torchvision.models.vit_b_16(pretrained=True)
model.heads.head = nn.Identity()
model = model.to(device)

import torch.optim as optim

criterion = nn.TripletMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def get_augmented_embeddings(model, images, aug, aug_times=0):
  return [model(aug(images)).detach().cpu() for _ in range(aug_times)]

def get_predictions(embeddings_ref, embeddings_val, labels_ref, distance):
  # also works for predicting only a batch - can be used during training
  predicted_labels = []
  for emb_val in embeddings_val:
    distances = torch.Tensor([distance(emb_val, emb_ref) for emb_ref in embeddings_ref])
    vals, indices = torch.topk(distances, K, largest=False)
    labels_ref = torch.index_select(labels_ref, 0, indices)

    pred_label = torch.mean(vals / sum(vals) * labels_ref)
    pred_label = torch.round(pred_label)
    predicted_labels.append(pred_label)
  return predicted_labels

def predict_dataset(model, dataloader_ref, dataloader_val, distance, aug_ref=None, aug_times=0):
  embeddings_ref = []
  labels_ref = []
  for imgs, labels in dataloader_ref:
    imgs = imgs.to(device)
    embeddings_ref.extend(model(imgs).detach().cpu())
    labels_ref.extend(labels)

    if aug_ref is not None:
      embeddings_ref.extend(get_augmented_embeddings(model, imgs, aug_ref, aug_times))
      labels_ref.extend(list(labels) * aug_times)

  embeddings_val = []
  labels_val = []
  for imgs, labels in dataloader_val:
    imgs = imgs.to(device)
    embeddings_val.extend(model(imgs).detach().cpu())
    labels_val.extend(labels)

  return get_predictions(embeddings_ref, embeddings_val, labels_ref, distance), labels_val

def evaluate_batch(model, batch_ref, batch_val, labels_ref, labels_val, distance, aug_ref=None, aug_times=0):
  if aug_times > 0:
    batch_ref = torch.cat([batch_ref, get_augmented_embeddings(model, batch_ref, aug_ref, aug_times=0)])
    labels_ref = torch.cat([labels_ref, list(labels_ref) * aug_times])

  embeddings_ref = model(batch_ref)
  embeddings_val = model(batch_val)

  predictions = get_predictions(embeddings_ref, embeddings_val, labels_ref, distance)
  acc = sum([pred == label for pred, label in zip(predictions, labels_val)]) / len(labels_val)
  return acc

def evaluate_dataset(model, dataloader_ref, dataloader_val, distance, aug_ref=None, aug_times=0):
  with torch.no_grad():
    predictions, labels_val = predict_dataset(model, dataloader_ref, dataloader_val, distance, aug_ref=None, aug_times=0)
    acc = sum([pred == label for pred, label in zip(predictions, labels_val)]) / len(labels_val)
    return acc

def train_model(model, dataloader, optimizer, criterion, aug=None, n_epochs=10, eval_args=None, eval_kwargs=None):
  if eval_args is not None:
    acc = evaluate_dataset(*eval_args, **eval_kwargs)
    print('Before training accuracy:', acc)

  for i in range(n_epochs):
      running_loss = 0.0
      for imgs, _ in dataloader:
          x, x_pos, x_neg = imgs
          x = x.to(device)
          x_pos = x_pos.to(device)
          x_neg = x_neg.to(device)

          
          if aug is None:
            x_embedding = model(x)
            neg_embedding = model(x_neg)
            pos_embedding = model(x_pos)
          else:
            x_embedding = model(aug(x))
            pos_embedding = model(aug(x_pos))
            neg_embedding = model(aug(x_neg))

          triplet_loss = criterion(x_embedding, pos_embedding, neg_embedding)
          triplet_loss.backward()
          optimizer.step()
          running_loss += triplet_loss.item()

      print(f"Epoch: {i} loss: {running_loss/len(dataloader)}")
      if eval_args is not None:
        acc = evaluate_dataset(*eval_args, **eval_kwargs)
        print('accuracy:', acc)

'''
noise_pos = torch.rand(4, 3, 224, 224)
dataloader = [((noise_pos, noise_pos + torch.rand(4, 3, 224, 224) * 0.2,
              torch.rand(4, 3, 224, 224) * 1.2), torch.rand(4,))]
'''

#train_model(model, dataloader, optimizer, criterion, n_epochs=10)

transform_ref = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(),
        ToTensorV2(),
    ]
)

ref_dataset = ReferenceDataset(
    img_dir=os.path.join(path, "cropped_ref"),
    annotations_file_path=os.path.join(path, "ref1_merged_with_crops.csv"),
    transform=transform_ref,
)
print(len(ref_dataset))

transform_val = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(),
        ToTensorV2(),
    ]
)

val_dataset = ReferenceDataset(
    img_dir=os.path.join(path, "cropped_val"),
    annotations_file_path=os.path.join(path, "val_merged_with_crops.csv"),
    transform=transform_val,
)
print(len(val_dataset))

dataloader_val = DataLoader(val_dataset, batch_size=4, shuffle=True)
dataloader_ref = DataLoader(ref_dataset, batch_size=4, shuffle=True)
distance = torch.nn.MSELoss()

eval_args = (model, dataloader_ref, dataloader_val, distance)
eval_kwargs = {'aug_ref': None, 'aug_times': 0}
train_model(model, dataloader, optimizer, criterion, n_epochs=10,
eval_args=eval_args, eval_kwargs=eval_kwargs)

torch.save(model.state_dict(), os.path.join(path, "vit.pth"))

#model_inception = torchvision.models.inception_v3(pretrained=True)

