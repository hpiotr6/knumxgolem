import torch
import torchvision
import os
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
import pandas as pd
from skimage import io

def get_augmented_embeddings(model, images, aug, aug_times=0):
  return [model(aug(images)).detach().cpu() for _ in range(aug_times)]

def get_predictions(embeddings_ref, embeddings_val, labels_ref, distance):
  # also works for predicting only a batch - can be used during training
  predicted_labels = []
  for emb_val in embeddings_val:
    distances = torch.Tensor([distance(emb_val, emb_ref) for emb_ref in embeddings_ref])
    predicted_labels.append(labels_ref[torch.argmin(distances)])
  return predicted_labels

def predict_dataset(model, dataloader_ref, dataloader_val, distance, aug_ref=None, aug_times=0):
  with torch.no_grad():
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

def evaluate_dataset(model, dataloader_ref, dataloader_val, distance, aug_ref=None, aug_times=0):
  with torch.no_grad():
    predictions, labels_val = predict_dataset(model, dataloader_ref, dataloader_val, distance, aug_ref=None, aug_times=0)
    acc = sum([pred == label for pred, label in zip(predictions, labels_val)]) / len(labels_val)
    return acc

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

class ValDataset(Dataset):
    def __init__(self, img_dir, annotations_file_path, ref_group_label, transform=None):
        self.ref_group_label = ref_group_label
        self.transform = transform
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotations_file_path)
        
        self._filter_annotations()
        
        #self.annotations[transform]
        self.labels_group_ind = self.__init_labels_group()

    def _filter_annotations(self):
        length = len(self.annotations)
        self.annotations = self.annotations[self.annotations["area"] > 2000]
        print(f'Dropped {length - len(self.annotations)} as outliers')
        length = len(self.annotations)
        self.annotations = self.annotations[
            self.annotations["category_id"].isin(self.ref_group_label.keys())
        ]
        print(f'Dropped {length - len(self.annotations)} as wrong classes')

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

if __name__ == '__main__':
    loss = torch.nn.MSELoss()
    path = 'datas'
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    path_ref_images = os.path.join(path, "cropped_ref")
    path_ref_csv = os.path.join(path, "ref1_merged_with_crops.csv")
    path_val_images = os.path.join(path, "cropped_val")
    path_val_csv = os.path.join(path, "val_merged_with_crops.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = torchvision.models.vit_b_16(pretrained=True)
    model.heads.head = torch.nn.Identity()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.encoder.layers.encoder_layer_11.parameters():
        param.requires_grad = True

    model = model.to(device)
    checkpoint = torch.load(os.path.join(path, "vit.pth"))
    model.load_state_dict(checkpoint)

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

    ref_dataset = ReferenceDataset(
        img_dir=path_ref_images,
        annotations_file_path=path_ref_csv,
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

    val_dataset = ValDataset(
        img_dir=path_val_images,
        annotations_file_path=path_val_csv,
        ref_group_label=ref_dataset.labels_group_ind,
        transform=transform_val,
    )
    print(len(val_dataset))

    dataloader_val = DataLoader(val_dataset, batch_size=4, shuffle=True)
    dataloader_ref = DataLoader(ref_dataset, batch_size=4, shuffle=True)

    eval_args = (model, dataloader_ref, dataloader_val, loss)
    eval_kwargs = {'aug_ref': None, 'aug_times': 0}

    with torch.no_grad():
        acc = evaluate_dataset(*eval_args, **eval_kwargs)
        print('Before training accuracy:', acc.item())

