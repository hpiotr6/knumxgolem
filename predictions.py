import torch
import torchvision
import os

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

if __name__ == '__main__':
    loss = torch.nn.MSELoss()
    path = 'datas'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = torchvision.models.vit_b_16(pretrained=True)
    model.heads.head = torch.nn.Identity()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.encoder.layers.encoder_layer_11.parameters():
        param.requires_grad = True

    model = model.to(device)
    model.load_state_dict(os.path.join(path, "vit.pth"))


