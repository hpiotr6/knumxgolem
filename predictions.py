import torch
import torchvision
import os

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
model.load_state_dict()
torch.save(model.state_dict(), os.path.join(path, "vit.pth"))

