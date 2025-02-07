import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#dateset init
class PlayingCardDataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data = ImageFolder(data_dir, transform=transform)
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return self.data[idx]
  
  @property
  def classes(self):
    return self.data.classes

transform=transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = PlayingCardDataset(
  data_dir = 'archive/train',
  transform=transform
)
  
data_dir = 'archive/train/'
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}



#dataloader init
for image, label in dataset:
  break

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
  break


#model init
class SimpleCardClassifier(nn.Module):
  def __init__(self, num_classes = 53):
    super(SimpleCardClassifier, self).__init__()
    
    self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
    self.features = nn.Sequential(*list(self.base_model.children())[:-1])

    enet_out_size = 1280
    self.classifier = nn.Linear(enet_out_size, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x,1)
    output = self.classifier(x)
    return output
  
model = SimpleCardClassifier(num_classes=53)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)

train_folder = 'archive/train/' 
valid_folder = 'archive/valid/'
test_folder = 'archive/test/'

train_dataset = PlayingCardDataset(train_folder, transform=transform)
valid_dataset = PlayingCardDataset(valid_folder, transform=transform)
test_dataset = PlayingCardDataset(test_folder, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#training loop
num_epochs = 5
training_losses, val_losses = [], []

for epoch in range(num_epochs):
  model.train()
  running_loss = 0.0
  for images, labels in tqdm(train_dataloader, desc='Training loop'):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item() * labels.size(0)
  training_loss = running_loss / len(train_dataloader.dataset)
  training_losses.append(training_loss)

  model.eval()
  running_loss = 0.0
  with torch.no_grad():
    for images, labels in tqdm(valid_dataloader, desc='Validation loop'):
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)
      running_loss += loss.item() * labels.size(0)
  val_loss = running_loss / len(valid_dataloader.dataset)
  val_losses.append(val_loss)

  print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {training_loss}, Valid Loss: {val_loss}")

  