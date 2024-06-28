import os
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split

# Equalize function for image preprocessing
def equalize(img):
    return ImageOps.equalize(img)

# Data preprocessing
train_transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((224, 224)),
    transforms.Lambda(equalize),  # Apply equalization
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

# Path to your dataset
train_path = '../gtsrb-german-traffic-sign/train'

# Dataset and DataLoader
dataset = datasets.ImageFolder(root=train_path, transform=train_transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet and modify it for the GTSRB classification task
resnet = models.resnet18(weights=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 43)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Save the model
torch.save(resnet.state_dict(), 'resnet_gtsrb.pth')

# Validate the model
resnet.load_state_dict(torch.load('resnet_gtsrb.pth'))
resnet.eval()

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the validation images: {100 * correct / total} %')
