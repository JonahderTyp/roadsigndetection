import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class ResNetGTSRB(pl.LightningModule):
    def __init__(self, num_classes=43):
        super(ResNetGTSRB, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# Function to load the trained model
def load_model(checkpoint_path, num_classes=43):
    model = ResNetGTSRB.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
    model.eval()
    return model

# Function to preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to make a prediction
def predict_image(model, image_path, class_names, device):
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    return predicted_class


# Load class names (ensure this matches how the dataset was loaded/trained)
class_names = pd.read_csv('../gtsrb-german-traffic-sign/Meta.csv')['ClassId'].tolist()  # Adjust path and column name as necessary

# Path to the saved model checkpoint
checkpoint_path = './logs/resnet_gtsrb/version_2/checkpoints/epoch=0-step=1226.ckpt'  # Adjust filename as necessary

# Determine the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model and move it to the device
model = load_model(checkpoint_path).to(device)

# Path to the image to be predicted
image_path = 'C:/Users/jonah/development/roadsigndetection/gtsrb-german-traffic-sign/Train/38/00038_00062_00018.png'  # Replace with the actual image path

# Make a prediction
predicted_class = predict_image(model, image_path, class_names, device)
print(f'The predicted class is: {predicted_class}')