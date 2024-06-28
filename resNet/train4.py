import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, dtype={'Path': str, 'ClassId': int})
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx]["Path"]))
        image = Image.open(img_name)
        label = int(self.annotations.iloc[idx]["ClassId"])

        # Debugging: Print the path and label
        # print(f"Image path: {img_name}, Label: {label}")

        if label < 0 or label >= 43:
            raise ValueError(f"Label {label} is out of range!")

        if self.transform:
            image = self.transform(image)

        return image, label

class GTSRBDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, train_dir, test_csv, test_dir, batch_size=32):
        super().__init__()
        self.train_csv = train_csv
        self.train_dir = train_dir
        self.test_csv = test_csv
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        self.train_dataset = GTSRBDataset(csv_file=self.train_csv, root_dir=self.train_dir, transform=self.transform)
        self.test_dataset = GTSRBDataset(csv_file=self.test_csv, root_dir=self.test_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

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

def main():
    data_module = GTSRBDataModule(
        train_csv='../gtsrb-german-traffic-sign/Train.csv',
        train_dir='../gtsrb-german-traffic-sign',
        test_csv='../gtsrb-german-traffic-sign/Test.csv',
        test_dir='../gtsrb-german-traffic-sign',
        batch_size=32
    )

    model = ResNetGTSRB()

    trainer = pl.Trainer(max_epochs=10, num_nodes=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, data_module)
    trainer.validate(model, data_module.val_dataloader())

if __name__ == '__main__':
    main()
