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
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import accuracy, precision, recall, f1_score

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
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
        preds = torch.argmax(outputs, dim=1)
        
        acc = accuracy(preds, labels, task='multiclass', num_classes=43)
        prec = precision(preds, labels, task='multiclass', average='macro', num_classes=43)
        rec = recall(preds, labels, task='multiclass', average='macro', num_classes=43)
        f1 = f1_score(preds, labels, task='multiclass', average='macro', num_classes=43)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_precision', prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_recall', rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        
        acc = accuracy(preds, labels, task='multiclass', num_classes=43)
        prec = precision(preds, labels, task='multiclass', average='macro', num_classes=43)
        rec = recall(preds, labels, task='multiclass', average='macro', num_classes=43)
        f1 = f1_score(preds, labels, task='multiclass', average='macro', num_classes=43)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', prec, prog_bar=True)
        self.log('val_recall', rec, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

def main(version=0):
    data_module = GTSRBDataModule(
        train_csv='../gtsrb-german-traffic-sign/Train.csv',
        train_dir='../gtsrb-german-traffic-sign',
        test_csv='../gtsrb-german-traffic-sign/Test.csv',
        test_dir='../gtsrb-german-traffic-sign',
        batch_size=32
    )

    model = ResNetGTSRB()

    logger = CSVLogger("logs", name="resnet_gtsrb", version=version)

    # Create version directory
    checkpoint_dir = os.path.join("checkpoints", f"version_{version}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Add ModelCheckpoint callback to save model after every epoch
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='resnet_gtsrb-{epoch:02d}-{val_loss:.2f}',
        save_top_k=-1,  # Save all models
        verbose=True,
        save_last=True,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        logger=logger,
        callbacks=[checkpoint_callback]  # Include the checkpoint callback
    )
    trainer.fit(model, data_module)
    trainer.validate(model, data_module.val_dataloader())

if __name__ == '__main__':
    main(version=10)