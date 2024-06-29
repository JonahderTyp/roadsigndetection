import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
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
import numpy as np

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

        return image, label, img_name

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
        # self.train_dataset = GTSRBDataset(csv_file=self.train_csv, root_dir=self.train_dir, transform=self.transform)
        self.test_dataset = GTSRBDataset(csv_file=self.test_csv, root_dir=self.test_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
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


def draw_text(image, text, position):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=(255, 0, 0))


def test_model(checkpoint_path, test_csv, test_dir, meta_csv):
    # Load the trained model from the checkpoint
    model = ResNetGTSRB.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Check if GPU is available and move model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load metadata
    meta = pd.read_csv(meta_csv, dtype={'ClassId': int, 'SignName': str})
    class_id_to_name = dict(zip(meta['ClassId'], meta['SignName']))

    # Setup the data module
    data_module = GTSRBDataModule(
        train_csv='',  # not used in testing
        train_dir='',  # not used in testing
        test_csv=test_csv,
        test_dir=test_dir,
        batch_size=32
    )
    data_module.setup(stage='test')

    # Prepare the test dataloader
    test_loader = data_module.test_dataloader()

    # Transform to revert normalized image
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])

    # Directory to save falsely labeled images
    false_label_dir = 'false_labels'
    os.makedirs(false_label_dir, exist_ok=True)

    # Iterate over test data
    for batch in test_loader:
        images, labels, img_names = batch

        # Move images and labels to device
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        # Find falsely labeled images
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                img_name = img_names[i]
                img = inv_transform(images[i].cpu()).permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img)

                # Add text to image
                actual_class = labels[i].item()
                predicted_class = preds[i].item()
                actual_class_name = class_id_to_name[actual_class]
                predicted_class_name = class_id_to_name[predicted_class]
                text = f"Actual: {actual_class} - {actual_class_name}\n Predicted: {predicted_class} - {predicted_class_name}"
                draw_text(img_pil, text, position=(10, 10))

                save_path = os.path.join(false_label_dir, os.path.basename(img_name))
                img_pil.save(save_path)
                print(f'Saved {save_path} as label {actual_class} ({actual_class_name}) was misclassified as {predicted_class} ({predicted_class_name})')

if __name__ == '__main__':
    checkpoint_path = 'checkpoints/version_10/resnet_gtsrb-epoch=99-val_loss=0.05.ckpt'
    test_csv = '../gtsrb-german-traffic-sign/Test.csv'
    test_dir = '../gtsrb-german-traffic-sign'
    meta_csv = '../gtsrb-german-traffic-sign\Meta.csv'
    test_model(checkpoint_path, test_csv, test_dir, meta_csv)
