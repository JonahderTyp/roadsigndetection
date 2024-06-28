import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import time

# CSV-Dateien laden
train_df = pd.read_csv('Dataset/Train.csv')
test_df = pd.read_csv('Dataset/Test.csv')
meta_df = pd.read_csv('Dataset/Meta.csv')

# Hyperparameter
batch_size = 4  # Weiter reduzierte Batch-Größe
learning_rate = 0.001
num_epochs = 10

# Transformationen definieren
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class TrafficSignDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, -1])
        image = Image.open(img_name)
        label = self.dataframe.iloc[idx, -2]  # ClassId steht in der vorletzten Spalte
        
        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Debugging: Phase Start
            print(f'Starting phase: {phase}')
            print(f'Dataloader length: {len(dataloaders[phase])}')

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Debugging: Batch-Ende
                print(f'Phase: {phase}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    return model

if __name__ == '__main__':
    # Ladezeiten der Bilder überprüfen
    sample_paths = train_df['Path'].head(10).tolist()
    root_dir = 'Dataset'

    for img_name in sample_paths:
        img_path = os.path.join(root_dir, img_name)
        start_time = time.time()
        image = Image.open(img_path)
        image = data_transforms(image)
        end_time = time.time()
        print(f'Laden und Transformieren von {img_name} dauerte {end_time - start_time:.4f} Sekunden')

    # Datasets und Dataloaders erstellen
    train_dataset = TrafficSignDataset(dataframe=train_df, root_dir='Dataset', transform=data_transforms)
    test_dataset = TrafficSignDataset(dataframe=test_df, root_dir='Dataset', transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    dataloaders = {'train': train_loader, 'val': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(test_dataset)}

    # Modell laden
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Letzten Layer anpassen
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(meta_df['ClassId'].unique()))

    # Modell auf GPU übertragen
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Verlustfunktion und Optimierer definieren
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Modell trainieren
    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=num_epochs)

    # Modell speichern
    torch.save(model.state_dict(), 'traffic_sign_model.pth')

    # Modell evaluieren
    model.eval()
    val_acc = 0.0
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data)

    val_acc = val_acc.double() / dataset_sizes['val']
    print(f'Validation Accuracy: {val_acc:.4f}')
