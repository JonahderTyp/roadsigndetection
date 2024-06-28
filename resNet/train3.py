import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# CSV-Dateien laden
train_df = pd.read_csv('../gtsrb-german-traffic-sign/Train.csv')
test_df = pd.read_csv('../gtsrb-german-traffic-sign/Test.csv')
meta_df = pd.read_csv('../gtsrb-german-traffic-sign/Meta.csv')

# Hyperparameter
batch_size = 4
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
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

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
                print(f'Phase: {phase}, Batch: {batch_idx}, Loss: {loss.item():.4f}', end='\r')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

if __name__ == '__main__':
    ROOTDIR = '../gtsrb-german-traffic-sign'
    devicename = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {devicename}')

    # Datasets und Dataloaders erstellen
    train_dataset = TrafficSignDataset(dataframe=train_df, root_dir=ROOTDIR, transform=data_transforms)
    test_dataset = TrafficSignDataset(dataframe=test_df, root_dir=ROOTDIR, transform=data_transforms)

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
    device = torch.device(devicename)
    model = model.to(device)

    # Verlustfunktion und Optimierer definieren
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Modell trainieren
    model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model(
        model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=num_epochs
    )

    # Modell speichern
    torch.save(model.state_dict(), 'traffic_sign_model.pth')

    # Plotting the training and validation loss
    epochs = range(1, num_epochs+1)

    plt.figure()
    plt.plot(epochs, train_loss_history, 'r', label='Training loss')
    plt.plot(epochs, val_loss_history, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_history.png')

    plt.figure()
    plt.plot(epochs, train_acc_history, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc_history, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_history.png')
