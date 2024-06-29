import os
import pandas as pd
import matplotlib.pyplot as plt


FOLDER = 'logs/resnet_gtsrb/version_6/'

# Load the sample data into a DataFrame
df = pd.read_csv(os.path.join(FOLDER, 'metrics.csv'))

# Drop rows where all values are NaN (these are the validation rows)
df_train = df.dropna(subset=['train_acc', 'train_loss'])
df_val = df.dropna(subset=['val_acc', 'val_loss'])

# Plot the training and validation accuracy and loss
plt.figure(figsize=(12, 6))

# Plot training accuracy and loss
plt.plot(df_train['epoch'], df_train['train_acc'], label='Train Accuracy', marker='o')
plt.plot(df_train['epoch'], df_train['train_loss'], label='Train Loss', marker='o')

# Plot validation accuracy and loss
plt.plot(df_val['epoch'], df_val['val_acc'], label='Val Accuracy', marker='x')
plt.plot(df_val['epoch'], df_val['val_loss'], label='Val Loss', marker='x')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training and Validation Accuracy and Loss')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig(os.path.join(FOLDER, 'plot.png'))