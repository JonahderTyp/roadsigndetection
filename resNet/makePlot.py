import os
import pandas as pd
import matplotlib.pyplot as plt

FOLDER = 'logs/resnet_gtsrb/version_10/'

# Load the sample data into a DataFrame
df = pd.read_csv(os.path.join(FOLDER, 'metrics.csv'))

# Drop rows where all values are NaN (these are the validation rows)
df_train = df.dropna(subset=['train_acc', 'train_loss'])
df_val = df.dropna(subset=(['val_acc', 'val_loss']))

# Define the columns to plot for training and validation
low_value_columns = ['train_loss', 'val_loss']
high_value_columns = ['train_acc', 'train_f1', 'train_precision', 'train_recall', 'val_acc', 'val_f1', 'val_precision', 'val_recall']

fig, axs = plt.subplots(1, 2, figsize=(20, 8))

# Plot low value metrics
for column in low_value_columns:
    if column in df_train.columns or column in df_val.columns:
        if 'train' in column:
            axs[0].plot(df_train['epoch'], df_train[column], label=f'Train {column.split("_")[1].capitalize()}')
        else:
            axs[0].plot(df_val['epoch'], df_val[column], label=f'Val {column.split("_")[1].capitalize()}')

axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Value')
axs[0].set_title('Training and Validation Loss')
axs[0].legend()
axs[0].grid(True)

# Plot high value metrics
for column in high_value_columns:
    if column in df_train.columns or column in df_val.columns:
        if 'train' in column:
            axs[1].plot(df_train['epoch'], df_train[column], label=f'Train {column.split("_")[1].capitalize()}')
        else:
            axs[1].plot(df_val['epoch'], df_val[column], label=f'Val {column.split("_")[1].capitalize()}')

axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Value')
axs[1].set_title('Training and Validation Accuracy, Precision, Recall, F1')
axs[1].legend()
axs[1].grid(True)

# Save the plot
plt.savefig(os.path.join(FOLDER, 'plot.png'))

# Show the plot
plt.show()
