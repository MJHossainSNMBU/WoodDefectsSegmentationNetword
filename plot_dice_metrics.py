import pandas as pd
import matplotlib.pyplot as plt

# Define paths to log files
unet_log_path = '/content/log.csv'
resunet_log_path = '/content/ResUnetmodel_log.csv'
unet3plus_log_path = '/content/unet3plusResnet50_log.csv'
attention_unet_log_path = '/content/AttentionUnetmodel_log.csv'

# Load the log files
try:
    unet_log = pd.read_csv(unet_log_path)
    resunet_log = pd.read_csv(resunet_log_path)
    unet3plus_log = pd.read_csv(unet3plus_log_path)
    attention_unet_log = pd.read_csv(attention_unet_log_path)
except FileNotFoundError as e:
    print(f"File not found: {e}")
    print("Please check the file paths and try again.")
    raise

def plot_dice_coefficient(log_files, labels, colors):
    plt.figure(figsize=(10, 6))
    for log_file, label, color in zip(log_files, labels, colors):
        plt.plot(log_file['epoch'], log_file['dice_coef'], label=f'{label} Training Dice Coefficient', color=color, linestyle='-')
        plt.plot(log_file['epoch'], log_file['val_dice_coef'], label=f'{label} Validation Dice Coefficient', color=color, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient Comparison Across Models')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('dice_coefficient_plot_all_models.png')
    plt.show()

def plot_dice_loss(log_files, labels, colors):
    plt.figure(figsize=(10, 6))
    for log_file, label, color in zip(log_files, labels, colors):
        plt.plot(log_file['epoch'], log_file['loss'], label=f'{label} Training Dice Loss', color=color, linestyle='-')
        plt.plot(log_file['epoch'], log_file['val_loss'], label=f'{label} Validation Dice Loss', color=color, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.title('Dice Loss Comparison Across Models')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('dice_loss_plot_all_models.png')
    plt.show()

# Prepare logs, labels, and colors
log_files = [unet_log, resunet_log, unet3plus_log, attention_unet_log]
labels = ['UNet', 'ResUNet', 'UNet3PlusResNet50', 'Attention UNet']
colors = ['b', 'g', 'm', 'c']  # Blue, Green, Magenta, Cyan
