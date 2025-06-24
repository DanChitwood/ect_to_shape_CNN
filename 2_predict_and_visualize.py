import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F # Needed for Up-sampling in UNet

# --- Import necessary components from leaf_segmentation.py ---
# We'll re-define them here for clarity and to make this script standalone.
# In a larger project, you'd put these into a separate 'models.py', 'datasets.py' etc.

# --- Configuration (must match training script) ---
METADATA_FILE = Path("processed_leaf_data/metadata.csv")
BASE_IMAGE_DIR = Path("processed_leaf_data/")
IMAGE_SIZE = 256
BATCH_SIZE = 1 # Use batch size 1 for individual image prediction and visualization
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1 # This script will primarily use the test set
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MODEL_SAVE_DIR = Path("models/")
BEST_MODEL_PATH = MODEL_SAVE_DIR / "best_unet_model.pth"

# --- Custom Dataset Class (Copied from leaf_segmentation.py) ---
class LeafSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_df: pd.DataFrame, base_img_dir: Path, transform=None):
        self.metadata = metadata_df
        self.base_img_dir = base_img_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ect_img_full_path_str = self.metadata.iloc[idx]['file_radial_ect']
        mask_img_full_path_str = self.metadata.iloc[idx]['file_shape_mask']

        try:
            ect_img_relative_to_subdirs = Path(ect_img_full_path_str).relative_to(self.base_img_dir.name)
            mask_img_relative_to_subdirs = Path(mask_img_full_path_str).relative_to(self.base_img_dir.name)
        except ValueError:
            ect_img_relative_to_subdirs = Path(ect_img_full_path_str)
            mask_img_relative_to_subdirs = Path(mask_img_full_path_str)

        ect_img_path = self.base_img_dir / ect_img_relative_to_subdirs
        mask_img_path = self.base_img_dir / mask_img_relative_to_subdirs

        try:
            ect_image = Image.open(ect_img_path).convert('L')
            mask_image = Image.open(mask_img_path).convert('L')
        except FileNotFoundError as e:
            print(f"Error: File not found for index {idx}. ECT path: {ect_img_path}, Mask path: {mask_img_path}")
            raise e
        except Exception as e:
            print(f"Error loading images for index {idx}: {e}. ECT path: {ect_img_path}, Mask path: {mask_img_path}")
            raise e

        if self.transform:
            ect_image = self.transform(ect_image)
            mask_image = self.transform(mask_image)

        mask_image = (mask_image > 0.5).float() # Binarize target mask to 0.0 or 1.0

        return ect_image, mask_image

# --- U-Net Architecture Definition (Copied from leaf_segmentation.py) ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # Adjust dimensions for concatenation (if there's a slight difference due to pooling/upsampling)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1) # Concatenate along the channel dimension
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64) # Input layer
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024) # Bottom of the U

        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1) # Output layer

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4) # Skip connection from x4
        x = self.up2(x, x3)  # Skip connection from x3
        x = self.up3(x, x2)  # Skip connection from x2
        x = self.up4(x, x1)  # Skip connection from x1
        logits = self.outc(x)
        return logits

# --- Data Loading Function (Modified for prediction focus) ---
# We'll only need the test loader or a way to get individual samples
def get_test_dataloader(metadata_file: Path, base_img_dir: Path, image_size: int, batch_size: int):
    metadata_df = pd.read_csv(metadata_file)
    metadata_df = metadata_df[metadata_df['is_processed_valid'] == True]
    if len(metadata_df) == 0:
        raise ValueError("No valid processed shapes found in metadata.csv. Check the file and processing script.")

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    full_dataset = LeafSegmentationDataset(metadata_df, base_img_dir, transform=image_transform)

    # For consistent test set selection with the training run
    total_samples = len(full_dataset)
    train_size = int(TRAIN_RATIO * total_samples)
    val_size = int(VAL_RATIO * total_samples)
    test_size = total_samples - train_size - val_size
    if train_size + val_size + test_size != total_samples:
        test_size = total_samples - train_size - val_size

    # Split using the same seed as training for consistent test set selection
    _, _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True) # Use 0 workers for simplicity in visualization script
    print(f"Loaded Test Dataset with {len(test_dataset)} samples.")
    return test_loader

# --- Main Prediction and Visualization Logic ---
if __name__ == "__main__":
    # 1. Load the trained model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    if BEST_MODEL_PATH.exists():
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        print(f"Successfully loaded model from {BEST_MODEL_PATH}")
    else:
        print(f"Error: Model not found at {BEST_MODEL_PATH}. Please run training script first.")
        exit()

    # 2. Get the test data loader
    test_loader = get_test_dataloader(METADATA_FILE, BASE_IMAGE_DIR, IMAGE_SIZE, BATCH_SIZE)

    # 3. Predict and visualize a few samples
    num_samples_to_visualize = 5 # You can change this number
    
    print(f"\n--- Visualizing {num_samples_to_visualize} samples from the Test Set ---")
    fig, axes = plt.subplots(num_samples_to_visualize, 3, figsize=(12, 4 * num_samples_to_visualize))
    
    if num_samples_to_visualize == 1: # Handle case for single row if only 1 sample
        axes = axes.reshape(1, -1)

    for i, (inputs, targets) in enumerate(test_loader):
        if i >= num_samples_to_visualize:
            break

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
        
        # Process outputs
        predicted_mask = torch.sigmoid(outputs) # Convert logits to probabilities
        predicted_mask = (predicted_mask > 0.5).float() # Binarize

        # Move tensors to CPU and convert to numpy for plotting
        input_img = inputs.squeeze().cpu().numpy() # Remove batch and channel dim
        true_mask = targets.squeeze().cpu().numpy()
        predicted_mask_np = predicted_mask.squeeze().cpu().numpy()

        # Plotting
        ax = axes[i, 0]
        ax.imshow(input_img, cmap='gray')
        ax.set_title(f'Sample {i+1} ECT Input')
        ax.axis('off')

        ax = axes[i, 1]
        ax.imshow(true_mask, cmap='gray')
        ax.set_title(f'Sample {i+1} Ground Truth Mask')
        ax.axis('off')

        ax = axes[i, 2]
        ax.imshow(predicted_mask_np, cmap='gray')
        ax.set_title(f'Sample {i+1} Predicted Mask')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("\nVisualization complete. Close the plot to exit.")