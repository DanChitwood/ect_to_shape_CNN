# leaf_segmentation.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
import os
import time # For tracking training time
import torch.nn.functional as F

# --- Configuration ---
METADATA_FILE = Path("processed_leaf_data/metadata.csv")
BASE_IMAGE_DIR = Path("processed_leaf_data/")

IMAGE_SIZE = 256
BATCH_SIZE = 16
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

NUM_EPOCHS = 50 # You might want to increase this later based on convergence
LEARNING_RATE = 1e-4 # Common starting point for Adam
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_WORKERS = 2 # Adjusted based on your system's stability
print(f"Using {NUM_WORKERS} DataLoader workers.")

MODEL_SAVE_DIR = Path("models/")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODEL_SAVE_DIR / "best_unet_model.pth"


# --- Custom Dataset Class ---
class LeafSegmentationDataset(Dataset):
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

        # Correct path construction for existing metadata.csv
        try:
            # Attempt to make paths relative to the base_img_dir's name,
            # this correctly handles cases where metadata stores "subdir/file.png"
            # and base_img_dir is "processed_leaf_data/"
            ect_img_relative_to_subdirs = Path(ect_img_full_path_str).relative_to(self.base_img_dir.name)
            mask_img_relative_to_subdirs = Path(mask_img_full_path_str).relative_to(self.base_img_dir.name)
        except ValueError:
            # This can happen if the path is already in the desired relative format (e.g., "shape_masks/file.png")
            # or if the base_img_dir.name is not a parent. In such cases, use the path directly.
            ect_img_relative_to_subdirs = Path(ect_img_full_path_str)
            mask_img_relative_to_subdirs = Path(mask_img_full_path_str)

        ect_img_path = self.base_img_dir / ect_img_relative_to_subdirs
        mask_img_path = self.base_img_dir / mask_img_relative_to_subdirs

        try:
            ect_image = Image.open(ect_img_path).convert('L')
            mask_image = Image.open(mask_img_path).convert('L')
        except FileNotFoundError as e:
            print(f"Error: File not found for index {idx}. ECT path: {ect_img_path}, Mask path: {mask_img_path}")
            # Raising an error during training can stop the process.
            # For robustness in large datasets, consider logging and returning None/a dummy sample,
            # then handling None in DataLoader (e.g., with a custom collate_fn).
            # For now, re-raising to ensure critical path issues are resolved.
            raise e
        except Exception as e:
            print(f"Error loading images for index {idx}: {e}. ECT path: {ect_img_path}, Mask path: {mask_img_path}")
            raise e

        if self.transform:
            ect_image = self.transform(ect_image)
            mask_image = self.transform(mask_image)

        mask_image = (mask_image > 0.5).float() # Binarize target mask to 0.0 or 1.0

        return ect_image, mask_image

# --- Data Loading and Splitting Function ---
def get_dataloaders(metadata_file: Path, base_img_dir: Path, image_size: int, batch_size: int,
                    train_ratio: float, val_ratio: float, num_workers: int):
    
    metadata_df = pd.read_csv(metadata_file)
    metadata_df = metadata_df[metadata_df['is_processed_valid'] == True]
    if len(metadata_df) == 0:
        raise ValueError("No valid processed shapes found in metadata.csv. Check the file and processing script.")

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    full_dataset = LeafSegmentationDataset(metadata_df, base_img_dir, transform=image_transform)

    total_samples = len(full_dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    # Ensure sizes sum up correctly and adjust if there are rounding errors
    if train_size + val_size + test_size != total_samples:
        test_size = total_samples - train_size - val_size


    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # For reproducibility of split
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")
    print(f"Test Loader Batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)

# --- U-Net Architecture Definition ---

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

# --- Metrics and Loss Functions ---

def dice_coeff(preds, targets):
    """
    Calculates the Dice coefficient (F1 score) for binary segmentation.
    Args:
        preds (torch.Tensor): Predicted mask (logits or probabilities), expected shape (N, 1, H, W).
                              Will be binarized using a threshold of 0.5 if not already.
        targets (torch.Tensor): Ground truth mask (binary 0 or 1), expected shape (N, 1, H, W).
    Returns:
        float: Dice coefficient.
    """
    smooth = 1e-6 # Small epsilon to prevent division by zero
    
    # Apply sigmoid and binarize predictions
    preds = torch.sigmoid(preds) # Convert logits to probabilities
    preds = (preds > 0.5).float() # Binarize at 0.5 threshold

    # Flatten tensors for calculation
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# --- Training and Validation Loops ---

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, best_model_path):
    best_dice = 0.0
    # Initialize lists to store metrics for each epoch
    train_losses = []
    train_dices = []
    val_losses = []
    val_dices = []

    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        train_dice = 0.0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device) # ECT image
            targets = targets.to(device) # Shape mask

            optimizer.zero_grad() # Zero the parameter gradients

            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, targets) # Calculate loss
            
            loss.backward() # Backward pass
            optimizer.step() # Optimize

            running_loss += loss.item()
            train_dice += dice_coeff(outputs, targets).item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_dice = train_dice / len(train_loader)
        
        # --- Validation ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad(): # No gradient calculations during validation
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                val_dice += dice_coeff(outputs, targets).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{num_epochs} finished in {epoch_duration:.2f}s")
        print(f"  Train Loss: {epoch_loss:.4f}, Train Dice: {epoch_dice:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        # Store metrics for this epoch
        train_losses.append(epoch_loss)
        train_dices.append(epoch_dice)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        # Save the best model based on validation Dice
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved with Val Dice: {best_dice:.4f}")
            
    print("\n--- Training Complete ---")
    print(f"Best Validation Dice achieved: {best_dice:.4f}")

    # Return the collected metrics
    return train_losses, train_dices, val_losses, val_dices


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
        METADATA_FILE, BASE_IMAGE_DIR, IMAGE_SIZE, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, NUM_WORKERS
    )

    # 2. Initialize Model
    # U-Net input: 1 channel (grayscale ECT)
    # U-Net output: 1 channel (binary mask logits)
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    
    # 3. Define Loss Function and Optimizer
    # Binary Cross Entropy with Logits is stable for binary segmentation (no need for sigmoid on output)
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Train the Model and capture metrics
    train_losses, train_dices, val_losses, val_dices = train_model(
        model, train_loader, val_loader, optimizer, criterion, DEVICE, NUM_EPOCHS, BEST_MODEL_PATH
    )

    # After training, save the metrics to a CSV for analysis
    metrics_df = pd.DataFrame({
        'Epoch': range(1, NUM_EPOCHS + 1),
        'Train_Loss': train_losses,
        'Train_Dice': train_dices,
        'Val_Loss': val_losses,
        'Val_Dice': val_dices
    })
    metrics_csv_path = MODEL_SAVE_DIR / "training_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nTraining metrics saved to: {metrics_csv_path}")

    # Optional: Evaluate on Test Set after training (using the best saved model)
    print(f"\n--- Evaluating best model on Test Set ---")
    best_model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    if BEST_MODEL_PATH.exists():
        best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        best_model.eval()
        test_dice = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = best_model(inputs)
                test_loss += criterion(outputs, targets).item()
                test_dice += dice_coeff(outputs, targets).item()
        
        test_loss /= len(test_loader)
        test_dice /= len(test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}")
    else:
        print("Best model not found. Run training first.")