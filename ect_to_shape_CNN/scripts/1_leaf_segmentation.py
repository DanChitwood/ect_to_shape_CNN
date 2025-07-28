# 1_leaf_segmentation.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
import time # For tracking training time
import torch.nn.functional as F
import random # For setting overall random seed

# --- Configuration Parameters ---
# Set random seeds for reproducibility
MANUAL_SEED = 42
torch.manual_seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
random.seed(MANUAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(MANUAL_SEED)
    torch.cuda.manual_seed_all(MANUAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Recommended for reproducibility

# Input/Output Directories (relative to the script's assumed execution location: ect_to_shape_CNN/scripts/)
DATA_ROOT_DIR = Path("../outputs/processed_leaf_data/") # Root directory for processed ECTs and masks
METADATA_FILE = DATA_ROOT_DIR / "metadata.csv"

# Output directory for trained models and training logs
MODEL_OUTPUT_DIR = Path("../outputs/models/")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists
BEST_MODEL_PATH = MODEL_OUTPUT_DIR / "best_unet_model.pth"
TRAINING_METRICS_PATH = MODEL_OUTPUT_DIR / "training_metrics.csv"


# Training Hyperparameters
IMAGE_SIZE = 256 # Expected input/output image resolution
BATCH_SIZE = 16
TRAIN_RATIO = 0.8 # Percentage of data for training
VAL_RATIO = 0.1   # Percentage of data for validation
TEST_RATIO = 0.1  # Percentage of data for final testing

NUM_EPOCHS = 50 # Number of training epochs
LEARNING_RATE = 1e-4 # Adam optimizer learning rate

# Device configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Dataloader workers
NUM_WORKERS = 2 # Number of subprocesses to use for data loading. Adjust based on system (0 for main thread).
print(f"Using {NUM_WORKERS} DataLoader workers.")


# --- Custom Dataset Class ---
class LeafSegmentationDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, base_img_dir: Path, transform=None):
        """
        Initializes the dataset with metadata and image base directory.
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'file_radial_ect' and 'file_shape_mask' paths.
            base_img_dir (Path): Base directory where the ECT and mask images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = metadata_df
        self.base_img_dir = base_img_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construct full paths by joining base_img_dir with the relative paths from metadata
        ect_img_relative_path = Path(self.metadata.iloc[idx]['file_radial_ect'])
        mask_img_relative_path = Path(self.metadata.iloc[idx]['file_shape_mask'])

        ect_img_path = self.base_img_dir / ect_img_relative_path
        mask_img_path = self.base_img_dir / mask_img_relative_path

        try:
            # Load images as grayscale (L mode)
            ect_image = Image.open(ect_img_path).convert('L')
            mask_image = Image.open(mask_img_path).convert('L')
        except FileNotFoundError as e:
            print(f"Error: File not found for index {idx}. ECT path: {ect_img_path}, Mask path: {mask_img_path}")
            raise e # Re-raise to halt training if critical files are missing
        except Exception as e:
            print(f"Error loading images for index {idx}: {e}. ECT path: {ect_img_path}, Mask path: {mask_img_path}")
            raise e

        if self.transform:
            ect_image = self.transform(ect_image)
            mask_image = self.transform(mask_image)

        # Binarize target mask to 0.0 or 1.0. This is crucial for BCEWithLogitsLoss.
        mask_image = (mask_image > 0.5).float()

        return ect_image, mask_image

# --- Data Loading and Splitting Function ---
def get_dataloaders(metadata_file: Path, base_img_dir: Path, image_size: int, batch_size: int,
                    train_ratio: float, val_ratio: float, num_workers: int, manual_seed: int):
    """
    Loads metadata, filters valid entries, creates dataset, and splits into DataLoaders.
    """
    metadata_df = pd.read_csv(metadata_file)
    # Filter for successfully processed samples
    metadata_df = metadata_df[metadata_df['is_processed_valid'] == True]
    if len(metadata_df) == 0:
        raise ValueError("No valid processed shapes found in metadata.csv. Check the file and processing script (0_radial_ect_and_masks.py).")

    # Define transformations for images (resizing and converting to Tensor)
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor() # Converts PIL Image to FloatTensor and scales pixel values to [0.0, 1.0]
    ])

    full_dataset = LeafSegmentationDataset(metadata_df, base_img_dir, transform=image_transform)

    total_samples = len(full_dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size # Assign remaining to test set to ensure sum == total

    # Perform a reproducible random split of the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(manual_seed) # For reproducibility of data split
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")
    print(f"Test Loader Batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)

# --- U-Net Architecture Definition ---
# Reusing the existing U-Net structure, which is a standard implementation.

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
            self.conv = DoubleConv(in_channels, out_channels) # Changed from in_channels + out_channels to in_channels as concat will be handled by call in UNet forward.
                                                              # Corrected: it should be in_channels because the input to this DoubleConv *after* concatenation is in_channels.
                                                              # The input to `self.conv` is the concatenated tensor.
                                                              # If bilinear=True, the input to DoubleConv is in_channels_from_up + in_channels_from_skip_connection.
                                                              # For UNet, it's (1024/2 + 512) for the first Up block if in_channels is 1024 (from previous down block output) and skip connection is 512.
                                                              # Let's align with common U-Net practice for concatenation: input to DoubleConv is `in_channels_after_concat`.
                                                              # In a typical UNet, the `in_channels` to the `DoubleConv` here should be `in_channels_from_up_branch + in_channels_from_skip_connection`.
                                                              # The `Up` class is responsible for receiving the output from the lower layer (`x1`) and the skip connection (`x2`).
                                                              # `in_channels` is the `in_channels` of `x1`. The output of upsampling `x1` will have `in_channels // 2`.
                                                              # This upsampled output is then concatenated with `x2`.
                                                              # So the `DoubleConv` should take `(in_channels // 2) + out_channels` as its input.
                                                              # The original code's `DoubleConv(in_channels + out_channels, out_channels)` for bilinear was correct, 
                                                              # assuming `in_channels` means `x1`'s channels and `out_channels` means `x2`'s channels.
                                                              # Let's revert the change to `in_channels + out_channels` for `DoubleConv` if bilinear is True, which is standard.
                                                              # The `in_channels` for Up module usually refers to the input to `self.up` (which is `x1`).
                                                              # The `out_channels` for Up module usually refers to the desired output channels *after* the DoubleConv.

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # When bilinear, the `in_channels` to DoubleConv is the sum of channels from upsampled path and skip connection.
            # `in_channels` to Up is what comes from the lower layer.
            # `out_channels` to Up is what goes to the next higher layer (after concat and conv).
            # The upsampled feature map will have `in_channels` channels (unless explicitly reduced).
            # The skip connection will have `out_channels` (from the Down block).
            # So, the concatenated feature map will have `in_channels + out_channels` channels.
            self.conv = DoubleConv(in_channels + out_channels, out_channels) # Corrected back to original logic

        else: # Transpose convolution
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # Here, the `in_channels` to DoubleConv is the sum of channels from upsampled path (in_channels // 2) and skip connection (out_channels).
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels) # Corrected logic for transpose conv

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust dimensions for concatenation if there's a slight difference due to pooling/upsampling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
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

        # `in_channels` for Up module is the channel count of the input from the layer below (x5 for up1, x for up2 etc.)
        # `out_channels` for Up module is the channel count of the corresponding skip connection (x4 for up1, x3 for up2 etc.)
        # And also the desired output channel count of the DoubleConv in Up.
        # So, for self.up1, input `x5` has 1024 channels. Skip connection `x4` has 512 channels. Output should be 512.
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1) # Output layer

    def forward(self, x):
        x1 = self.inc(x) # 1 -> 64
        x2 = self.down1(x1) # 64 -> 128
        x3 = self.down2(x2) # 128 -> 256
        x4 = self.down3(x3) # 256 -> 512
        x5 = self.down4(x4) # 512 -> 1024

        x = self.up1(x5, x4) # 1024 -> 512 (after upsample and concat with 512) -> 512 (after conv)
        x = self.up2(x, x3)  # 512 -> 256 (after upsample and concat with 256) -> 256 (after conv)
        x = self.up3(x, x2)  # 256 -> 128 (after upsample and concat with 128) -> 128 (after conv)
        x = self.up4(x, x1)  # 128 -> 64 (after upsample and concat with 64) -> 64 (after conv)
        logits = self.outc(x) # 64 -> n_classes (1)
        return logits

# --- Metrics and Loss Functions ---

def dice_coeff(preds, targets):
    """
    Calculates the Dice coefficient (F1 score) for binary segmentation.
    Args:
        preds (torch.Tensor): Predicted mask (logits), expected shape (N, 1, H, W).
                              Will be binarized using a sigmoid and threshold of 0.5.
        targets (torch.Tensor): Ground truth mask (binary 0 or 1), expected shape (N, 1, H, W).
    Returns:
        float: Dice coefficient averaged over the batch.
    """
    smooth = 1e-6 # Small epsilon to prevent division by zero
    
    # Apply sigmoid to convert logits to probabilities, then binarize
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float() # Binarize predictions

    # Flatten tensors for calculation
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# --- Training and Validation Loops ---

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, best_model_path):
    """
    Trains the U-Net model and evaluates its performance on the validation set.
    Saves the best model based on validation Dice coefficient.
    """
    best_dice = 0.0
    train_losses, train_dices, val_losses, val_dices = [], [], [], []

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

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}")

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

    return train_losses, train_dices, val_losses, val_dices


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
        METADATA_FILE, DATA_ROOT_DIR, IMAGE_SIZE, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, NUM_WORKERS, MANUAL_SEED
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
    metrics_df.to_csv(TRAINING_METRICS_PATH, index=False)
    print(f"\nTraining metrics saved to: {TRAINING_METRICS_PATH}")

    # Optional: Evaluate on Test Set after training (using the best saved model)
    print(f"\n--- Evaluating best model on Test Set ---")
    best_model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    if BEST_MODEL_PATH.exists():
        best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        best_model.eval() # Set to evaluation mode
        
        test_dice_sum = 0.0
        test_loss_sum = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = best_model(inputs)
                test_loss_sum += criterion(outputs, targets).item()
                test_dice_sum += dice_coeff(outputs, targets).item()
            
        test_loss = test_loss_sum / len(test_loader)
        test_dice = test_dice_sum / len(test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}")
    else:
        print(f"Error: Best model not found at {BEST_MODEL_PATH}. Cannot perform test evaluation.")