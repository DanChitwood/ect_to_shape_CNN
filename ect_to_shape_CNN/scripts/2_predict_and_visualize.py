# 2_predict_and_visualize.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageChops
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F
from skimage import measure # For finding contours


# --- Configuration ---
# Seed for consistent data splitting (must match 1_leaf_segmentation.py)
MANUAL_SEED = 42 

# Seed for reproducible random selection of leaves for *this* visualization
MANUAL_SEED_VIZ = 123 

random.seed(MANUAL_SEED_VIZ) # For random.sample
np.random.seed(MANUAL_SEED_VIZ) # For numpy operations if any randoms used
torch.manual_seed(MANUAL_SEED_VIZ) # For PyTorch operations if any randoms used here
if torch.cuda.is_available():
    torch.cuda.manual_seed(MANUAL_SEED_VIZ)
    torch.cuda.manual_seed_all(MANUAL_SEED_VIZ)

# Input/Output Directories (relative to the script's assumed execution location: ect_to_shape_CNN/scripts/)
DATA_ROOT_DIR = Path("../outputs/processed_leaf_data/") # Root directory for processed ECTs and masks
METADATA_FILE = DATA_ROOT_DIR / "metadata.csv"

MODEL_OUTPUT_DIR = Path("../outputs/models/")
BEST_MODEL_PATH = MODEL_OUTPUT_DIR / "best_unet_model.pth"

FIGURE_OUTPUT_DIR = Path("../outputs/figures/")
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE_PATH = FIGURE_OUTPUT_DIR / "leaf_segmentation_figure.png"

# Training Hyperparameters (must match training script's config for dataset split consistency)
IMAGE_SIZE = 256
BATCH_SIZE = 1 # Use batch size 1 for individual image prediction and visualization
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_WORKERS = 0 # Use 0 workers for simplicity in visualization script to avoid multiprocessing issues with matplotlib
print(f"Using {NUM_WORKERS} DataLoader workers for visualization.")

# Figure Layout Parameters
NUM_ROWS_FIGURE = 10 # Number of rows in the figure
PAIRS_PER_ROW = 4 # Number of ECT+Overlay pairs per row
TOTAL_LEAVES_TO_VIZ = NUM_ROWS_FIGURE * PAIRS_PER_ROW
TOTAL_PANELS_PER_ROW = PAIRS_PER_ROW * 2 # 2 panels per leaf (ECT, Overlay)
FIG_WIDTH_INCHES = 8.5
FIG_HEIGHT_INCHES = 11

# --- Custom Dataset Class (Copied from 1_leaf_segmentation.py) ---
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

        ect_img_relative_path = Path(self.metadata.iloc[idx]['file_radial_ect'])
        mask_img_relative_path = Path(self.metadata.iloc[idx]['file_shape_mask'])

        ect_img_path = self.base_img_dir / ect_img_relative_path
        mask_img_path = self.base_img_dir / mask_img_relative_path

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

# --- U-Net Architecture Definition (Copied from 1_leaf_segmentation.py) ---
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
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
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

# --- Data Loading Function (Modified for prediction focus and consistent test set) ---
def get_test_dataset_for_visualization(metadata_file: Path, base_img_dir: Path, image_size: int,
                                       train_ratio: float, val_ratio: float, test_ratio: float,
                                       manual_seed_for_split: int):
    """
    Loads metadata, filters valid entries, creates dataset, and returns the test split.
    Uses the same split logic as the training script for consistency.
    """
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
    test_size = total_samples - train_size - val_size # Assign remaining to test set

    # Adjust for potential rounding issues
    if train_size + val_size + test_size != total_samples:
        test_size = total_samples - train_size - val_size

    # Split using the same seed as training for consistent test set selection
    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(manual_seed_for_split)
    )

    print(f"Loaded Test Dataset with {len(test_dataset)} samples for visualization.")
    return test_dataset

def create_segmentation_overlay(true_mask_np: np.ndarray, predicted_mask_np: np.ndarray) -> np.ndarray:
    """
    Creates an RGB image overlay visualizing True Positives, False Positives, False Negatives, and background.
    Args:
        true_mask_np (np.ndarray): Binary ground truth mask (0 or 1). Shape (H, W).
        predicted_mask_np (np.ndarray): Binary predicted mask (0 or 1). Shape (H, W).
    Returns:
        np.ndarray: RGB image (H, W, 3) with specified color coding, scaled to [0, 1].
    """
    H, W = true_mask_np.shape
    overlay = np.zeros((H, W, 3), dtype=np.float32)

    # Define colors (RGB values from 0-1)
    WHITE = np.array([1.0, 1.0, 1.0])       # True Positives
    MAGENTA = np.array([1.0, 0.0, 1.0])     # False Positives
    DODGERBLUE = np.array([0.117, 0.565, 1.0]) # False Negatives (approx RGB for dodgerblue)
    GRAY = np.array([0.5, 0.5, 0.5])       # Background (True Negatives)

    # True Positives: true_mask == 1 and predicted_mask == 1
    tp_mask = (true_mask_np == 1) & (predicted_mask_np == 1)
    overlay[tp_mask] = WHITE

    # False Positives: true_mask == 0 and predicted_mask == 1
    fp_mask = (true_mask_np == 0) & (predicted_mask_np == 1)
    overlay[fp_mask] = MAGENTA

    # False Negatives: true_mask == 1 and predicted_mask == 0
    fn_mask = (true_mask_np == 1) & (predicted_mask_np == 0)
    overlay[fn_mask] = DODGERBLUE

    # True Negatives (Background): true_mask == 0 and predicted_mask == 0
    tn_mask = (true_mask_np == 0) & (predicted_mask_np == 0)
    overlay[tn_mask] = GRAY
    
    return overlay


# --- Main Prediction and Visualization Logic ---
if __name__ == "__main__":
    # 1. Load the trained model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    if BEST_MODEL_PATH.exists():
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        print(f"Successfully loaded model from {BEST_MODEL_PATH}")
    else:
        print(f"Error: Model not found at {BEST_MODEL_PATH}. Please run training script (1_leaf_segmentation.py) first.")
        exit()

    # 2. Get the test dataset (using the same seed as training for split consistency)
    test_dataset = get_test_dataset_for_visualization(METADATA_FILE, DATA_ROOT_DIR, IMAGE_SIZE,
                                                       TRAIN_RATIO, VAL_RATIO, TEST_RATIO, MANUAL_SEED) 

    # 3. Randomly select indices from the test dataset for visualization
    if len(test_dataset) < TOTAL_LEAVES_TO_VIZ:
        print(f"Warning: Not enough samples in the test set ({len(test_dataset)}) to visualize {TOTAL_LEAVES_TO_VIZ} leaves.")
        TOTAL_LEAVES_TO_VIZ = len(test_dataset)
        NUM_ROWS_FIGURE = (TOTAL_LEAVES_TO_VIZ + PAIRS_PER_ROW - 1) // PAIRS_PER_ROW # Adjust rows
        print(f"Adjusted to visualize {TOTAL_LEAVES_TO_VIZ} leaves across {NUM_ROWS_FIGURE} rows.")


    selected_indices = random.sample(range(len(test_dataset)), TOTAL_LEAVES_TO_VIZ)
    print(f"\n--- Visualizing {TOTAL_LEAVES_TO_VIZ} samples from the Test Set ---")

    # 4. Set up the figure
    fig, axes = plt.subplots(NUM_ROWS_FIGURE, TOTAL_PANELS_PER_ROW,
                             figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES),
                             dpi=300) # High DPI for quality figure
    
    # Flatten axes array for easier iteration, regardless of single row/column case
    if NUM_ROWS_FIGURE == 1 and TOTAL_PANELS_PER_ROW == 1:
        axes = np.array([[axes]]) # Ensure it's 2D for consistent indexing
    elif NUM_ROWS_FIGURE == 1 or TOTAL_PANELS_PER_ROW == 1:
        axes = axes.reshape(NUM_ROWS_FIGURE, TOTAL_PANELS_PER_ROW) # Ensure it's 2D
    axes = axes.flatten()

    for i, idx in enumerate(selected_indices):
        ect_input, true_mask = test_dataset[idx] # Get the tensors from dataset directly
        
        # Add batch dimension and move to device
        ect_input = ect_input.unsqueeze(0).to(DEVICE)
        true_mask = true_mask.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(ect_input)
        
        predicted_mask = torch.sigmoid(outputs) # Convert logits to probabilities
        predicted_mask = (predicted_mask > 0.5).float() # Binarize

        # Move tensors to CPU and convert to numpy for plotting
        ect_input_np = ect_input.squeeze().cpu().numpy()
        true_mask_np = true_mask.squeeze().cpu().numpy()
        predicted_mask_np = predicted_mask.squeeze().cpu().numpy()

        # --- Plotting Left Panel: Radial ECT (inverted and darkened with outline) ---
        ax_ect = axes[i * 2] # Left panel for each pair
        
        # Set facecolor to white
        ax_ect.set_facecolor('white')

        # Invert the original ECT. If ect_input_np has 0 for black and 1 for white (high ECT)
        # then 1.0 - ect_input_np makes high ECT values black (0) and low ECT values white (1).
        inverted_ect_np = 1.0 - ect_input_np 

        # To "darken" this inverted image (make the white background/low ECT parts less bright)
        # apply a power transformation (gamma correction with gamma > 1 compresses brights)
        darkened_inverted_ect_np = inverted_ect_np ** 2.0 
        
        ax_ect.imshow(darkened_inverted_ect_np, cmap='gray', vmin=0, vmax=1) # Ensure scaling is 0-1
        
        # Plot leaf outline
        # Find contours on the true mask
        contours = measure.find_contours(true_mask_np, 0.5) # Find contours at level 0.5 (between 0 and 1)
        for contour in contours:
            # Matplotlib plot expects (x, y) coordinates, so swap (row, col) to (col, row)
            ax_ect.plot(contour[:, 1], contour[:, 0], color='black', linewidth=0.5)

        ax_ect.axis('off') # No titles

        # --- Plotting Right Panel: Original/Predicted Overlay ---
        ax_overlay = axes[i * 2 + 1] # Right panel for each pair
        segmentation_overlay = create_segmentation_overlay(true_mask_np, predicted_mask_np)
        ax_overlay.imshow(segmentation_overlay)
        ax_overlay.axis('off') # No titles

    # Remove any unused subplots if TOTAL_LEAVES_TO_VIZ is less than max
    for j in range(i * 2 + 2, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=0.5) # Adjust padding
    plt.savefig(OUTPUT_FIGURE_PATH, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    print(f"\nFigure saved to: {OUTPUT_FIGURE_PATH}")
    print("\nVisualization complete. Close the plot to exit.")