# radial_ect_and_masks.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image # For image loading and manipulation
import sys
import shutil

# Ensure the ect library is installed and accessible
try:
    from ect import ECT, EmbeddedGraph
    # ECTResult is imported but not strictly necessary for this file since we
    # directly use the ect_result object in save_grayscale_radial_ect
    # from ect.results import ECTResult
except ImportError:
    print("Error: The 'ect' library is not found. Please ensure it's installed and accessible.")
    print("Add its directory to PYTHONPATH or install it correctly (e.g., pip install ect-morphology).")
    sys.exit(1)

# --- Configuration Parameters (Consolidated from generate_superformula_data.py and process_leaf_shapes.py) ---
BOUND_RADIUS = 1  # Max radius for ECT calculation (shapes scaled to fit this)
NUM_ECT_DIRECTIONS = 90
ECT_THRESHOLDS = np.linspace(0, BOUND_RADIUS, NUM_ECT_DIRECTIONS)
IMAGE_SIZE = (256, 256) # Desired pixel size for output images (width, height)

# Input/Output Directories for processing real leaf shapes
RAW_LEAF_SHAPES_DIR = Path("raw_leaf_shapes/") # Input directory with .npy files
PROCESSED_DATA_OUTPUT_DIR = Path("processed_leaf_data/") # Output root directory for all processed data

# Subdirectories for different image types (relative to PROCESSED_DATA_OUTPUT_DIR)
SHAPE_MASK_DIR = PROCESSED_DATA_OUTPUT_DIR / "shape_masks"
RADIAL_ECT_DIR = PROCESSED_DATA_OUTPUT_DIR / "radial_ects"
COMBINED_VIZ_DIR = PROCESSED_DATA_OUTPUT_DIR / "combined_viz"
METADATA_FILE = PROCESSED_DATA_OUTPUT_DIR / "metadata.csv"

# --- Visualization Functions (Moved from visualization_utils.py) ---

def save_grayscale_shape_mask(processed_points: np.ndarray, save_path: Path):
    """
    Saves a grayscale shape mask using Matplotlib.
    The shape is drawn as white on a black background.
    The transformation (flip + rotate) is applied here.
    """
    fig, ax = plt.subplots(figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Apply Left/Right flip AND 90-degree clockwise rotation
    # (x_orig, y_orig) -> (-x_orig, y_orig) -> (y_orig, -(-x_orig)) -> (y_orig, x_orig)
    transformed_x = processed_points[:, 1] # This is y_original
    transformed_y = processed_points[:, 0] # This is x_original

    ax.fill(transformed_x, transformed_y, color='white')

    ax.set_xlim([-BOUND_RADIUS, BOUND_RADIUS])
    ax.set_ylim([-BOUND_RADIUS, BOUND_RADIUS])

    ax.set_aspect('equal', adjustable='box')

    ax.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close(fig)

def save_grayscale_radial_ect(ect_result, save_path: Path):
    """Saves a grayscale radial ECT image using Matplotlib."""
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"),
                           figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
    thetas = ect_result.directions.thetas
    thresholds = ect_result.thresholds
    THETA, R = np.meshgrid(thetas, thresholds)
    im = ax.pcolormesh(THETA, R, ect_result.T, cmap="gray")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim([0, BOUND_RADIUS])
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

def create_combined_viz_from_images(ect_image_path: Path, mask_image_path: Path, save_path: Path,
                                    mask_color=(255, 0, 255), mask_alpha=0.8):
    """
    Combines a grayscale ECT image and a grayscale shape mask into a single RGB visualization.
    The mask is overlaid in a specified color with transparency.
    """
    try:
        ect_img = Image.open(ect_image_path).convert("RGB")
        mask_img = Image.open(mask_image_path).convert("L") # Ensure mask is grayscale (Luminance)

        ect_np = np.array(ect_img)
        mask_np = np.array(mask_img) # This is 0 (black) or 255 (white)

        overlay_color_np = np.array(mask_color, dtype=np.uint8)
        colored_overlay = np.zeros_like(ect_np)
        colored_overlay[mask_np == 255] = overlay_color_np

        alpha_val = int(mask_alpha * 255)
        blend_alpha = np.zeros_like(mask_np)
        blend_alpha[mask_np == 255] = alpha_val
        blend_alpha_img = Image.fromarray(blend_alpha, mode='L')

        combined_img = Image.composite(Image.fromarray(colored_overlay), ect_img, blend_alpha_img)

        combined_img.save(save_path)

    except FileNotFoundError:
        print(f"Error: One or both image files not found: {ect_image_path}, {mask_image_path}")
    except Exception as e:
        print(f"Error combining images {ect_image_path} and {mask_image_path}: {e}")

# --- Main Data Processing Logic (Moved and adapted from process_leaf_shapes.py) ---

def process_raw_leaf_shapes(raw_input_dir: Path, output_base_dir: Path, clear_existing_data: bool = True):
    """
    Processes raw leaf shape .npy files, calculates ECTs, and saves images/metadata.
    """
    print(f"Starting processing of raw leaf shapes from: {raw_input_dir}")
    print(f"Output will be saved to: {output_base_dir}")

    # Setup output directories
    if clear_existing_data and output_base_dir.exists():
        print(f"Clearing existing output directory: {output_base_dir}")
        shutil.rmtree(output_base_dir)

    # Re-define these paths relative to the current output_base_dir for robustness
    # (even though they are globally defined constants above, they are used here directly)
    shape_mask_dir = output_base_dir / "shape_masks"
    radial_ect_dir = output_base_dir / "radial_ects"
    combined_viz_dir = output_base_dir / "combined_viz"
    metadata_file = output_base_dir / "metadata.csv"

    shape_mask_dir.mkdir(parents=True, exist_ok=True)
    radial_ect_dir.mkdir(parents=True, exist_ok=True)
    combined_viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directories: {shape_mask_dir}, {radial_ect_dir}, {combined_viz_dir}")

    ect_calculator = ECT(num_dirs=NUM_ECT_DIRECTIONS, thresholds=ECT_THRESHOLDS, bound_radius=BOUND_RADIUS)

    metadata_records = []
    processed_count = 0
    skipped_count = 0

    npy_files = list(raw_input_dir.glob("*.npy"))
    total_files = len(npy_files)

    if total_files == 0:
        print(f"No .npy files found in {raw_input_dir}. Exiting.")
        return

    print(f"Found {total_files} .npy files to process.")

    for i, npy_file_path in enumerate(npy_files):
        leaf_id = npy_file_path.stem

        if (i + 1) % 100 == 0 or (i + 1) == total_files or (i + 1) == 1:
            print(f"Processing leaf shape {i+1}/{total_files} ({leaf_id})")

        raw_shape_points = None
        G = None

        try:
            raw_shape_points = np.load(npy_file_path)

            if not (raw_shape_points.ndim == 2 and raw_shape_points.shape[1] == 2):
                raise ValueError(f"Invalid .npy file format. Expected (N, 2), got {raw_shape_points.shape}")
            if raw_shape_points.shape[0] < 3:
                raise ValueError(f"Shape has too few points ({raw_shape_points.shape[0]}) to form a valid polygon.")
            if np.any(np.isnan(raw_shape_points)) or np.any(np.isinf(raw_shape_points)):
                raise ValueError("Raw shape points contain NaN or Inf values.")

            G = EmbeddedGraph()
            G.add_cycle(raw_shape_points)
            G.center_coordinates(center_type="origin")
            G.transform_coordinates()
            G.scale_coordinates(BOUND_RADIUS)

            if np.all(G.coord_matrix == 0):
                raise ValueError("Degenerate shape (all points at origin after scaling)")
            if G.coord_matrix.shape[0] < 3:
                raise ValueError(f"Processed shape has too few points ({G.coord_matrix.shape[0]}) to form a valid polygon.")

            ect_result = ect_calculator.calculate(G)

        except Exception as e:
            num_raw_pts = raw_shape_points.shape[0] if raw_shape_points is not None else 0
            num_proc_pts = G.coord_matrix.shape[0] if G is not None and G.coord_matrix is not None else 0

            print(f"  Skipped processing '{leaf_id}' due to error: {e}")
            skipped_count += 1
            metadata_records.append({
                "leaf_id": leaf_id,
                "raw_file_path": str(npy_file_path),
                "is_processed_valid": False,
                "reason_skipped": str(e),
                "num_raw_points": num_raw_pts,
                "num_processed_points": num_proc_pts,
                "file_shape_mask": "", "file_radial_ect": "", "file_combined_viz": ""
            })
            continue

        output_image_name = f"{leaf_id}.png"
        mask_path = shape_mask_dir / output_image_name
        ect_path = radial_ect_dir / output_image_name
        viz_path = combined_viz_dir / output_image_name

        try:
            # Save individual image components
            save_grayscale_shape_mask(G.coord_matrix, mask_path)
            save_grayscale_radial_ect(ect_result, ect_path)

            # Create combined visualization FROM THE SAVED IMAGES
            create_combined_viz_from_images(ect_path, mask_path, viz_path)

        except Exception as e:
            print(f"  Error saving images for '{leaf_id}': {e}. Marking as invalid.")
            skipped_count += 1
            metadata_records.append({
                "leaf_id": leaf_id,
                "raw_file_path": str(npy_file_path),
                "is_processed_valid": False,
                "reason_skipped": f"Image saving failed: {e}",
                "num_raw_points": raw_shape_points.shape[0],
                "num_processed_points": G.coord_matrix.shape[0],
                "file_shape_mask": "", "file_radial_ect": "", "file_combined_viz": ""
            })
            continue

        metadata_records.append({
            "leaf_id": leaf_id,
            "raw_file_path": str(npy_file_path),
            "is_processed_valid": True,
            "reason_skipped": "",
            "num_raw_points": raw_shape_points.shape[0],
            "num_processed_points": G.coord_matrix.shape[0],
            "file_shape_mask": str(mask_path.relative_to(output_base_dir)),
            "file_radial_ect": str(ect_path.relative_to(output_base_dir)),
            "file_combined_viz": str(viz_path.relative_to(output_base_dir))
        })
        processed_count += 1

    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(metadata_file, index=False)

    print(f"\n--- Processing Complete ---")
    print(f"Total files considered: {total_files}")
    print(f"Shapes successfully processed and saved: {processed_count}")
    print(f"Shapes skipped (invalid processing/saving): {skipped_count}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"Processed images saved in: {output_base_dir}/{{shape_masks, radial_ects, combined_viz}}")

if __name__ == "__main__":
    if not RAW_LEAF_SHAPES_DIR.exists():
        print(f"Error: Input directory '{RAW_LEAF_SHAPES_DIR}' not found.")
        print("Please create this directory and place your .npy files inside.")
        sys.exit(1)

    # You can change PROCESSED_DATA_OUTPUT_DIR if you want to test output in a different location
    # For example: process_raw_leaf_shapes(RAW_LEAF_SHAPES_DIR, Path("test_output"))
    process_raw_leaf_shapes(RAW_LEAF_SHAPES_DIR, PROCESSED_DATA_OUTPUT_DIR)