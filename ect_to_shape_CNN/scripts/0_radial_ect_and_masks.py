# 0_radial_ect_and_masks.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw # Added ImageChops, ImageDraw for outline and color inversion
import sys
import shutil

# Ensure the ect library is installed and accessible
try:
    from ect import ECT, EmbeddedGraph
except ImportError:
    print("Error: The 'ect' library is not found. Please ensure it's installed and accessible.")
    print("Add its directory to PYTHONPATH or install it correctly (e.g., pip install ect-morphology).")
    sys.exit(1)

# --- Configuration Parameters ---
BOUND_RADIUS = 1  # Max radius for ECT calculation (shapes scaled to fit this)
NUM_ECT_DIRECTIONS = 360 # Number of angular directions for ECT calculation
ECT_THRESHOLDS = np.linspace(0, BOUND_RADIUS, NUM_ECT_DIRECTIONS) # Radial thresholds for ECT, derived from NUM_ECT_DIRECTIONS
IMAGE_SIZE = (256, 256) # Desired pixel size for output images (width, height)

# Input/Output Directories (relative to the script's assumed execution location: ect_to_shape_CNN/scripts/)
RAW_DATA_ROOT_DIR = Path("../data/") # Input root directory to recursively find .npy files
PROCESSED_DATA_OUTPUT_DIR = Path("../outputs/processed_leaf_data/") # Output root directory for all processed data

# Subdirectories for different image types (relative to PROCESSED_DATA_OUTPUT_DIR)
SHAPE_MASK_DIR = PROCESSED_DATA_OUTPUT_DIR / "shape_masks"
RADIAL_ECT_DIR = PROCESSED_DATA_OUTPUT_DIR / "radial_ects"
COMBINED_VIZ_DIR = PROCESSED_DATA_OUTPUT_DIR / "combined_viz"
METADATA_FILE = PROCESSED_DATA_OUTPUT_DIR / "metadata.csv"

# --- Visualization Functions ---

def save_grayscale_shape_mask(processed_points: np.ndarray, save_path: Path):
    """
    Saves a grayscale shape mask as a PNG image using Matplotlib.
    The shape is drawn as white on a black background.
    A specific transformation (flip + rotate) is applied to align with ECT visualization.
    This output is used for CNN input and ground truth.
    """
    fig, ax = plt.subplots(figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Apply Left/Right flip AND 90-degree clockwise rotation for alignment
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
    """
    Saves a grayscale radial ECT image as a PNG using Matplotlib with polar projection.
    This output is used for CNN input.
    """
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"),
                           figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
    thetas = ect_result.directions.thetas
    thresholds = ect_result.thresholds
    THETA, R = np.meshgrid(thetas, thresholds)
    im = ax.pcolormesh(THETA, R, ect_result.T, cmap="gray") # ect_result.T is the ECT matrix
    ax.set_theta_zero_location("N") # North corresponds to 0 degrees
    ax.set_theta_direction(-1) # Clockwise direction
    ax.set_rlim([0, BOUND_RADIUS])
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

def create_combined_viz_from_images(ect_image_path: Path, mask_image_path: Path, save_path: Path):
    """
    Combines a grayscale ECT image (reverse grayscale) and a grayscale shape mask (outline)
    into a single RGB visualization on a white background.
    """
    try:
        # 1. Open ECT image and reverse grayscale
        ect_img = Image.open(ect_image_path).convert("L") # Ensure it's Luminance for inversion
        ect_img = ImageChops.invert(ect_img).convert("RGB") # Invert and convert to RGB
        
        # 2. Create a white background image of the desired size
        combined_img = Image.new("RGB", IMAGE_SIZE, "white")
        
        # 3. Paste the (inverted) ECT image onto the white background
        combined_img.paste(ect_img, (0, 0)) # Assuming ECT image is already 256x256 due to save_grayscale_radial_ect's dpi/figsize settings

        # 4. Open the mask image and draw its outline
        mask_img = Image.open(mask_image_path).convert("L") # Load the original mask
        mask_np = np.array(mask_img)

        # Find the contour/outline from the binary mask
        # This is a simple approach, for more robust contour finding,
        # libraries like OpenCV might be used, but for binary masks from fill, this works.
        outline_mask = np.zeros_like(mask_np)
        
        # Simple edge detection using morphological operations or differences
        # Dilate the mask slightly and subtract the original to get an outline
        from scipy.ndimage import binary_erosion, binary_dilation
        
        eroded_mask = binary_erosion(mask_np, iterations=1)
        outline_np = mask_np - eroded_mask
        
        # Convert the outline to an image and prepare for drawing
        outline_img = Image.fromarray(outline_np.astype(np.uint8) * 255, mode='L')
        
        # Create a blank image to draw the black outline
        outline_draw_img = Image.new("RGBA", IMAGE_SIZE, (0, 0, 0, 0)) # Transparent background
        draw = ImageDraw.Draw(outline_draw_img)
        
        # Get coordinates of white pixels (outline)
        outline_coords = np.argwhere(outline_np > 0)
        # Convert numpy (row, col) to PIL (x, y) coordinates for drawing points/lines
        # PIL coordinates are (column, row) from top-left.
        # Our mask is already aligned for the visualization purposes by save_grayscale_shape_mask,
        # so simply drawing points might not form a continuous line.
        # A better way for outline is to use Image.filter(ImageFilter.FIND_EDGES) but it needs PIL >= 9.2
        # Or, we can use the original points (G.coord_matrix) if we pass them,
        # but the request was to use the *saved image*.
        #
        # A simpler approach using the mask itself: make mask a transparent black overlay.
        # Create a black image where mask is white
        black_mask_color = (0, 0, 0, 255) # Black, opaque
        mask_overlay = Image.new("RGBA", IMAGE_SIZE, (0, 0, 0, 0))
        draw_mask = ImageDraw.Draw(mask_overlay)
        
        # Draw a black filled shape where the original mask was white, then use a small border
        # The original `save_grayscale_shape_mask` drew `transformed_x, transformed_y`.
        # To get the outline from the mask PNG itself, we can either:
        # 1. Load the numpy points again (not ideal if we strictly want to use the PNG)
        # 2. Use image processing to find edges from the mask PNG.

        # Let's try finding contours directly from the loaded mask_img (L mode)
        # This requires converting the binary image to a format suitable for contour finding.
        # For simplicity and given the nature of the mask, we can use a small convolution
        # or just trace the perimeter.
        
        # Simpler approach: create a solid black mask on a transparent background,
        # and then draw a slightly smaller version in white to create an outline effect.
        
        # Create a fully opaque black mask
        black_shape = Image.new("RGBA", IMAGE_SIZE, (0,0,0,0))
        draw_black_shape = ImageDraw.Draw(black_shape)
        
        # The mask_np has 255 for the shape, 0 for background.
        # We need to find the boundary of the 255 region.
        
        # One way is to iterate pixels or use Pillow's capabilities
        # A more direct way to outline with Pillow:
        # Get pixels and iterate to find transitions.
        
        # Alternative for outline: use a transparent black overlay directly from the mask image
        # This will outline whatever shape is *inside* the mask image.
        
        # Convert mask to RGBA, making black transparent and white opaque black.
        mask_rgba = Image.open(mask_image_path).convert("RGBA")
        datas = mask_rgba.getdata()
        new_datas = []
        for item in datas:
            # Change all white (255, 255, 255, 255) to opaque black (0, 0, 0, 255)
            # and all black (0, 0, 0, 255) to transparent (0, 0, 0, 0)
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                new_datas.append((0, 0, 0, 255)) # Opaque black for the shape
            else:
                new_datas.append((0, 0, 0, 0)) # Transparent for background
        mask_outline_img = Image.new("RGBA", IMAGE_SIZE, (0, 0, 0, 0))
        mask_outline_img.putdata(new_datas)

        # The request was to use `plot` for outline, implying direct coordinate plotting.
        # Since we're loading the PNG, we need to extract points from the PNG or re-use G.coord_matrix.
        # Re-using G.coord_matrix is cleaner for direct plotting, but requires passing it.
        # If we *must* use the PNG to derive outline, it's more complex (e.g., OpenCV findContours).

        # Given the instruction "rather than fill, use plot instead to create a leaf shape mask OUTLINE and make the outline 'black'",
        # and "create a combined_viz from images", it implies we need to work with the *image* data.
        # The simplest way to achieve a black outline *from the image* is by a convolution filter or
        # by creating a thicker black version and then punching out a slightly smaller transparent hole.

        # Let's use a simpler approach for outline using Pillow's draw functions,
        # by drawing a border for the original mask.
        
        # Generate outline directly in combined_img using original points is most robust for plotting.
        # However, the functions pass image paths.
        
        # To explicitly use 'plot' on the combined image, we need the transformed points again.
        # This breaks the "create_combined_viz_from_images" premise of only using images.
        # If we stick to using only the images:
        
        # Create a black silhouette
        silhouette_img = Image.open(mask_image_path).convert("L")
        silhouette_img_rgb = Image.new("RGB", IMAGE_SIZE, "black")
        silhouette_img_rgb.paste(Image.new("RGB", IMAGE_SIZE, "black"), mask=silhouette_img)
        
        # Create a slightly eroded white silhouette to create the outline effect
        eroded_mask_np = binary_erosion(mask_np, structure=np.ones((3,3)), iterations=1) # 3x3 square structure
        eroded_mask_img = Image.fromarray(eroded_mask_np.astype(np.uint8) * 255, mode='L')
        
        # Blend the eroded white mask over the black silhouette to create an outline
        # This will create a black outline with a white interior for the shape.
        # If the user wants a transparent interior, this gets complex without an alpha channel.
        
        # Let's revert to a simpler interpretation: if the `mask_image_path` contains the filled mask,
        # and we want an *outline* for the combined viz, we need to extract that outline.
        # The most straightforward way *given the current inputs* is to draw it on top.
        
        # Re-evaluating: The request is "rather than fill, use plot instead to create a leaf shape mask OUTLINE".
        # This implies getting the polygon points. The `save_grayscale_shape_mask` function already does the
        # transformation. For the combined viz, we need those points again or a method to get outline from image.
        
        # A simple method to get outline from a binary image: find edges.
        # This often works by convolution. Pillow has `FIND_EDGES` filter.
        
        # Try `Image.filter(ImageFilter.FIND_EDGES)` if PIL version supports it.
        # Otherwise, a manual convolution or morphology.
        # The prompt implies drawing lines, not just edges from filter.
        
        # Let's assume for `create_combined_viz_from_images`, we need to derive the outline from `mask_image_path`.
        # This is a bit tricky without knowing the exact pixel representation or original polygon.
        # The best way without passing the `G.coord_matrix` again is to use Pillow's image processing.

        # Method 1: Use ImageDraw to outline the shape based on its white pixels.
        # This would require extracting the contours which PIL doesn't do natively as easily as OpenCV.
        
        # Method 2: Create a slightly bigger black mask, and then paste a slightly smaller transparent/white mask inside.
        # This will give a black outline effect.
        
        # Let's use a robust way to get the contour, which requires converting to a format where `find_contours` works.
        # This usually means OpenCV or scikit-image. Since we want to stick to the provided libs:
        
        # *Self-correction*: The request implies drawing using the points. The `save_grayscale_shape_mask`
        # transforms and draws points. To get a *black outline* for the *combined visualization*,
        # we ideally need those *transformed points* again. Since the `mask_image_path` is just a PNG,
        # we can't easily extract the polygon points from it without complex image processing (e.g., contour finding).
        
        # The simplest way to fulfill "use plot instead to create a leaf shape mask OUTLINE and make the outline 'black'"
        # is to have the `create_combined_viz_from_images` accept the `processed_points` directly.
        # However, the current signature only takes image paths.
        
        # Let's modify the signature of `create_combined_viz_from_images` to take `processed_points` if possible,
        # or simulate it from the mask for the outline. Simulating it from mask is harder.
        
        # Okay, let's stick to the constraint of using *images*.
        # How to get a "plotted" outline from a binary PNG?
        # One common trick is to dilate the shape, then subtract the original, which gives a thin outline.
        # And then combine that with the ECT.

        # Re-attempting combined_viz:
        
        # 1. Open ECT image and reverse grayscale
        ect_img = Image.open(ect_image_path).convert("L") # Ensure it's Luminance for inversion
        ect_img_inverted = ImageChops.invert(ect_img)
        
        # Create a blank white background
        combined_img = Image.new("RGB", IMAGE_SIZE, "white")
        
        # Paste the inverted ECT onto the white background.
        # Convert inverted ECT to RGB before pasting if it's desired to maintain its grayscale tones.
        combined_img.paste(ect_img_inverted.convert("RGB"), (0, 0))

        # 2. Prepare the mask outline:
        mask_img = Image.open(mask_image_path).convert("L") # Load the original mask (white shape on black)
        mask_np = np.array(mask_img)

        # Generate outline: Dilate the mask and subtract the original to get a thin border
        # Or, a simpler way for visual outline: draw the shape in black on a transparent layer,
        # then erase the interior by drawing a slightly smaller white shape.
        
        # This method is more aligned with "plot outline" as it directly manipulates the shape drawn on a canvas.
        # Requires creating a dummy Matplotlib figure to get the coordinates scaled correctly for drawing.
        # This is getting complicated if we strictly stick to "from images".
        
        # Let's consider the most direct interpretation of the request:
        # "use plot instead to create a leaf shape mask OUTLINE and make the outline 'black'"
        # This implies we have the coordinates. The `save_grayscale_shape_mask` function already uses
        # `G.coord_matrix`. To avoid re-calculating, the `create_combined_viz_from_images` would need these.
        # This means the signature needs to change, or we re-load/re-process the NPY which is redundant.

        # Let's adjust `process_raw_leaf_shapes` to pass the `G.coord_matrix` to `create_combined_viz_from_images`.
        # This is the most direct way to 'plot' an outline accurately.

        # *Revised strategy for Combined Viz*:
        # 1. `create_combined_viz_from_images` will *also* accept `processed_points`.
        # 2. It will create a blank white canvas.
        # 3. It will paste the *inverted* grayscale ECT image.
        # 4. It will then `plot` the black outline using the `processed_points` on top.

        print("--- Re-evaluating combined_viz approach for clarity and accuracy ---")
        print("To accurately 'plot' a black outline from the shape, the original processed coordinates are needed.")
        print("Modifying `create_combined_viz_from_images` signature to accept `processed_points`.")
        print("The `process_raw_leaf_shapes` function will pass `G.coord_matrix` to it.")

        # This change requires adapting the call in `process_raw_leaf_shapes` and the signature here.
        # The `mask_image_path` will still be used to ensure the ECT is sized correctly, but the outline
        # will come from the raw points. This makes the logic cleaner for drawing the outline.

    except FileNotFoundError:
        print(f"Error: One or both image files not found for combined visualization: {ect_image_path}, {mask_image_path}")
    except Exception as e:
        print(f"Error combining images {ect_image_path} and {mask_image_path}: {e}")

# Re-defining create_combined_viz_from_images to accept processed_points
def create_combined_viz(ect_image_path: Path, processed_points: np.ndarray, save_path: Path):
    """
    Combines a grayscale ECT image (reverse grayscale) and a black shape outline
    into a single RGB visualization on a white background.
    """
    try:
        # 1. Open ECT image and reverse grayscale
        ect_img = Image.open(ect_image_path).convert("L") # Ensure it's Luminance for inversion
        ect_img_inverted = ImageChops.invert(ect_img)
        
        # Create a blank white background image
        combined_pil_img = Image.new("RGB", IMAGE_SIZE, "white")
        
        # Paste the inverted ECT image onto the white background.
        combined_pil_img.paste(ect_img_inverted.convert("RGB"), (0, 0)) # Assumes ECT image is 256x256

        # 2. Draw the black outline using Matplotlib onto a temporary figure
        # and then overlay it. This ensures correct scaling and plotting.
        fig, ax = plt.subplots(figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
        
        fig.patch.set_facecolor('white') # Set background white for the plot
        ax.set_facecolor('white') # Set axis background white

        # Apply Left/Right flip AND 90-degree clockwise rotation for alignment with ECT
        transformed_x = processed_points[:, 1]
        transformed_y = processed_points[:, 0]
        
        # Plot the outline
        ax.plot(transformed_x, transformed_y, color='black', linewidth=1)
        ax.set_xlim([-BOUND_RADIUS, BOUND_RADIUS])
        ax.set_ylim([-BOUND_RADIUS, BOUND_RADIUS])
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save the outline plot to a BytesIO object to load into PIL
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True, dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        outline_img_rgba = Image.open(buf).convert("RGBA")
        
        # Composite the outline onto the combined image
        combined_pil_img = Image.alpha_composite(combined_pil_img.convert("RGBA"), outline_img_rgba)
        
        combined_pil_img.convert("RGB").save(save_path) # Save as RGB

    except FileNotFoundError:
        print(f"Error: ECT image file not found for combined visualization: {ect_image_path}")
    except Exception as e:
        print(f"Error creating combined visualization for {ect_image_path}: {e}")


# --- Main Data Processing Logic ---

def process_raw_leaf_shapes(raw_data_root_dir: Path, output_base_dir: Path, clear_existing_data: bool = True):
    """
    Processes raw leaf shape .npy files found recursively in raw_data_root_dir,
    calculates ECTs, and saves images/metadata.
    """
    print(f"Starting processing of raw leaf shapes from: {raw_data_root_dir} (recursive scan)")
    print(f"Output will be saved to: {output_base_dir}")

    # Setup output directories
    if clear_existing_data and output_base_dir.exists():
        print(f"Clearing existing output directory: {output_base_dir}")
        shutil.rmtree(output_base_dir)

    # Re-define these paths relative to the current output_base_dir for robustness
    shape_mask_dir = output_base_dir / "shape_masks"
    radial_ect_dir = output_base_dir / "radial_ects"
    combined_viz_dir = output_base_dir / "combined_viz"
    metadata_file = output_base_dir / "metadata.csv"

    # Create directories, including parent 'outputs' if it doesn't exist
    shape_mask_dir.mkdir(parents=True, exist_ok=True)
    radial_ect_dir.mkdir(parents=True, exist_ok=True)
    combined_viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directories: {shape_mask_dir}, {radial_ect_dir}, {combined_viz_dir}")

    ect_calculator = ECT(num_dirs=NUM_ECT_DIRECTIONS, thresholds=ECT_THRESHOLDS, bound_radius=BOUND_RADIUS)

    metadata_records = []
    processed_count = 0
    skipped_count = 0

    npy_files = sorted(list(raw_data_root_dir.rglob("*.npy"))) # Recursively find all .npy files
    total_files = len(npy_files)

    if total_files == 0:
        print(f"No .npy files found in {raw_data_root_dir} or its subdirectories. Exiting.")
        return

    print(f"Found {total_files} .npy files to process.")

    for i, npy_file_path in enumerate(npy_files):
        # Use a unique identifier derived from the relative path to handle potential filename conflicts
        # across different subdirectories, and keep it friendly for file naming.
        # Replace '/' or '\' with '__' to make a valid filename.
        relative_path_stem = str(npy_file_path.relative_to(raw_data_root_dir).with_suffix("")).replace(str(Path("/")), "__").replace(str(Path("\\")), "__")
        leaf_id = relative_path_stem

        if (i + 1) % 100 == 0 or (i + 1) == total_files or (i + 1) == 1:
            print(f"Processing leaf shape {i+1}/{total_files} ({npy_file_path.name})")

        raw_shape_points = None
        G = None
        processed_coords_for_viz = None # To store G.coord_matrix for visualization

        try:
            # Load raw shape points (assumed to be N, 2 array)
            raw_shape_points = np.load(npy_file_path)

            if not (raw_shape_points.ndim == 2 and raw_shape_points.shape[1] == 2):
                raise ValueError(f"Invalid .npy file format. Expected (N, 2), got {raw_shape_points.shape}")
            if raw_shape_points.shape[0] < 3:
                raise ValueError(f"Shape has too few points ({raw_shape_points.shape[0]}) to form a valid polygon.")
            if np.any(np.isnan(raw_shape_points)) or np.any(np.isinf(raw_shape_points)):
                raise ValueError("Raw shape points contain NaN or Inf values.")

            # Create EmbeddedGraph, center, transform, and scale coordinates
            G = EmbeddedGraph()
            G.add_cycle(raw_shape_points)
            G.center_coordinates(center_type="origin") # Center the shape at (0,0)
            G.transform_coordinates() # Apply an arbitrary rotation/flip for consistent orientation (if needed by ECT lib)
            G.scale_coordinates(BOUND_RADIUS) # Scale shape to fit within BOUND_RADIUS
            processed_coords_for_viz = G.coord_matrix # Store for combined viz

            # Perform checks after transformation and scaling
            if np.all(G.coord_matrix == 0):
                raise ValueError("Degenerate shape (all points at origin after scaling)")
            if G.coord_matrix.shape[0] < 3:
                raise ValueError(f"Processed shape has too few points ({G.coord_matrix.shape[0]}) to form a valid polygon.")

            # Calculate ECT
            ect_result = ect_calculator.calculate(G)

        except Exception as e:
            num_raw_pts = raw_shape_points.shape[0] if raw_shape_points is not None else 0
            num_proc_pts = G.coord_matrix.shape[0] if G is not None and G.coord_matrix is not None else 0

            print(f"  Skipped processing '{npy_file_path.name}' due to error: {e}")
            skipped_count += 1
            metadata_records.append({
                "leaf_id": leaf_id,
                "raw_file_path": str(npy_file_path.relative_to(RAW_DATA_ROOT_DIR)),
                "is_processed_valid": False,
                "reason_skipped": str(e),
                "num_raw_points": num_raw_pts,
                "num_processed_points": num_proc_pts,
                "file_shape_mask": "", "file_radial_ect": "", "file_combined_viz": ""
            })
            continue

        # Define output file paths
        output_image_name = f"{leaf_id}.png"
        mask_path = shape_mask_dir / output_image_name
        ect_path = radial_ect_dir / output_image_name
        viz_path = combined_viz_dir / output_image_name

        try:
            # Save individual image components (these are kept as-is)
            save_grayscale_shape_mask(G.coord_matrix, mask_path)
            save_grayscale_radial_ect(ect_result, ect_path)

            # Create combined visualization (with new logic)
            # Pass processed_coords_for_viz for accurate outline plotting
            create_combined_viz(ect_path, processed_coords_for_viz, viz_path)

        except Exception as e:
            print(f"  Error saving images for '{npy_file_path.name}': {e}. Marking as invalid.")
            skipped_count += 1
            metadata_records.append({
                "leaf_id": leaf_id,
                "raw_file_path": str(npy_file_path.relative_to(RAW_DATA_ROOT_DIR)),
                "is_processed_valid": False,
                "reason_skipped": f"Image saving failed: {e}",
                "num_raw_points": raw_shape_points.shape[0],
                "num_processed_points": G.coord_matrix.shape[0],
                "file_shape_mask": "", "file_radial_ect": "", "file_combined_viz": ""
            })
            continue

        # Record successful processing in metadata
        metadata_records.append({
            "leaf_id": leaf_id,
            "raw_file_path": str(npy_file_path.relative_to(RAW_DATA_ROOT_DIR)), # Store relative path for portability
            "is_processed_valid": True,
            "reason_skipped": "",
            "num_raw_points": raw_shape_points.shape[0],
            "num_processed_points": G.coord_matrix.shape[0],
            "file_shape_mask": str(mask_path.relative_to(output_base_dir)),
            "file_radial_ect": str(ect_path.relative_to(output_base_dir)),
            "file_combined_viz": str(viz_path.relative_to(output_base_dir))
        })
        processed_count += 1

    # Save all metadata to a CSV file
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(metadata_file, index=False)

    print(f"\n--- Processing Complete ---")
    print(f"Total files considered: {total_files}")
    print(f"Shapes successfully processed and saved: {processed_count}")
    print(f"Shapes skipped (invalid processing/saving): {skipped_count}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"Processed images saved in: {output_base_dir}/{{shape_masks, radial_ects, combined_viz}}")

if __name__ == "__main__":
    # Ensure the parent 'outputs' directory exists before creating subdirectories
    PROCESSED_DATA_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_ROOT_DIR.exists():
        print(f"Error: Input data root directory '{RAW_DATA_ROOT_DIR}' not found.")
        print(f"Please ensure '{RAW_DATA_ROOT_DIR.resolve()}' exists and contains your .npy files.")
        sys.exit(1)

    process_raw_leaf_shapes(RAW_DATA_ROOT_DIR, PROCESSED_DATA_OUTPUT_DIR)