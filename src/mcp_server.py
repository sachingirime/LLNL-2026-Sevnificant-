import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
from fastmcp import FastMCP

try:
    # Package import for tests and programmatic use.
    from .skeletonization import skeletonize_mask
except ImportError:
    # Script import when FastMCP starts this file via ``python src/mcp_server.py``.
    from skeletonization import skeletonize_mask

# Initialize the MCP server
mcp = FastMCP("CT Segmentation")


def _load_volume(input_filepath: str) -> np.ndarray:
    """Loads a 3D array from a .npy or .tif/.tiff file."""
    ext = os.path.splitext(input_filepath)[1].lower()
    if ext == ".npy":
        return np.load(input_filepath)
    if ext in (".tif", ".tiff"):
        return tifffile.imread(input_filepath)
    raise ValueError(f"Unsupported file type '{ext}'. Expected .npy, .tif, or .tiff.")


@mcp.tool()
def segment_ct_dataset(input_filepath: str, output_filepath: str, threshold: float) -> str:
    """
    Segments a 3D CT dataset based on a given density threshold value.

    Args:
        input_filepath: Path to the input .npy or .tif/.tiff file containing the 3D CT scan data.
        output_filepath: Path indicating where the segmented .npy file should be saved.
        threshold: The density value to use as a threshold. Voxels >= threshold will be set to 1, others to 0.

    Returns:
        A status message indicating success and the save location, or an error message.
    """
    if not os.path.isfile(input_filepath):
        return f"Error: input file not found at {input_filepath}"
    if not np.isfinite(threshold):
        return f"Error: threshold must be a finite number (got {threshold})"
    if os.path.splitext(output_filepath)[1].lower() != ".npy":
        return "Error: output_filepath must end with '.npy'"

    try:
        volume = _load_volume(input_filepath)
    except (OSError, ValueError) as error:
        return f"Error: could not load input array: {error}"

    if volume.ndim != 3:
        return f"Error: expected a 3D array, got shape {volume.shape}"

    # uint8 is compact and makes the output an explicit binary mask: 1 is
    # lattice material and 0 is background.
    segmentation = (volume >= threshold).astype(np.uint8)
    output_directory = os.path.dirname(os.path.abspath(output_filepath))

    try:
        os.makedirs(output_directory, exist_ok=True)
        np.save(output_filepath, segmentation)
    except OSError as error:
        return f"Error: could not save segmentation: {error}"

    foreground_voxels = int(segmentation.sum())
    return (
        f"Saved binary segmentation to {output_filepath} "
        f"(threshold={threshold}, shape={segmentation.shape}, "
        f"foreground_voxels={foreground_voxels})"
    )

@mcp.tool()
def visualize_slice(input_filepath: str, output_filepath: str, slice_index: int, axis: int = 0) -> str:
    """
    Loads a 3D CT dataset from a .npy or .tif file and saves a visualization of a specific slice to an image file.

    Args:
        input_filepath: Path to the input .npy or .tif file containing the 3D CT data.
        output_filepath: Path indicating where the output image should be saved (e.g., .png).
        slice_index: The index of the slice to visualize.
        axis: The axis along which to take the slice (0, 1, or 2). Default is 0.

    Returns:
        A status message indicating success and the save location, or an error message.
    """
    if not os.path.exists(input_filepath):
        return f"Error: input file not found at {input_filepath}"
    if axis not in (0, 1, 2):
        return f"Error: axis must be 0, 1, or 2 (got {axis})"

    try:
        volume = _load_volume(input_filepath)
    except ValueError as e:
        return f"Error: {e}"

    if volume.ndim != 3:
        return f"Error: expected a 3D array, got shape {volume.shape}"
    if not (0 <= slice_index < volume.shape[axis]):
        return (
            f"Error: slice_index {slice_index} out of range for axis {axis} "
            f"with size {volume.shape[axis]}"
        )

    slice_2d = np.take(volume, slice_index, axis=axis)

    out_dir = os.path.dirname(os.path.abspath(output_filepath))
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(slice_2d, cmap="gray")
    plt.title(f"Slice {slice_index} (axis={axis}) of {os.path.basename(input_filepath)}")
    plt.axis("off")
    plt.savefig(output_filepath, bbox_inches="tight", dpi=150)
    plt.close()

    return f"Saved slice {slice_index} (axis={axis}) visualization to {output_filepath}"

@mcp.tool()
def skeletonize(input_filepath: str, output_filepath: str) -> str:
    """
    Creates a skeleton from a 3D segmentation mask.
    
    Args:
        input_filepath: Path to the .npy file containing the 3D mask.
        output_filepath: Path to save the extracted skeleton (.npy).
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    if not os.path.isfile(input_filepath):
        return f"Error: input file not found at {input_filepath}"
    if os.path.splitext(input_filepath)[1].lower() != ".npy":
        return "Error: input_filepath must reference a '.npy' segmentation mask"
    if os.path.splitext(output_filepath)[1].lower() != ".npy":
        return "Error: output_filepath must end with '.npy'"

    try:
        mask = np.load(input_filepath, mmap_mode="r")
    except (OSError, ValueError) as error:
        return f"Error: could not load segmentation mask: {error}"

    if mask.ndim != 3:
        return f"Error: expected a 3D mask, got shape {mask.shape}"
    if not np.any(mask):
        return "Error: input mask has no foreground voxels to skeletonize"

    output_directory = os.path.dirname(os.path.abspath(output_filepath))
    os.makedirs(output_directory, exist_ok=True)

    try:
        extracted_skeleton = skeletonize_mask(input_filepath, output_filepath)
    except (OSError, ValueError, TypeError) as error:
        return f"Error: could not skeletonize mask: {error}"

    if extracted_skeleton is None:
        return "Error: skeletonization did not produce an output"

    return (
        f"Saved 3D skeleton to {output_filepath} "
        f"(shape={extracted_skeleton.shape}, "
        f"skeleton_voxels={int(np.count_nonzero(extracted_skeleton))})"
    )

if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
