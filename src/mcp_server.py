import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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


@mcp.tool()
def segment_ct_dataset(input_filepath: str, output_filepath: str, threshold: float) -> str:
    """
    Segments a 3D CT dataset based on a given density threshold value.

    Args:
        input_filepath: Path to the input .npy file containing the 3D CT scan data.
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
        volume = np.load(input_filepath)
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

def _load_volume(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext in (".tif", ".tiff"):
        return np.asarray(__import__("tifffile").imread(path))
    raise ValueError("unsupported file type")


def _json_position_to_volume_position(position: np.ndarray) -> np.ndarray:
    """Convert a JSON node position from [x, y, z] to TIFF memmap order [z, y, x]."""
    return np.asarray(position, dtype=float)[[2, 1, 0]]


def _pick_slice(volume: np.ndarray, index: int, axis: int) -> np.ndarray:
    return np.take(volume, index, axis=axis)

def _load_graph(path: str):
    """Loads a lattice-graph JSON file into {junction_id: xyz position} and a
    list of (junction0_id, junction1_id) strut pairs."""
    with open(path) as fh:
        data = json.load(fh)
    junctions = {
        j["id"]: _json_position_to_volume_position(np.array(j["position"], dtype=float))
        for j in data["junctions"]
    }
    struts = [(s["junction0"], s["junction1"]) for s in data["struts"]]
    return junctions, struts

def _save_slice_image(array: np.ndarray, destination: str, label: str) -> None:
    folder = os.path.dirname(os.path.abspath(destination))
    os.makedirs(folder, exist_ok=True)

    fig = plt.figure(figsize=(5, 5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(array, cmap="gray")
    ax.set_title(label)
    ax.axis("off")
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)

@mcp.tool()
def visualize_slice(input_filepath: str, output_filepath: str, slice_index: int, axis: int = 0) -> str:
    if not os.path.exists(input_filepath):
        return f"Error: input file not found at {input_filepath}"
    if axis not in (0, 1, 2):
        return f"Error: axis must be 0, 1, or 2 (got {axis})"

    volume = _load_volume(input_filepath)
    if volume.ndim != 3:
        return f"Error: expected a 3D array, got shape {volume.shape}"

    if not (0 <= slice_index < volume.shape[axis]):
        return f"Error: slice_index {slice_index} out of range for axis {axis}"

    plane = _pick_slice(volume, slice_index, axis)
    _save_slice_image(plane, output_filepath, f"slice {slice_index}")
    return f"Saved slice {slice_index} visualization to {output_filepath}"

@mcp.tool()
def skeletonize(input_filepath: str, output_filepath: str) -> str:
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

@mcp.tool()
def overlay_graph_on_slice(
    volume_filepath: str,
    json_filepath: str,
    output_filepath: str,
    slice_index: int,
    axis: int = 0,
    slab_tolerance: float = 3.0,
) -> str:
    """
    Overlays a lattice graph (junctions + struts) from a JSON file onto a single
    2D slice of its matching CT volume, so the graph's coordinates and the
    volume's voxel coordinates can be visually compared on one image.

    IMPORTANT: This only produces a meaningful (correctly aligned) result for
    JSON/volume pairs whose coordinate systems are already registered to each
    other, e.g. data/missing_struts/registered_jsons/*.json paired with its
    matching .tif in data/missing_struts/tif_stacks/. JSON files describing
    lattice designs in their own CAD-space coordinates (e.g.
    octet_truss_9x9x9.json, octet_truss_8x8x8.json, polyhedron_1x1x1.json) are
    NOT registered to any volume's voxel grid and should not be passed here
    without a separate registration/scaling step.

    A CT slice is a 2D plane, but junctions are 3D points, so almost none will
    land exactly on the slice. Instead, this draws every junction whose
    coordinate along the slicing axis falls within +/- slab_tolerance voxels
    of slice_index (a thin "slab" around the slice), and every strut whose two
    endpoints straddle or fall within that slab.

    Args:
        volume_filepath: Path to the .npy or .tif/.tiff CT volume.
        json_filepath: Path to the registered lattice-graph JSON file (junctions + struts).
        output_filepath: Path indicating where the overlay image should be saved (e.g. .png).
        slice_index: The index of the slice to visualize.
        axis: The axis along which to take the slice (0, 1, or 2). Default is 0.
        slab_tolerance: How many voxels on either side of slice_index count as
            "on this slice" for selecting junctions/struts to draw. Default is 3.0.

    Returns:
        A status message indicating success (with counts of junctions/struts drawn)
        and the save location, or an error message.
    """
    if not os.path.isfile(volume_filepath):
        return f"Error: volume file not found at {volume_filepath}"
    if not os.path.isfile(json_filepath):
        return f"Error: json file not found at {json_filepath}"
    if axis not in (0, 1, 2):
        return f"Error: axis must be 0, 1, or 2 (got {axis})"
    if not np.isfinite(slab_tolerance) or slab_tolerance < 0:
        return f"Error: slab_tolerance must be a non-negative finite number (got {slab_tolerance})"

    try:
        volume = _load_volume(volume_filepath)
    except (OSError, ValueError) as error:
        return f"Error: could not load volume: {error}"

    if volume.ndim != 3:
        return f"Error: expected a 3D volume, got shape {volume.shape}"
    if not (0 <= slice_index < volume.shape[axis]):
        return (
            f"Error: slice_index {slice_index} out of range for axis {axis} "
            f"(size {volume.shape[axis]})"
        )

    try:
        junctions, struts = _load_graph(json_filepath)
    except (OSError, ValueError, KeyError) as error:
        return f"Error: could not load lattice graph json: {error}"

    if not junctions:
        return "Error: json file has no junctions"

    # The two axes that remain after taking a slice along `axis`. np.take
    # preserves the relative order of the axes that are not sliced away, so
    # the smaller-numbered remaining axis becomes the 2D plane's row
    # (imshow's vertical/Y), and the larger-numbered one becomes its column
    # (imshow's horizontal/X). plot_axes is therefore [larger, smaller] so
    # that plot_axes[0] can be used directly as the x-coordinate and
    # plot_axes[1] as the y-coordinate when overlaying points on the image.
    other_axes = [a for a in (0, 1, 2) if a != axis]
    plot_axes = [other_axes[1], other_axes[0]]

    slab_min = slice_index - slab_tolerance
    slab_max = slice_index + slab_tolerance

    in_slab_junction_ids = {
        jid for jid, pos in junctions.items() if slab_min <= pos[axis] <= slab_max
    }

    # Struts whose axis-range overlaps the slab. Rather than drawing the full
    # 3D strut projected flat (which misrepresents its true position except
    # very close to a junction plane), clip each strut to the portion that
    # actually falls within the slab, so the drawn segment reflects only
    # where the strut truly crosses this slice.
    drawn_struts = []
    for j0, j1 in struts:
        if j0 not in junctions or j1 not in junctions:
            continue
        p0, p1 = junctions[j0], junctions[j1]
        a0, a1 = p0[axis], p1[axis]
        if a0 == a1:
            if slab_min <= a0 <= slab_max:
                drawn_struts.append((p0[plot_axes], p1[plot_axes]))
            continue
        t_lo = (slab_min - a0) / (a1 - a0)
        t_hi = (slab_max - a0) / (a1 - a0)
        t_lo, t_hi = sorted((t_lo, t_hi))
        t_lo = max(t_lo, 0.0)
        t_hi = min(t_hi, 1.0)
        if t_lo > t_hi:
            continue
        clipped_p0 = p0 + t_lo * (p1 - p0)
        clipped_p1 = p0 + t_hi * (p1 - p0)
        drawn_struts.append((clipped_p0[plot_axes], clipped_p1[plot_axes]))

    plane = _pick_slice(volume, slice_index, axis)

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    ax.imshow(plane, cmap="gray")

    if drawn_struts:
        segments = [[tuple(p0), tuple(p1)] for p0, p1 in drawn_struts]
        ax.add_collection(LineCollection(segments, colors="lime", linewidths=1.2, alpha=0.9))

    if in_slab_junction_ids:
        points = np.array([junctions[jid][plot_axes] for jid in in_slab_junction_ids])
        ax.scatter(
            points[:, 0], points[:, 1],
            s=14, c="red", edgecolors="black", linewidths=0.4, zorder=3,
        )

    ax.set_xlim(0, plane.shape[1])
    ax.set_ylim(plane.shape[0], 0)
    ax.set_title(
        f"slice {slice_index} (axis={axis}) + graph overlay (slab tol={slab_tolerance})\n"
        f"{len(in_slab_junction_ids)} junctions, {len(drawn_struts)} struts drawn"
    )
    ax.axis("off")

    output_directory = os.path.dirname(os.path.abspath(output_filepath))
    try:
        os.makedirs(output_directory, exist_ok=True)
        fig.savefig(output_filepath, dpi=150, bbox_inches="tight")
    except OSError as error:
        plt.close(fig)
        return f"Error: could not save overlay image: {error}"
    plt.close(fig)

    return (
        f"Saved graph overlay for slice {slice_index} (axis={axis}) to {output_filepath} "
        f"({len(in_slab_junction_ids)} junctions, {len(drawn_struts)} struts within "
        f"+/-{slab_tolerance} voxels)"
    )

if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
