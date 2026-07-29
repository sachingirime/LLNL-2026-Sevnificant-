import base64
import collections
import io
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
from fastmcp import FastMCP

try:
    # Package import for tests and programmatic use.
    from .skeletonization import skeletonize_mask
    from . import lattice_iou
    from . import node_planes_2d
    from . import stl_ground_truth
except ImportError:
    # Script import when FastMCP starts this file via ``python src/mcp_server.py``.
    from skeletonization import skeletonize_mask
    import lattice_iou
    import node_planes_2d
    import stl_ground_truth

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


def _save_volume(output_filepath: str, volume: np.ndarray) -> None:
    """Saves a 3D array as .npy or compressed .tif/.tiff."""
    ext = os.path.splitext(output_filepath)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    if ext == ".npy":
        np.save(output_filepath, volume)
    elif ext in (".tif", ".tiff"):
        tifffile.imwrite(output_filepath, volume, compression="zlib")
    else:
        raise ValueError(f"Unsupported output type '{ext}'. Expected .npy, .tif, or .tiff.")


# Physical geometry of the octet lattice, used to align rasterization metrics.
# An octet strut spans a unit-cell edge diagonally, so its length is
# cell_edge / sqrt(2). Measuring the median strut length in a JSON's own
# coordinate units and dividing the known physical length by it recovers the
# scale (voxels per JSON unit) -- this makes the output metrics correct whether
# the JSON is in nominal half-cell units (positions 0..18) or already in CT
# voxels (positions ~24..774), with no hand-tuned radius to get wrong.
def _octet_strut_length_vox(cell_edge_mm: float, voxel_size_um: float) -> float:
    return (cell_edge_mm / np.sqrt(2.0)) * 1000.0 / voxel_size_um


def _rasterize_struts(positions_zyx, strut_pairs, shape, radius_vox, origin):
    """Paint each strut as a capsule of ``radius_vox`` into a uint8 volume.

    positions_zyx : (N,3) float array, already scaled to voxels and in (z,y,x).
    strut_pairs   : (M,2) int array of junction indices.
    shape         : (Z,Y,X) of the output volume.
    origin        : (3,) offset subtracted from positions before indexing.
    """
    vol = np.zeros(shape, dtype=np.uint8)
    Z, Y, X = shape
    pad = int(np.ceil(radius_vox)) + 1
    r2 = radius_vox * radius_vox
    a = positions_zyx[strut_pairs[:, 0]] - origin
    b = positions_zyx[strut_pairs[:, 1]] - origin
    for p0, p1 in zip(a, b):
        lo = np.maximum(np.floor(np.minimum(p0, p1)).astype(int) - pad, 0)
        hi = np.minimum(np.ceil(np.maximum(p0, p1)).astype(int) + pad, [Z, Y, X])
        if np.any(hi <= lo):
            continue
        zz, yy, xx = np.meshgrid(
            np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), np.arange(lo[2], hi[2]),
            indexing="ij",
        )
        grid = np.stack([zz, yy, xx], axis=-1).astype(np.float64)
        seg = p1 - p0
        L2 = float(seg @ seg)
        t = np.clip(((grid - p0) @ seg) / L2, 0.0, 1.0) if L2 > 0 else np.zeros(grid.shape[:-1])
        closest = p0 + t[..., None] * seg
        d2 = ((grid - closest) ** 2).sum(-1)
        sub = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        sub[d2 <= r2] = 1
    return vol


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

def _volume_to_pv_surface(volume, max_dim):
    """Downsample a binary volume and marching-cubes it into a pyvista surface.

    Returns (pyvista.PolyData, downsample_factor). Raises ValueError if the volume is
    empty or yields no surface, ImportError if a rendering dependency is missing. Max-pool
    downsampling preserves thin struts, and marching cubes avoids building a giant
    intermediate grid, so this stays memory-bounded on full 700+ voxel lattices.
    """
    import pyvista as pv
    from skimage.measure import block_reduce, marching_cubes

    mask = volume > 0
    if not mask.any():
        raise ValueError("volume has no foreground voxels to render")
    factor = max(1, int(np.ceil(max(mask.shape) / max_dim)))
    if factor > 1:
        mask = block_reduce(mask, (factor, factor, factor), np.max)
    padded = np.pad(mask.astype(np.float32), 1)  # pad so border surfaces close
    verts, faces, _, _ = marching_cubes(padded, level=0.5)
    if len(faces) == 0:
        raise ValueError("no surface extracted from the volume")
    faces_pv = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)]
    ).ravel()
    return pv.PolyData(verts, faces_pv), factor


@mcp.tool()
def rasterize_lattice(
    input_filepath: str,
    output_filepath: str,
    voxel_size_um: float = 58.1,
    strut_diameter_um: float = 350.0,
    cell_edge_mm: float = 4.56,
    shape_z: int = 0,
    shape_y: int = 0,
    shape_x: int = 0,
) -> str:
    """
    Rasterizes a lattice graph (JSON) into a 3D binary voxel volume, painting each
    strut as a solid capsule. The coordinates in the JSON are used exactly as given
    (no registration or tilt correction is applied).

    The scale is derived from the lattice's own geometry so the physical metrics come
    out correct regardless of the JSON's coordinate units: an octet strut is
    cell_edge/sqrt(2) long, so the median strut length in the file's units is matched
    to that physical length. This yields a strut radius of strut_diameter_um/2 at the
    given voxel size whether the JSON is in nominal half-cell units or in CT voxels.

    Args:
        input_filepath: Path to the lattice .json with "junctions" (each a "position"
            [x, y, z]) and "struts" (each with "junction0" and "junction1" indices).
        output_filepath: Where to save the volume (.npy or .tif/.tiff). Saved as uint8
            (1 = lattice material, 0 = background), indexed [z, y, x].
        voxel_size_um: Physical voxel size in microns (CT default 58.1).
        strut_diameter_um: Nominal strut diameter in microns (default 350).
        cell_edge_mm: Unit-cell edge length in mm (default 4.56); sets the scale.
        shape_z, shape_y, shape_x: If all > 0, rasterize into a volume of exactly this
            shape at absolute origin (0,0,0) -- use this to align with a CT volume of
            the same shape. If left 0 (default), auto-size to the lattice bounding box
            with padding, translating the lattice to the volume origin.

    Returns:
        A status message with the derived scale, radius, shape, and foreground fraction,
        or an error message.
    """
    if not os.path.isfile(input_filepath):
        return f"Error: input file not found at {input_filepath}"
    if os.path.splitext(input_filepath)[1].lower() != ".json":
        return "Error: input_filepath must reference a '.json' lattice graph"
    if os.path.splitext(output_filepath)[1].lower() not in (".npy", ".tif", ".tiff"):
        return "Error: output_filepath must end with '.npy', '.tif', or '.tiff'"
    for name, val in (("voxel_size_um", voxel_size_um),
                      ("strut_diameter_um", strut_diameter_um),
                      ("cell_edge_mm", cell_edge_mm)):
        if not (np.isfinite(val) and val > 0):
            return f"Error: {name} must be a positive finite number (got {val})"

    try:
        with open(input_filepath) as fh:
            graph = json.load(fh)
        positions = np.array([j["position"] for j in graph["junctions"]], dtype=np.float64)
        struts = np.array(
            [[s["junction0"], s["junction1"]] for s in graph["struts"]], dtype=np.int64
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        return f"Error: could not parse lattice JSON: {error}"

    if positions.ndim != 2 or positions.shape[1] != 3:
        return f"Error: expected junction positions of shape (N, 3), got {positions.shape}"
    if len(struts) == 0:
        return "Error: lattice has no struts to rasterize"

    # (x, y, z) in the file -> (z, y, x) for array indexing.
    pos_zyx = positions[:, [2, 1, 0]]

    # Derive voxels-per-unit from the median strut length so physical metrics align.
    seg = pos_zyx[struts[:, 0]] - pos_zyx[struts[:, 1]]
    median_len = float(np.median(np.linalg.norm(seg, axis=1)))
    if median_len <= 0:
        return "Error: degenerate lattice (median strut length is zero)"
    scale = _octet_strut_length_vox(cell_edge_mm, voxel_size_um) / median_len
    # If the JSON is already in CT voxels (a registered file), its strut length already
    # matches the physical voxel length, so scale comes out ~1. Snap it to exactly 1 so
    # we do NOT rescale a pre-registered lattice to nominal size -- that 0.6% shrink
    # would de-register it from the CT by several voxels at the specimen edges. A
    # unitless nominal file has scale >> 1 and is unaffected.
    if abs(scale - 1.0) < 0.15:
        scale = 1.0
    pos_vox = pos_zyx * scale
    radius_vox = (strut_diameter_um / 2.0) / voxel_size_um

    requested = [shape_z, shape_y, shape_x]
    if any(v < 0 for v in requested):
        return "Error: shape_z, shape_y, shape_x must be >= 0"
    if any(requested) and not all(requested):
        return "Error: to set an explicit shape, all of shape_z, shape_y, shape_x must be > 0"

    if all(requested):
        shape = (int(shape_z), int(shape_y), int(shape_x))
        origin = np.zeros(3)
    else:
        pad = int(np.ceil(radius_vox)) + 2
        origin = np.floor(pos_vox.min(0)).astype(int) - pad
        shape = tuple(int(v) for v in (np.ceil(pos_vox.max(0)).astype(int) - origin + pad))

    try:
        volume = _rasterize_struts(pos_vox, struts, shape, radius_vox, origin.astype(float))
        _save_volume(output_filepath, volume)
    except (OSError, ValueError, MemoryError) as error:
        return f"Error: could not rasterize/save lattice: {error}"

    frac = float(volume.mean())
    return (
        f"Saved rasterized lattice to {output_filepath} "
        f"(junctions={len(positions)}, struts={len(struts)}, shape={shape}, "
        f"scale={scale:.4f} vox/unit, strut_radius={radius_vox:.3f} vox, "
        f"foreground_fraction={frac*100:.3f}%)"
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
def compare_slices(
    design_filepath: str,
    ct_mask_filepath: str,
    output_montage: str,
    axis: int = 0,
    worst_n: int = 9,
    tolerance_vox: int = 2,
    min_design_voxels: int = 200,
    tile_size: int = 256,
    slice_indices: str = "",
) -> str:
    """
    Compares an as-designed rasterized lattice volume against an as-built CT segmentation
    mask, slice by slice, and flags the slices where the CT is most likely missing struts.

    All slices are scored deterministically (no image tokens). The primary metric is
    *design coverage*: the fraction of design-foreground voxels in a slice that have CT
    material within ``tolerance_vox`` (accounts for the CT struts being fatter than the
    ideal cylinders). Low coverage means design struts with no CT material there --
    candidate missing/broken struts. Dice is also reported for reference. Only the
    ``worst_n`` lowest-coverage slices are composited into a single montage PNG (diff
    overlays) for an agent to review, keeping vision cost to one image.

    Overlay colors: green = design & CT agree, red = design only (candidate defect),
    blue = CT only (dross / fat junctions / mis-scale).

    Args:
        design_filepath: Rasterized design volume (.npy/.tif), same shape as the CT mask.
        ct_mask_filepath: Binary CT segmentation mask (.npy/.tif).
        output_montage: Where to save the worst-slice montage (.png).
        axis: Slicing axis (0, 1, or 2). Default 0.
        worst_n: Number of lowest-coverage slices to show in the montage (default 9).
        tolerance_vox: Dilation radius (voxels) applied to the CT before coverage, to
            absorb strut-thickness/sub-voxel differences (default 2).
        min_design_voxels: Slices with fewer design voxels than this are ignored when
            ranking (they carry no lattice, e.g. outside the specimen). Default 200.
        tile_size: Pixel size of each montage tile (downscaled). Default 256.
        slice_indices: Optional comma-separated slice indices to render instead of the
            worst-by-coverage ranking (e.g. "183,222,301"). Use this to review specific
            slices chosen from a 3D analysis rather than by per-slice coverage. Per-slice
            coverage/Dice are still reported.

    Returns:
        A compact summary: mean coverage, and the ranked worst slices with their coverage
        and Dice, plus the montage path. Or an error message.
    """
    if axis not in (0, 1, 2):
        return f"Error: axis must be 0, 1, or 2 (got {axis})"
    if os.path.splitext(output_montage)[1].lower() != ".png":
        return "Error: output_montage must end with '.png'"
    if worst_n < 1 or worst_n > 64:
        return f"Error: worst_n must be in [1, 64] (got {worst_n})"

    try:
        from scipy import ndimage as ndi
    except ImportError:
        ndi = None

    try:
        design = _load_volume(design_filepath)
        ct = _load_volume(ct_mask_filepath)
    except (OSError, ValueError) as error:
        return f"Error: could not load volumes: {error}"
    if design.shape != ct.shape:
        return f"Error: shape mismatch design {design.shape} vs ct {ct.shape}"
    if design.ndim != 3:
        return f"Error: expected 3D volumes, got {design.shape}"

    design = design > 0
    ct = ct > 0
    n = design.shape[axis]

    def _dilate(mask2d):
        if tolerance_vox <= 0:
            return mask2d
        if ndi is not None:
            return ndi.binary_dilation(mask2d, iterations=tolerance_vox)
        # Fallback: iterative 4-neighbour dilation without scipy.
        out = mask2d.copy()
        for _ in range(tolerance_vox):
            out[:-1, :] |= out[1:, :]; out[1:, :] |= out[:-1, :]
            out[:, :-1] |= out[:, 1:]; out[:, 1:] |= out[:, :-1]
        return out

    coverage = np.full(n, np.nan)
    dice = np.full(n, np.nan)
    d_counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        D = np.take(design, i, axis=axis)
        C = np.take(ct, i, axis=axis)
        dsum = int(D.sum())
        d_counts[i] = dsum
        if dsum < min_design_voxels:
            continue
        covered = int((D & _dilate(C)).sum())
        coverage[i] = covered / dsum
        inter = int((D & C).sum())
        dice[i] = 2 * inter / (dsum + int(C.sum())) if (dsum + C.sum()) > 0 else 0.0

    valid = ~np.isnan(coverage)
    if not valid.any():
        return (f"Error: no slices with >= {min_design_voxels} design voxels along axis "
                f"{axis}; nothing to compare")

    if slice_indices.strip():
        try:
            worst = [int(s) for s in slice_indices.split(",") if s.strip()]
        except ValueError:
            return f"Error: slice_indices must be comma-separated integers (got '{slice_indices}')"
        out_of_range = [i for i in worst if not (0 <= i < n)]
        if out_of_range:
            return f"Error: slice_indices out of range [0,{n}): {out_of_range}"
        selection_desc = f"{len(worst)} requested slices"
    else:
        order = np.argsort(np.where(valid, coverage, np.inf))
        worst = [int(i) for i in order[:worst_n]]
        selection_desc = f"worst {len(worst)} by coverage"

    # Montage of the worst slices: one downscaled diff-overlay tile each.
    cols = int(np.ceil(np.sqrt(len(worst))))
    rows = int(np.ceil(len(worst) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    for ax_i, slice_idx in enumerate(worst):
        D = np.take(design, slice_idx, axis=axis)
        C = np.take(ct, slice_idx, axis=axis)
        overlay = np.zeros(D.shape + (3,), dtype=np.uint8)
        overlay[..., 1] = (D & C) * 255            # green: agree
        overlay[..., 0] = (D & ~C) * 255           # red: design only (candidate defect)
        overlay[..., 2] = (~D & C) * 255           # blue: CT only
        axes[ax_i].imshow(overlay, interpolation="nearest")
        axes[ax_i].set_title(f"slice {slice_idx}\ncov={coverage[slice_idx]:.2f} "
                             f"dice={dice[slice_idx]:.2f}", fontsize=9)
        axes[ax_i].set_xticks([]); axes[ax_i].set_yticks([])
    for ax_i in range(len(worst), len(axes)):
        axes[ax_i].axis("off")
    fig.suptitle(f"{selection_desc} (axis {axis}) — "
                 f"green=agree red=missing blue=CT-only", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_montage)), exist_ok=True)
    fig.savefig(output_montage, dpi=max(50, tile_size // 3))
    plt.close(fig)

    mean_cov = float(np.nanmean(coverage))
    ranked = ", ".join(f"z{i}(cov={coverage[i]:.2f},dice={dice[i]:.2f})" for i in worst)
    return (
        f"Compared {int(valid.sum())} lattice-bearing slices along axis {axis}. "
        f"mean_coverage={mean_cov:.3f}. Showing {selection_desc}: {ranked}. "
        f"Montage saved to {output_montage}"
    )


@mcp.tool()
def visualize_lattice_3d(
    input_filepath: str,
    output_filepath: str,
    azimuth: float = 30.0,
    elevation: float = 20.0,
    color: str = "lightgray",
    background: str = "white",
    image_size: int = 900,
    max_dim: int = 256,
) -> str:
    """
    Renders a 3D binary lattice volume to a PNG using PyVista (offscreen), so an agent
    can visually inspect the rasterized structure. The surface is extracted with marching
    cubes and rendered from the requested camera angle.

    Large volumes are block-downsampled (by max-pooling, which preserves thin struts)
    before meshing so a full 700+ voxel lattice does not exhaust memory -- a full-resolution
    surface of such a volume is tens of millions of triangles and can crash the host. The
    downsampled surface is more than enough to judge the structure visually.

    Args:
        input_filepath: Path to a 3D binary volume (.npy or .tif/.tiff), 1 = material.
        output_filepath: Where to save the rendered image (.png).
        azimuth: Camera azimuth in degrees (rotation about the vertical axis).
        elevation: Camera elevation in degrees.
        color: Surface color of the lattice.
        background: Render background color.
        image_size: Output image size in pixels (square). Smaller = cheaper for an agent
            to view; 900 is a good default, drop to ~512 for token-lean review.
        max_dim: Largest volume dimension to mesh; bigger volumes are downsampled to this
            first (default 256). Keeps meshing memory bounded.

    Returns:
        A status message with the surface triangle count and save location, or an error.
    """
    if not os.path.isfile(input_filepath):
        return f"Error: input file not found at {input_filepath}"
    if os.path.splitext(output_filepath)[1].lower() != ".png":
        return "Error: output_filepath must end with '.png'"
    if image_size < 64 or image_size > 4096:
        return f"Error: image_size must be in [64, 4096] (got {image_size})"
    if max_dim < 32 or max_dim > 1024:
        return f"Error: max_dim must be in [32, 1024] (got {max_dim})"

    try:
        import pyvista as pv
        from skimage.measure import block_reduce, marching_cubes
    except ImportError as error:
        return (f"Error: a rendering dependency is missing ({error}). "
                "visualize_lattice_3d needs pyvista and scikit-image.")

    try:
        volume = _load_volume(input_filepath)
    except (OSError, ValueError) as error:
        return f"Error: could not load input volume: {error}"
    if volume.ndim != 3:
        return f"Error: expected a 3D volume, got shape {volume.shape}"

    mask = volume > 0
    if not mask.any():
        return "Error: volume has no foreground voxels to render"

    # Downsample big volumes (max-pool keeps thin struts connected) to bound memory.
    factor = max(1, int(np.ceil(max(mask.shape) / max_dim)))
    if factor > 1:
        mask = block_reduce(mask, (factor, factor, factor), np.max)

    try:
        # Marching cubes straight to a triangle mesh -- no giant intermediate grid.
        padded = np.pad(mask.astype(np.float32), 1)  # pad so border surfaces close
        verts, faces, _, _ = marching_cubes(padded, level=0.5)
        if len(faces) == 0:
            return "Error: no surface extracted from the volume"
        faces_pv = np.hstack(
            [np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)]
        ).ravel()
        surface = pv.PolyData(verts, faces_pv)

        pv.global_theme.allow_empty_mesh = True
        plotter = pv.Plotter(off_screen=True, window_size=[image_size, image_size])
        plotter.background_color = background
        plotter.add_mesh(surface, color=color, smooth_shading=True)
        plotter.view_isometric()
        plotter.camera.azimuth = azimuth
        plotter.camera.elevation = elevation
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        plotter.screenshot(output_filepath)
        plotter.close()
    except Exception as error:  # noqa: BLE001 - VTK raises a wide range of errors
        return f"Error: rendering failed: {error}"

    return (
        f"Saved 3D lattice render to {output_filepath} "
        f"(surface_triangles={surface.n_cells}, azimuth={azimuth}, elevation={elevation}, "
        f"size={image_size}px)"
    )


@mcp.tool()
def export_lattice_html(
    input_filepath: str,
    output_filepath: str,
    color: str = "lightgray",
    background: str = "white",
    max_dim: int = 160,
) -> str:
    """
    Exports a 3D binary lattice volume as a self-contained interactive HTML file that opens
    in any web browser -- the user can rotate, pan, and zoom the lattice with the mouse.

    This is the way to get a LIVE, interactive 3D view in a headless environment (no X
    server): the scene renders in the browser via WebGL, so it needs no display. (A native
    PyVista window via plotter.show() would require a working X server and is not usable
    here.) The volume is block-downsampled before meshing to keep the file size and memory
    reasonable; the HTML embeds the geometry, so bigger max_dim -> larger file.

    Args:
        input_filepath: Path to a 3D binary volume (.npy or .tif/.tiff), 1 = material.
        output_filepath: Where to save the interactive scene (.html).
        color: Surface color of the lattice.
        background: Scene background color.
        max_dim: Largest volume dimension to mesh; bigger volumes are downsampled to this
            first (default 160). Raise for more detail at the cost of HTML file size.

    Returns:
        A status message with the triangle count, file size, and save location, or an error.
    """
    if not os.path.isfile(input_filepath):
        return f"Error: input file not found at {input_filepath}"
    if os.path.splitext(output_filepath)[1].lower() != ".html":
        return "Error: output_filepath must end with '.html'"
    if max_dim < 32 or max_dim > 512:
        return f"Error: max_dim must be in [32, 512] (got {max_dim})"

    try:
        import pyvista as pv
    except ImportError as error:
        return (f"Error: a rendering dependency is missing ({error}). export_lattice_html "
                "needs pyvista, scikit-image, and trame (pip install trame trame-vtk "
                "trame-vuetify nest_asyncio2).")

    try:
        volume = _load_volume(input_filepath)
    except (OSError, ValueError) as error:
        return f"Error: could not load input volume: {error}"
    if volume.ndim != 3:
        return f"Error: expected a 3D volume, got shape {volume.shape}"

    try:
        surface, factor = _volume_to_pv_surface(volume, max_dim)
    except ImportError as error:
        return (f"Error: a rendering dependency is missing ({error}). export_lattice_html "
                "needs pyvista, scikit-image, and trame.")
    except ValueError as error:
        return f"Error: {error}"

    try:
        pv.global_theme.allow_empty_mesh = True
        plotter = pv.Plotter(off_screen=True)
        plotter.background_color = background
        plotter.add_mesh(surface, color=color, smooth_shading=True)
        plotter.view_isometric()
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        plotter.export_html(output_filepath)
        plotter.close()
    except Exception as error:  # noqa: BLE001 - trame/VTK raise a wide range of errors
        return (f"Error: HTML export failed: {error}. If it mentions a trame server, "
                "ensure trame and nest_asyncio2 are installed in this environment.")

    size_mb = os.path.getsize(output_filepath) / 1e6
    return (
        f"Saved interactive lattice HTML to {output_filepath} "
        f"(surface_triangles={surface.n_cells}, downsample_factor={factor}, "
        f"file_size={size_mb:.1f} MB). Open it in a web browser to rotate/zoom."
    )


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


@mcp.tool()
def measure_lattice_iou(
    mask_filepath: str,
    design_filepath: str,
    output_directory: str,
    correction_filepath: str = "",
    strut_diameter_um: float = 350.0,
    cell_mm: float = 4.56,
    trim_fraction: float = 0.20,
    envelope_factor: float = 1.6,
    stations: int = 12,
    embedded_fraction: float = 0.80,
) -> str:
    """
    Compares every strut and node of a registered lattice design against a CT
    segmentation mask, by intersection-over-union with the nominal geometry.

    For each strut the design gives a line segment between two junctions. The segment
    is trimmed at both ends to exclude the (much fatter) junctions, a nominal cylinder
    of `strut_diameter_um` is painted around what remains, and that cylinder is compared
    against the mask inside a local crop:

        IoU = |cylinder & mask| / (|cylinder| + |mask & envelope| - |cylinder & mask|)

    The cylinder is also divided into `stations` bands along its axis, giving a fill
    profile as you traverse the edge -- a wholly missing strut reads low everywhere, a
    severed one reads healthy at the ends with a dip between, and one scalar cannot tell
    those apart. Nodes are measured against a nominal sphere and sized directly, both by
    max inscribed radius and by the radius at which the ball is still 90% material.

    Voxel size is derived from the registered strut length against the known unit cell
    (an octet strut spans cell/sqrt(2)), so no header metadata is trusted.

    Writes struts.csv, nodes.csv, profiles.npz, summary.json and two distribution plots.
    No threshold is applied and none is suggested: summary.json carries each metric's
    percentiles and full binned histogram, and the plots show the shapes, so the cut is
    chosen by looking at the distribution.

    Args:
        mask_filepath: Path to the binary segmentation .tif/.npy (z, y, x), e.g. the Otsu mask.
        design_filepath: Path to the registered lattice .json (junctions + struts).
        output_directory: Directory to write the CSVs, arrays, summary and plots into.
        correction_filepath: Optional registration correction.json from
            scripts/refit_registration.py. Strongly recommended: the shipped registration
            carries a ~1% residual scale error, which at a ~2 voxel as-built strut radius
            walks the design off the material toward the specimen corners and shows up as
            a fake defect rate.
        strut_diameter_um: Nominal design strut diameter. 350 for these specimens.
        cell_mm: Nominal unit cell edge length in mm. 4.56 for these specimens.
        trim_fraction: Fraction of the strut length excluded at each end as junction.
        envelope_factor: Union envelope radius as a multiple of the nominal radius.
        stations: Number of bands along each strut for the traversal profile.
        embedded_fraction: Local material fraction above which a strut or node is treated
            as buried in bulk metal (these specimens are fused into solid build plates at
            both z ends) and held out of the statistics, since a missing strut inside
            solid metal leaves no signature to detect.

    Returns:
        A summary of the measured distributions and recommended cuts, or an error message.
    """
    if not os.path.isfile(mask_filepath):
        return f"Error: mask file not found at {mask_filepath}"
    if not os.path.isfile(design_filepath):
        return f"Error: design JSON not found at {design_filepath}"
    if correction_filepath and not os.path.isfile(correction_filepath):
        return f"Error: correction file not found at {correction_filepath}"
    if not 0.0 <= trim_fraction < 0.5:
        return f"Error: trim_fraction must be in [0, 0.5) (got {trim_fraction})"
    if envelope_factor <= 1.0:
        return f"Error: envelope_factor must exceed 1.0 (got {envelope_factor})"
    if stations < 2:
        return f"Error: stations must be at least 2 (got {stations})"

    try:
        summary = lattice_iou.run(
            mask_filepath, design_filepath, output_directory,
            correction_path=correction_filepath or None,
            trim_frac=trim_fraction, env_factor=envelope_factor, stations=stations,
            strut_diameter_um=strut_diameter_um, cell_mm=cell_mm,
            embedded_frac=embedded_fraction, log=lambda *_: None,
        )
    except (OSError, ValueError, KeyError) as error:
        return f"Error: could not measure the lattice: {error}"

    geometry = summary["geometry"]
    counts = summary["counts"]
    built = summary["as_built"]
    lines = [
        f"Measured {counts['struts']} struts and {counts['physical_nodes']} nodes "
        f"against {design_filepath}.",
        f"  registration correction applied: {summary['correction_applied']}",
        f"  scale: {geometry['um_per_voxel']:.3f} um/voxel, nominal strut radius "
        f"{geometry['nominal_strut_radius_vox']:.3f} voxels",
        f"  held out: {counts['boundary_struts']} boundary-cap struts, "
        f"{counts['embedded_struts']} struts embedded in bulk metal "
        f"-> {counts['measurable_struts']} measurable",
        f"  as-built strut diameter (median) {built['median_strut_diameter_um']:.0f} um "
        f"vs nominal {strut_diameter_um:.0f} um",
        f"  as-built node diameter (median) {built['median_node_diameter_um']:.0f} um "
        f"at 90% fill, {built['median_node_inscribed_diameter_um']:.0f} um inscribed",
        "",
        "Distributions (no threshold applied -- read the cut off these):",
    ]
    for key, report in summary["distributions"].items():
        if not report.get("n"):
            continue
        pct = report["percentiles"]
        lines.append(
            f"  {key}: n={report['n']}, median {report['median']:.4f}, "
            f"p1 {pct['1']:.4f}, p5 {pct['5']:.4f}, p95 {pct['95']:.4f}, "
            f"range [{report['min']:.4f}, {report['max']:.4f}]"
        )
    lines.append("")
    lines.append(f"Wrote struts.csv, nodes.csv, profiles.npz, summary.json and "
                 f"two distribution plots to {output_directory}")
    return "\n".join(lines)


@mcp.tool()
def refit_lattice_registration(
    mask_filepath: str,
    design_filepath: str,
    output_directory: str,
    block: int = 170,
    grid: int = 4,
) -> str:
    """
    Refits the registration between a lattice design and a CT mask, and writes the
    correction every downstream measurement needs.

    RUN THIS FIRST. The shipped registrations for these specimens carry a ~1% residual
    scale error. At a ~3 voxel nominal strut radius that is enough to walk the design off
    the material toward the specimen corners, and it does not degrade the answer
    gracefully -- it destroys it. Same detector, shipped vs corrected coordinates: median
    strut IoU 0.256 vs 0.607, and the empty gap that separates missing struts from healthy
    ones has width 0.000 on the shipped coordinates, i.e. the defect population is
    completely swamped and no cut exists. Every defect count from an uncorrected run is
    meaningless.

    Method: the volume is divided into a `grid` x `grid` x `grid` arrangement of blocks of
    side `block` voxels. In each block holding at least 40 struts, the local translation
    that best aligns the design to the mask is found by a coarse-to-fine offset search.
    Those per-block offsets are then fitted by a single affine field, `corrected = p + A p
    + t`, which captures the scale error as the diagonal of A.

    Check the residual in the output before trusting the result. It is reported per axis
    in voxels RMS; values comparable to the strut radius mean the fit did not converge and
    the design and mask may not correspond.

    Args:
        mask_filepath: Binary segmentation .tif/.npy in (z, y, x).
        design_filepath: Registered lattice .json (junctions + struts).
        output_directory: Written to as correction.json and design_corrected.json.
        block: Block side in voxels for the local offset search.
        grid: Blocks per axis.

    Returns:
        The fitted scale correction, per-axis residuals and junction displacement, or an
        error message.
    """
    if not os.path.isfile(mask_filepath):
        return f"Error: mask file not found at {mask_filepath}"
    if not os.path.isfile(design_filepath):
        return f"Error: design JSON not found at {design_filepath}"
    if block < 20:
        return f"Error: block must be at least 20 voxels (got {block})"
    if grid < 2:
        return f"Error: grid must be at least 2 (got {grid})"

    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.refit_registration import refit, load_design, apply_correction

        A, t, resid, C, _ = refit(mask_filepath, design_filepath, block, grid,
                                  verbose=False)
        J, pos, _ = load_design(design_filepath)
        pc = apply_correction(pos, A, t)
        shift = np.linalg.norm(pc - pos, axis=1)

        os.makedirs(output_directory, exist_ok=True)
        scale = 1 + np.diag(A)
        with open(os.path.join(output_directory, "correction.json"), "w") as fh:
            json.dump({"A_zyx": A.tolist(), "t_zyx": t.tolist(),
                       "residual_rms_zyx": resid.tolist(),
                       "scale_correction_zyx": scale.tolist(),
                       "note": "corrected_zyx = p + A @ p + t, p in (z,y,x) voxel coords"},
                      fh, indent=2)
        for j, q in zip(J["junctions"], pc):
            j["position"] = [float(v) for v in q[::-1]]      # back to (x, y, z)
        with open(os.path.join(output_directory, "design_corrected.json"), "w") as fh:
            json.dump(J, fh)
    except (OSError, ValueError, KeyError, ImportError) as error:
        return f"Error: could not refit the registration: {error}"

    lines = [
        f"Refitted registration on {len(C)} blocks of {block} voxels.",
        f"  scale correction (z, y, x) = "
        f"{', '.join(f'{100 * v:+.2f}%' for v in np.diag(A))}",
        f"  residual (voxels RMS)      = "
        f"{', '.join(f'{v:.2f}' for v in resid)}",
        f"  junction displacement: median {np.median(shift):.2f} vox, "
        f"max {shift.max():.2f} vox",
        "",
    ]
    if max(resid) > 3.0:
        lines.append("WARNING: residual is comparable to the strut radius. The fit may "
                     "not have converged; check that the design and mask correspond "
                     "before using this correction.")
    else:
        lines.append("Residual is well under the strut radius; the fit converged.")
    lines.append(f"Wrote correction.json and design_corrected.json to {output_directory}. "
                 f"Pass the correction.json to detect_lattice_defects and "
                 f"detect_missing_nodes.")
    return "\n".join(lines)


@mcp.tool()
def detect_lattice_defects(
    mask_filepath: str,
    design_filepath: str,
    output_directory: str,
    correction_filepath: str = "",
    strut_diameter_um: float = 350.0,
    cell_mm: float = 4.56,
    sections: int = 25,
    tolerance_fraction: float = 0.25,
    use_cache: bool = True,
) -> str:
    """
    Classifies every strut of a lattice into defect classes, from a CT mask and the
    registered design. This is the end-to-end strut detector.

    Runs the full chain: nominal-cylinder fill, node-to-node connectivity, perpendicular
    cross-sections, then the class rules. EXPENSIVE -- about 10 minutes on an 18,000-strut
    lattice from cold. Intermediate arrays are cached in `output_directory`, and
    `use_cache` reuses them, which cuts a re-run with different tolerance bands to seconds.

    Classes, and what decides each:

      missing  no voxel of the nominal cylinder is material AND every cross-section is
               empty. A count == 0, so there is NO threshold. Justified by the data: the
               struts at exactly zero are separated from the next value by a wide empty
               gap, so any cut inside it gives the same answer.
      broken   no geodesic path through material from one node to the other, inside a tube
               of 2x the nominal radius about the design axis. A boolean, so again NO
               threshold. This is topological and cannot be recovered from cross-sections:
               most severed struts have no empty section at all, because a crack narrower
               than the section spacing reads full on every plane.
      thin     median section radius below the tolerance band
      thick    median section radius above it
      necked   normal median radius but a local pinch (min section radius under the 1st
               percentile) -- a partial break rather than a thin strut
      nominal  everything else

    Rules are applied severity-first, later overriding earlier: necked, thick, thin,
    broken, missing.

    READ THE THIN/THICK COUNTS WITH CARE. The band is anchored on the DESIGN diameter, not
    on percentiles of this specimen, because a percentile cut is self-fulfilling -- it
    returns a fixed fraction of thin struts however the part came out, and would call a
    uniformly undersized lattice healthy. The consequence is that systematic process bias
    lands in the class counts. In particular, laser powder-bed struts are thinner in the
    unsupported horizontal orientation than at 45 degrees, so a single band across a mixed
    lattice can classify by build orientation rather than by health. The returned report
    breaks the two families out so this is visible; `missing` and `broken` are immune
    because they are count-based.

    Struts that cannot be measured are excluded and reported separately, never mixed into
    the tallies: outer boundary caps, struts buried in bulk metal (these specimens are
    fused into solid build plates at both z ends, where an absent strut leaves no
    signature), and struts whose cross-section runs into the edge of its window.

    Args:
        mask_filepath: Binary segmentation .tif/.npy in (z, y, x).
        design_filepath: Registered lattice .json.
        correction_filepath: Registration correction.json from refit_lattice_registration.
            Strongly recommended -- without it the defect population is swamped by
            registration error and the counts are meaningless.
        output_directory: Written to as strut_classes.csv plus cached arrays.
        sections: Cross-sections per strut across the trimmed span.
        tolerance_fraction: Thin/thick band as a fraction of the nominal design diameter.
        use_cache: Reuse sections.npz and connectivity.npz if present. A cache measured at
            a different section count is honoured at its own count, since a mismatch
            silently breaks both `missing` and the break rule.

    Returns:
        The class table with counts and median diameters, the exclusions, and the
        orientation breakdown, or an error message.
    """
    if not os.path.isfile(mask_filepath):
        return f"Error: mask file not found at {mask_filepath}"
    if not os.path.isfile(design_filepath):
        return f"Error: design JSON not found at {design_filepath}"
    if correction_filepath and not os.path.isfile(correction_filepath):
        return f"Error: correction file not found at {correction_filepath}"
    if sections < 4:
        return f"Error: sections must be at least 4 (got {sections})"
    if not 0.0 < tolerance_fraction < 1.0:
        return (f"Error: tolerance_fraction must be in (0, 1) "
                f"(got {tolerance_fraction})")

    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.classify_strut_defects import classify, measurable
        from strut_sections import measure_sections

        os.makedirs(output_directory, exist_ok=True)
        summary = lattice_iou.run(
            mask_filepath, design_filepath, output_directory,
            correction_path=correction_filepath or None,
            strut_diameter_um=strut_diameter_um, cell_mm=cell_mm,
            make_plots=False, log=lambda *_: None,
        )
        um = summary["geometry"]["um_per_voxel"]
        r_strut = summary["geometry"]["nominal_strut_radius_vox"]
        nominal_um = summary["geometry"]["nominal_strut_diameter_um"]

        # The voxel size is DERIVED from |strut| = cell/sqrt(2), which is an octet
        # identity. On a lattice with more than one strut length the median is not a
        # length of anything, and every um figure below -- including the thin/thick band
        # -- is then silently wrong rather than absent. The spread is the cheap tell.
        spread_frac = (summary["geometry"]["strut_length_spread_vox"]
                       / max(summary["geometry"]["strut_length_vox"], 1e-9))
        if spread_frac > 0.05:
            return (
                f"Error: strut lengths span {100 * spread_frac:.1f}% of the median "
                f"({summary['geometry']['strut_length_vox']:.1f} vox). This detector "
                f"derives the voxel size from the octet identity |strut| = cell/sqrt(2), "
                f"which needs one strut length; with several, every micron figure would "
                f"be silently wrong. Supply the voxel size directly, or use a geometry "
                f"model for this lattice, before trusting any count.")

        pos, pairs, _, corrected = lattice_iou.load_design(
            design_filepath, correction_filepath or None)
        mask = _load_volume(mask_filepath) > 0

        sec_cache = os.path.join(output_directory, "sections.npz")
        if use_cache and os.path.isfile(sec_cache):
            z = np.load(sec_cache, allow_pickle=False)
            sec = {k[4:]: z[k] for k in z.files if k.startswith("sec_")}
            sections = int(z["prof_r_eq"].shape[1])
        else:
            sec, prof = measure_sections(mask, pos, pairs, r_strut,
                                         n_sections=sections, log=lambda *_: None)
            np.savez_compressed(sec_cache,
                                **{f"sec_{k}": v for k, v in sec.items()},
                                **{f"prof_{k}": v for k, v in prof.items()})

        conn_cache = os.path.join(output_directory, "connectivity.npz")
        if use_cache and os.path.isfile(conn_cache):
            zc = np.load(conn_cache, allow_pickle=False)
            conn = {k: zc[k] for k in zc.files}
        else:
            conn = lattice_iou.measure_connectivity(mask, pos, pairs, r_strut,
                                                    log=lambda *_: None)
            np.savez_compressed(conn_cache, **conn)

        d = np.genfromtxt(os.path.join(output_directory, "struts.csv"),
                          delimiter=",", names=True)
        n = len(sec["r_eq_med"])
        d = {k: d[k][:n] for k in d.dtype.names}
        ok = measurable(d, sec)

        thin_um = (1 - tolerance_fraction) * nominal_um
        thick_um = (1 + tolerance_fraction) * nominal_um
        cuts = {"thin": thin_um / 2.0 / um, "thick": thick_um / 2.0 / um,
                "neck": float(np.percentile(sec["r_eq_min"][ok], 1))}
        lab = classify(sec, d["n_matched"], cuts, sections, 0, conn)

        with open(os.path.join(output_directory, "strut_classes.csv"), "w") as fh:
            fh.write("strut_id,label,r_eq_med_um,measurable\n")
            for i in range(n):
                fh.write(f"{i},{lab[i]},{sec['r_eq_med'][i] * 2 * um:.1f},"
                         f"{int(ok[i])}\n")
    except (OSError, ValueError, KeyError, ImportError) as error:
        return f"Error: could not classify the lattice: {error}"

    order = ["missing", "broken", "thin", "thick", "necked", "nominal"]
    lines = [
        f"Classified {int(ok.sum())} measurable struts of {n}.",
        f"  registration correction applied: {corrected}"
        + ("" if corrected else "   <-- NOT APPLIED, counts are unreliable"),
        f"  scale {um:.3f} um/voxel, nominal strut radius {r_strut:.3f} vox",
        f"  excluded {n - int(ok.sum())}: boundary caps, plate-embedded and "
        f"window-touching struts",
        f"  thin/thick band: +/-{100 * tolerance_fraction:.0f}% of the nominal "
        f"{nominal_um:.0f} um  ->  {thin_um:.0f} / {thick_um:.0f} um",
        f"  as-built median diameter "
        f"{np.median(sec['r_eq_med'][ok]) * 2 * um:.0f} um "
        f"({100 * np.median(sec['r_eq_med'][ok]) * 2 * um / nominal_um:.0f}% of nominal)",
        "",
        f"  {'class':<9s} {'count':>6s} {'pct':>8s}   median diameter",
    ]
    for name in order:
        m = ok & (lab == name)
        c = int(m.sum())
        dia = (f"{np.median(sec['r_eq_med'][m]) * 2 * um:.0f} um"
               if c and name != "missing" else "--")
        lines.append(f"  {name:<9s} {c:6d} {100 * c / max(int(ok.sum()), 1):7.3f}%   {dia}")

    # Build orientation, because a single thin/thick band can classify by it rather than
    # by health and the reader has no way to see that from the counts alone.
    v = pos[pairs[:n, 1]] - pos[pairs[:n, 0]]
    ang = np.degrees(np.arccos(np.abs(v[:, 0]) / np.linalg.norm(v, axis=1)))
    horiz = ang > 70
    lines += ["", "  build orientation (watch this before reading thin/thick):",
              f"  {'family':<12s} {'n':>6s} {'median dia':>11s} {'thin':>6s} {'thick':>6s}"]
    for name, m in (("inclined", ok & ~horiz), ("horizontal", ok & horiz)):
        if not m.any():
            continue
        lines.append(
            f"  {name:<12s} {int(m.sum()):6d} "
            f"{np.median(sec['r_eq_med'][m]) * 2 * um:8.0f} um "
            f"{int((lab[m] == 'thin').sum()):6d} {int((lab[m] == 'thick').sum()):6d}")
    lines.append("")
    lines.append(f"Wrote strut_classes.csv to {output_directory}. "
                 f"missing and broken carry no threshold; thin/thick depend on "
                 f"tolerance_fraction.")
    return "\n".join(lines)


@mcp.tool()
def detect_missing_nodes(
    mask_filepath: str,
    design_filepath: str,
    correction_filepath: str = "",
    results_directory: str = "",
    strut_diameter_um: float = 350.0,
    cell_mm: float = 4.56,
    radius_factor: float = 1.467,
) -> str:
    """
    Finds junctions that were never printed, by sphere fill, and cross-checks each against
    the struts that should meet there.

    The test is simply how much of a sphere about the design junction is material:

        fill = |{sphere of radius_factor x r_strut} & mask| / |sphere|

    EVERY node in the design JSON is tested, with no filtering -- go to the point the
    design declares, look at the voxels there, report what is missing. Counts restricted
    to interior (degree 12) junctions are reported underneath as a second number, not as
    the headline, because filtering by degree answers a narrower question than "which
    junctions in this design have nothing behind them" and can hide the larger count.

    A junction that printed reads ~1.0 and one that did not reads 0.0, with nothing in
    between, so the cut is READ OFF THE EMPTY GAP in the sorted distribution rather than
    chosen. The tool reports the gap width; a wide gap means the answer does not depend on
    where in it the cut falls, and the radius does not matter either -- anywhere from the
    design node radius to ~3x the strut radius gives the same nodes.

    READ THE GEOMETRY OF THE HITS BEFORE QUOTING A RATE. If the flagged nodes all lie on
    one flat plane, that is the design reaching past the printed part on that face, and it
    is reported as such -- on the 0.5% specimen 181 of 183 hits are the single face
    y = 759, so the whole-part rate is 5.34% or 0.08% depending entirely on how that face
    is read. The tool states the geometry and does not decide for you.

    The reading is that clean because a node is present whole or absent whole. It is not a
    separately printed part: in the design a junction is a bare position with no radius and
    no thickness, and only struts carry thickness. The junction is where the struts overlap
    plus the fillet the melt pool leaves where their scan vectors converge. So do not look
    for a node absent while its struts are present -- the struts alone would fill the
    sphere, and there is nothing at a junction that could go missing on its own. A hit here
    is a localised build failure that took a whole junction with it.

    Every candidate is cross-checked against a second, independent instrument: the number
    of struts that actually arrive, which comes from the strut labels rather than from
    these voxels. Agreement between the two is the reason to believe a result; a
    disagreement is flagged and should be inspected before it is reported.

    Fast -- seconds, not minutes. Requires detect_lattice_defects to have been run into the
    same directory first, since the cross-check reads its connectivity output.

    Args:
        mask_filepath: Binary segmentation .tif/.npy in (z, y, x).
        design_filepath: Registered lattice .json.
        correction_filepath: Registration correction.json. Strongly recommended.
        results_directory: The output_directory a previous detect_lattice_defects run
            wrote to. Supplying it enables the independent degree cross-check; without it
            the sphere fill is reported on its own and is one instrument, not two.
        radius_factor: Sphere radius in nominal strut radii. The default is the design
            node radius; the answer is unchanged over roughly 1.5 to 3.0.

    Returns:
        The flagged nodes with their coordinates, the gap the cut was read from, and the
        agreement with the degree cross-check, or an error message.
    """
    if not os.path.isfile(mask_filepath):
        return f"Error: mask file not found at {mask_filepath}"
    if not os.path.isfile(design_filepath):
        return f"Error: design JSON not found at {design_filepath}"
    if correction_filepath and not os.path.isfile(correction_filepath):
        return f"Error: correction file not found at {correction_filepath}"
    if radius_factor <= 0:
        return f"Error: radius_factor must be positive (got {radius_factor})"

    try:
        pos, pairs, _, corrected = lattice_iou.load_design(
            design_filepath, correction_filepath or None)
        node_pos, node_of, degree = lattice_iou.dedupe_junctions(pos, pairs)
        geom = lattice_iou.lattice_geometry(pos, pairs, cell_mm=cell_mm,
                                            strut_diameter_um=strut_diameter_um)
        r_node = radius_factor * geom["nominal_strut_radius_vox"]
        mask = _load_volume(mask_filepath) > 0
        fill = lattice_iou.measure_node_sphere_fill(mask, node_pos, r_node,
                                                    log=lambda *_: None)
    except (OSError, ValueError, KeyError) as error:
        return f"Error: could not measure the nodes: {error}"

    # THE ANSWER IS THE UNFILTERED ONE: every node the JSON declares, tested where the
    # design puts it. Filtering by degree first answers a narrower question than was asked
    # and buries the biggest number in a footnote, which is how this tool used to read.
    interior = degree == 12
    sv = np.sort(fill)
    g = int(np.argmax(np.diff(sv)))
    cut = 0.5 * (sv[g] + sv[g + 1])
    gap = float(sv[g + 1] - sv[g])
    flagged = fill <= cut
    n_empty = int((fill < 0.01).sum())

    lines = [
        f"{len(node_pos)} physical nodes from {len(pos)} junction entries "
        f"({int(interior.sum())} interior, {int((~interior).sum())} surface).",
        f"  registration correction applied: {corrected}"
        + ("" if corrected else "   <-- NOT APPLIED, result is unreliable"),
        f"  sphere radius {radius_factor:.3f} r = {r_node:.2f} vox = "
        f"{r_node * geom['um_per_voxel']:.0f} um",
        f"  fill over ALL nodes: min {sv[0]:.3f}, p1 {np.percentile(sv, 1):.3f}, "
        f"median {np.median(sv):.3f}",
        f"  largest gap {sv[g]:.3f} -> {sv[g + 1]:.3f} ({gap:.3f} wide); "
        f"cut read off it at {cut:.3f}",
        "",
        f"MISSING NODES: {int(flagged.sum())} of {len(node_pos)} "
        f"({100 * flagged.sum() / len(node_pos):.2f}%)"
        + (f"   ({n_empty} of them exactly empty)" if n_empty != int(flagged.sum()) else ""),
    ]
    if gap < 0.2:
        lines.append("  NOTE: the gap is only "
                     f"{gap:.3f} wide, so this cut IS a choice. {n_empty} nodes are "
                     "exactly empty; the rest of the count depends on where the cut falls.")

    # Where they sit decides how the number should be read, so always say. A flat face of
    # empties is the design reaching past the printed part; scattered ones are build
    # failures. The tool reports the geometry and does not decide for the reader.
    emp = flagged
    if emp.sum() > 1:
        pts = node_pos[emp]
        names = ("z", "y", "x")
        tol = 0.5 * geom["strut_length_vox"]
        # COUNT how many sit on a plane rather than measuring the total spread: a single
        # stray hit elsewhere sends the spread to the full part width and hides a face
        # that is really there.
        best = max(((int((np.abs(pts[:, a] - np.median(pts[:, a])) < tol).sum()), a)
                    for a in range(3)), key=lambda t: t[0])
        n_face, axis = best
        if n_face > 0.5 * emp.sum():
            rest = int(emp.sum()) - n_face
            lines.append(
                f"  {n_face} of the {int(emp.sum())} lie on the single plane "
                f"{names[axis]} = {np.median(pts[:, axis]):.0f}"
                + (f"; the other {rest} are scattered." if rest else " -- all of them."))
            lines.append(
                "  A whole flat face reads as the design reaching past the printed part "
                "rather than as build failures, and the scattered ones read as build "
                "failures. Decide which before quoting a rate -- here the two readings "
                f"differ by {emp.sum() / max(rest, 1):.0f}x.")
    lines.append(f"  of the flagged, {int((flagged & interior).sum())} are interior "
                 f"(degree 12) and {int((flagged & ~interior).sum())} are surface.")

    # Surface nodes still deserve their own note: they read empty for two very different
    # reasons -- not printed, or the design extends past the part.
    surface = ~interior
    if surface.any() and interior.any():
        lines += [
            "",
            "The same count restricted to interior (degree 12) nodes only: "
            f"{int((flagged & interior).sum())} of {int(interior.sum())} "
            f"({100 * (flagged & interior).sum() / interior.sum():.2f}%).",
            "  Both are real answers to different questions. The headline counts every "
            "junction the design declares. This one counts only junctions fully "
            "surrounded by lattice, where an empty sphere cannot be a surface effect.",
            "  Cross-check with detect_missing_nodes_2d, which is design-free -- it "
            "proposes no site outside the printed part, so it will not see a face "
            "mismatch at all.",
        ]

    # The independent instrument: how many of the struts that should meet here actually
    # arrive. It comes from the strut labels, the fill above comes from the voxels.
    dead = None
    needed = ("connectivity.npz", "strut_classes.csv", "struts.csv")
    if results_directory and all(
            os.path.isfile(os.path.join(results_directory, f)) for f in needed):
        try:
            conn = dict(np.load(os.path.join(results_directory, "connectivity.npz"),
                                allow_pickle=False))
            cls = np.genfromtxt(os.path.join(results_directory, "strut_classes.csv"),
                                delimiter=",", names=True, dtype=None, encoding=None)
            st = np.genfromtxt(os.path.join(results_directory, "struts.csv"),
                               delimiter=",", names=True)
            m = len(cls["label"])
            ends = node_of[pairs[:m]]
            usable = ((st["is_boundary"][:m] == 0) & (st["embedded"][:m] == 0)
                      & ~conn["clipped"][:m] & ~conn["no_seed"][:m])
            bad = usable & ((cls["label"] == "missing") | (cls["label"] == "broken"))
            inc_u = np.bincount(ends[usable].ravel(), minlength=len(node_pos))
            inc_b = np.bincount(ends[bad].ravel(), minlength=len(node_pos))
            dead = interior & (inc_u >= 8) & ((inc_u - inc_b) == 0)
        except (OSError, ValueError, KeyError):
            dead = None

    listed = np.flatnonzero(flagged | (dead if dead is not None else flagged))
    lines.append("")
    # Interior first: those are the ones a reader acts on. A face mismatch produces
    # hundreds of surface entries and would bury them.
    listed = listed[np.argsort(~interior[listed], kind="stable")]
    cap = 25
    for k in listed[:cap]:
        p = node_pos[k]
        note = ""
        if dead is not None and interior[k]:
            note = ("  confirmed by both" if flagged[k] and dead[k]
                    else "  ONE INSTRUMENT ONLY -- inspect before reporting")
        lines.append(f"  node {k}  z={p[0]:.0f} y={p[1]:.0f} x={p[2]:.0f}  "
                     f"deg {int(degree[k]):2d}  fill {fill[k]:.3f}{note}")
    if len(listed) > cap:
        lines.append(f"  ... and {len(listed) - cap} more (all listed in nodes.csv, "
                     f"column `fill`)")

    lines.append("")
    if dead is None:
        lines.append("Degree cross-check NOT run: pass results_directory from a "
                     "detect_lattice_defects run to confirm each candidate against the "
                     "struts that should arrive. On its own this is one instrument.")
    else:
        # Scoped to interior nodes: `dead` needs >=8 measurable struts arriving, which a
        # surface node never has, so counting surface nodes as "disagreements" would
        # report 182 conflicts that are really just the cross-check not applying there.
        agree = int((flagged & dead & interior).sum())
        disagree = int(((flagged ^ dead) & interior).sum())
        lines.append(f"Degree cross-check (interior nodes only -- a surface node has too "
                     f"few measurable struts for it to mean anything):")
        lines.append(f"  {int(dead.sum())} nodes with no strut arriving.  "
                     f"agree with sphere fill: {agree}   disagree: {disagree}")
        if disagree:
            lines.append("  A disagreement means the voxels and the strut labels tell "
                         "different stories. Inspect those nodes before reporting them.")
        n_surf_flag = int((flagged & ~interior).sum())
        if n_surf_flag:
            lines.append(f"  The {n_surf_flag} flagged surface nodes are NOT cross-checked "
                         f"by this instrument -- they rest on the sphere fill alone.")

    lines += _malformed_node_section(results_directory,
                                     inc_bad=(inc_b if dead is not None else None))
    return "\n".join(lines)


def _malformed_node_section(results_directory: str, show: int = 12, inc_bad=None) -> list:
    """Rank junctions that printed but did not FORM, by how far out they stay solid.

    Absence and malformation are different failures and the sphere fill above can only see
    the first: its 257 um probe sits inside a 937 um junction, so anything short of total
    absence reads 1.000. The size already measured by `measure_nodes` does not saturate --
    `diameter_fill_um` is the largest ball about the junction that is still 90% material --
    and its low tail is where a junction whose struts arrived but never fused shows up.

    Ranked twice on purpose. Junction size falls with build height on this specimen
    (r = -0.47), so the raw low tail is partly a list of tall nodes; the detrended ratio
    says how small a junction is against others at ITS height. Both are printed because
    the trend is a real property of the print and hiding it would be worse than the bias.
    No threshold is applied -- the gaps are printed and the cut is read off them.

    TWO confounds, not one, and the second is the one that invalidates a naive reading.
    Size also falls with the number of defective struts arriving (r = -0.42 here, against
    -0.53 for height) -- a junction with struts missing has less material near it whether
    or not the junction itself failed. So a small junction is only evidence of a *junction*
    failure if it is small against nodes with the SAME number of dead struts. That
    comparison is printed; a candidate that is merely typical for its strut damage is a
    consequence of the strut defects and must not be reported as a separate finding.
    """
    path = os.path.join(results_directory or "", "nodes.csv")
    if not results_directory or not os.path.isfile(path):
        return ["", "Malformed-node ranking NOT run: pass results_directory from a "
                    "measure_lattice_iou / detect_lattice_defects run (it needs nodes.csv)."]
    try:
        d = np.genfromtxt(path, delimiter=",", names=True)
        ok = (d["is_interior"] == 1) & (d["embedded"] == 0)
        dia, z, ids = d["diameter_fill_um"], d["z"], d["node_id"]
    except (OSError, ValueError, KeyError) as error:
        return ["", f"Malformed-node ranking unavailable: {error}"]
    if ok.sum() < 50:
        return ["", "Malformed-node ranking skipped: too few measurable interior nodes."]

    absent = ok & (dia <= 0)
    ratio, expected = lattice_iou.detrend_by_height(dia, z, exclude=absent)
    live = ok & ~absent
    v = np.sort(dia[live])
    corr = float(np.corrcoef(z[live], dia[live])[0, 1])

    out = [
        "",
        "MALFORMED junctions -- printed but not formed. A different failure from absence, "
        "and invisible to the sphere fill above, which saturates.",
        f"  junction size (diameter at 90% ball fill) over {int(live.sum())} measurable "
        f"interior nodes: median {np.median(v):.0f} um, p1 {np.percentile(v, 1):.0f}, "
        f"p5 {np.percentile(v, 5):.0f} um",
        f"  size vs build height: r = {corr:+.3f}"
        + ("  -- strong, so the raw ranking is partly a list of tall nodes; rank on the "
           "detrended ratio" if abs(corr) > 0.2 else "  -- weak, raw and detrended agree"),
        "",
        "  rank  node    z     diameter_um   gap_to_next   ratio_to_height   bad_struts",
    ]
    nb = None
    if inc_bad is not None and len(inc_bad) >= len(dia):
        nb = np.asarray(inc_bad)[:len(dia)]
    order = np.flatnonzero(live)[np.argsort(dia[live])]
    for rank, k in enumerate(order[:show]):
        gap = (dia[order[rank + 1]] - dia[k]) if rank + 1 < len(order) else float("nan")
        out.append(f"  {rank:4d}  {int(ids[k]):5d}  {z[k]:5.0f}   {dia[k]:9.1f}   "
                   f"{gap:9.1f}      {ratio[k]:11.3f}   "
                   f"{(int(nb[k]) if nb is not None else -1):10d}")

    gaps = np.diff(v[:60])
    g = int(np.argmax(gaps))
    out += [
        "",
        f"  largest gap in the low tail: {v[g]:.1f} -> {v[g + 1]:.1f} um "
        f"({gaps[g]:.1f} um wide), separating {g + 1} node(s) below it.",
    ]
    if gaps[g] < 30:
        out.append("  That gap is narrow, so the low tail is a continuum and any cut here "
                   "IS a choice. Report the ranking, not a count.")
    r_order = np.flatnonzero(live)[np.argsort(ratio[live])]
    out.append(f"  detrended ranking agrees on the top "
               f"{sum(1 for a, b in zip(order[:5], r_order[:5]) if a == b)} of 5.")

    if nb is not None:
        corr_b = float(np.corrcoef(nb[live], dia[live])[0, 1])
        out += ["",
                f"  SECOND CONFOUND: size vs defective incident struts, r = {corr_b:+.3f}. "
                f"A junction with struts missing has less material near it whether or not "
                f"the junction failed, so compare only within a strut-damage class:",
                "     bad struts   nodes   median size um   the top candidate"]
        top = int(order[0])
        for kb in range(0, int(nb[live].max()) + 1):
            peer = live & (nb == kb)
            if not peer.any():
                continue
            here = "  <-- it is here" if int(nb[top]) == kb else ""
            out.append(f"     {kb:10d}  {int(peer.sum()):6d}   {np.median(dia[peer]):14.0f}"
                       f"{here}")
        peer = live & (nb == nb[top])
        pct = 100.0 * float((dia[peer] < dia[top]).mean())
        out.append(f"  node {int(ids[top])} sits at the {pct:.0f}th percentile of the "
                   f"{int(peer.sum())} nodes with {int(nb[top])} defective struts "
                   f"(their median {np.median(dia[peer]):.0f} um vs its {dia[top]:.0f} um).")
        if pct > 10.0:
            out.append("  It is NOT unusual for its strut damage -- treat it as a "
                       "consequence of those struts, not a separate junction failure.")
        else:
            out.append("  It is the smallest in its own class, so its size is not "
                       "explained by its dead struts alone.")
    if int(absent.sum()):
        out.append(f"  ({int(absent.sum())} absent nodes excluded from this ranking and "
                   f"from the height fit -- they are reported above.)")
    return out


@mcp.tool()
def detect_missing_nodes_2d(
    mask_filepath: str,
    r_thr_vox: float = 0.0,
    match_frac: float = 0.25,
    plate_frac: float = 0.20,
) -> str:
    """
    Finds junctions that were never printed WITHOUT using the design or the registration.

    Every other node tool here looks a design junction up in the CT, so its answer is only
    as good as the transform and it can never see a node the design does not mention. This
    one opens nothing but the mask. It locates the node planes from the periodicity of the
    image itself, finds the nodes as locally-fat regions, works out the lattice they sit on
    from the nodes, and asks which sites of that lattice have nothing at them. Agreement
    with `detect_missing_nodes` therefore means something, because the two share no inputs
    beyond the voxels.

    Why per-slice works for nodes when it failed for struts on this data: an octet strut
    runs at 45 degrees so no plane contains one, but nodes occupy discrete z planes ~39 vox
    apart and the specimen tilt is only ~3 vox across its full width, so a node plane is
    essentially one slice.

    Read the output in this order:

      * `pitch` and `basis angle` are the check that the lattice fit is real. They are
        recovered from the blobs alone; for an octet they must land on cell/sqrt(2) and
        90 degrees. If they do not, nothing below is trustworthy.
      * `interior sites empty` is the result. Sites are the integer lattice points inside
        the CONVEX HULL of the observed nodes, so the detector interpolates and never
        extrapolates -- it cannot invent nodes past the edge of the part. The price is that
        it is blind to a whole absent outer row, which is exactly the case only a design
        comparison can catch, so run `detect_missing_nodes` as well and compare.
      * The threshold sweep. An empty site means "no node-sized object here", which is
        absence OR severe undersizing; the sweep shows over what range of `r_thr` the count
        is stable. Separate the two by measuring each candidate (see
        `scripts/node_planes_2d.py`, which reports mask material fraction and inscribed
        radius per candidate and renders the annotated slice).

    Build-plate slices are excluded by foreground fraction, and any plane whose lattice fit
    does not converge is rejected and named rather than silently averaged in.

    Args:
        mask_filepath: Binary segmentation .tif/.npy in (z, y, x).
        r_thr_vox: Inscribed radius that defines a node blob. 0 (default) reads it off the
            valley of the bimodal EDT histogram instead of asserting one.
        match_frac: A site counts as answered by a node within this fraction of the pitch.
        plate_frac: Foreground fraction above which a slice is inside a build plate.

    Returns:
        Per-plane counts, the recovered lattice, the empty interior sites with coordinates,
        and the threshold sweep, or an error message.
    """
    if not os.path.isfile(mask_filepath):
        return f"Error: mask file not found at {mask_filepath}"
    if not 0.0 < match_frac < 0.5:
        return f"Error: match_frac must be in (0, 0.5) (got {match_frac})"

    try:
        mask = _load_volume(mask_filepath) > 0
        result = node_planes_2d.run(mask, r_thr=(r_thr_vox or None),
                                    plate_frac=plate_frac, match_frac=match_frac,
                                    log=lambda *_: None)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return f"Error: could not run the 2D node-plane detector: {error}"

    planes = result["planes"]
    if not planes:
        return ("Error: no usable node planes. Every candidate plane failed the lattice "
                "fit; check that this is a lattice mask and not a solid part.")

    pitch = float(np.median([p["pitch_vox"] for p in planes]))
    angle = float(np.median([p["basis_angle_deg"] for p in planes]))
    lines = [
        f"{len(planes)} usable node planes, z {planes[0]['z']}..{planes[-1]['z']} "
        f"(spacing median {np.median(np.diff([p['z'] for p in planes])):.1f} vox)",
        f"  node blob radius threshold {result['r_thr']:.2f} vox "
        + ("(read off the EDT histogram valley)" if not r_thr_vox else "(given)"),
        f"  lattice recovered from the nodes alone: pitch {pitch:.2f} vox at "
        f"{angle:.1f} deg",
    ]
    if not 85.0 <= angle <= 95.0:
        lines.append("  WARNING: the basis is not square. For an octet it should be 90 "
                     "deg; the fit has not found the lattice and the counts below mean "
                     "nothing.")
    for z, why in result["rejected"]:
        lines.append(f"  plane z={z} rejected: {why}")

    lines.append("")
    lines.append("   z    blobs  sites  interior  empty(interior)  empty(edge ring)")
    n_int = n_miss = n_edge = 0
    candidates = []
    for res in planes:
        interior = ~res["edge"]
        miss = interior & ~res["found"]
        n_int += int(interior.sum())
        n_miss += int(miss.sum())
        n_edge += int((res["edge"] & ~res["found"]).sum())
        lines.append(f"  {res['z']:4d}  {len(res['centroids']):5d}  "
                     f"{len(res['sites_xy']):5d}  {int(interior.sum()):8d}  "
                     f"{int(miss.sum()):15d}  {int((res['edge'] & ~res['found']).sum()):16d}")
        for s in np.flatnonzero(miss):
            candidates.append((res["z"], res["sites_xy"][s], res["dist"][s]))

    lines += [
        "",
        f"interior sites tested {n_int}, EMPTY {n_miss} "
        f"({100 * n_miss / max(n_int, 1):.3f}%)",
        f"edge-ring sites empty {n_edge} -- an extent question, not a defect claim; the "
        f"hull cannot tell a missing outer node from a part that ends there.",
    ]
    for z, xy, d in candidates:
        lines.append(f"  empty interior site  z={z} y={xy[0]:.0f} x={xy[1]:.0f}  "
                     f"(nearest node blob {d:.1f} vox away)")
    if candidates:
        lines.append("  Each is 'no node-sized object here' -- absence or severe "
                     "undersizing. Measure them before reporting which.")

    sweep = node_planes_2d.threshold_sweep(
        mask, [p["z"] for p in planes], [3.5, 4.0, 4.5, 5.0, 5.5, 6.0], match_frac)
    lines.append("")
    lines.append("threshold sweep -- how much of this is the threshold:")
    for r, ni, nm in sweep:
        mark = "  <-- used" if abs(r - result["r_thr"]) < 0.3 else ""
        lines.append(f"  r_thr {r:4.1f}   interior sites {ni:6d}   empty {nm:4d}{mark}")
    return "\n".join(lines)


@mcp.tool()
def visualize_strut_classes(
    mask_filepath: str,
    design_filepath: str,
    results_directory: str,
    output_filepath: str,
    correction_filepath: str = "",
    strut_ids: str = "",
    classes: str = "",
    per_class: int = 1,
    n_sections: int = 8,
    seed: int = 0,
) -> str:
    """
    Renders the VOXELS behind each strut's classification, one strut per row.

    Every strut number this project reports -- missing, broken, thin, thick, necked,
    nominal -- comes from a measurement you cannot check by reading a CSV. This draws the
    evidence: for each chosen strut, a lateral slice through its own axis on the left and
    a run of cross-sections cut PERPENDICULAR to that axis on the right, with the nominal
    350 um circle on every one.

    The two views fail in opposite directions, which is why both are drawn. The lateral
    view shows where material starts and stops -- a gap is a white column and needs no
    statistic -- but it is one plane through the axis, so it misses anything off that
    plane. The cross-sections see all the way round but are independent planes, so they
    cannot show continuity. Side by side each covers the other's blind spot.

    Use it two ways:
      * survey -- leave `strut_ids` empty to draw `per_class` random struts from each
        class. This is the figure to look at before trusting any class count.
      * inspection -- pass explicit `strut_ids` to interrogate particular struts, e.g.
        ones a reviewer questions or the borderline cases at a threshold.

    Reads the measurements a previous `detect_lattice_defects` run cached, so it is fast
    (seconds) and always shows exactly the struts those numbers describe. It does not
    re-measure or re-classify anything.

    Args:
        mask_filepath: Binary segmentation .tif/.npy in (z, y, x).
        design_filepath: Registered lattice .json.
        results_directory: Directory from a `detect_lattice_defects` run. Must contain
            summary.json, struts.csv, strut_classes.csv and sections.npz.
        output_filepath: Where to write the .png.
        correction_filepath: Registration correction.json. Strongly recommended -- the
            sections are cut about the design axis, so an uncorrected transform draws the
            window in the wrong place and every strut looks off-centre.
        strut_ids: Comma-separated strut ids to draw, in order. Overrides `classes` and
            `per_class`.
        classes: Comma-separated class names to include in the survey (default: all).
        per_class: Struts drawn per class in the survey view.
        n_sections: Cross-sections drawn per strut.

    Returns:
        What was drawn, with each strut's measurements, or an error message.
    """
    for path, what in ((mask_filepath, "mask"), (design_filepath, "design JSON")):
        if not os.path.isfile(path):
            return f"Error: {what} file not found at {path}"
    if correction_filepath and not os.path.isfile(correction_filepath):
        return f"Error: correction file not found at {correction_filepath}"
    needed = ("summary.json", "struts.csv", "strut_classes.csv", "sections.npz")
    missing = [f for f in needed
               if not os.path.isfile(os.path.join(results_directory or "", f))]
    if missing:
        return (f"Error: {results_directory} is missing {', '.join(missing)}. Run "
                f"detect_lattice_defects into that directory first -- this tool draws "
                f"what that run measured, it does not measure anything itself.")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from scripts.classify_strut_defects import (CLASS_COLORS, ORDER, measurable,
                                                    plot_section_gallery)
    except ImportError as error:
        return f"Error: could not import the gallery renderer: {error}"

    try:
        root = results_directory
        summary = json.load(open(os.path.join(root, "summary.json")))
        um = summary["geometry"]["um_per_voxel"]
        r_strut = summary["geometry"]["nominal_strut_radius_vox"]
        nominal_um = summary["geometry"]["nominal_strut_diameter_um"]

        pos, pairs, _, corrected = lattice_iou.load_design(
            design_filepath, correction_filepath or None)
        mask = _load_volume(mask_filepath) > 0

        z = np.load(os.path.join(root, "sections.npz"), allow_pickle=False)
        sec = {k[4:]: z[k] for k in z.files if k.startswith("sec_")}
        n_total = int(z["prof_r_eq"].shape[1])
        cls = np.genfromtxt(os.path.join(root, "strut_classes.csv"), delimiter=",",
                            names=True, dtype=None, encoding=None)
        lab = cls["label"]
        d = np.genfromtxt(os.path.join(root, "struts.csv"), delimiter=",", names=True)
        n = min(len(sec["r_eq_med"]), len(lab))
        d = {k: d[k][:n] for k in d.dtype.names}
        sec = {k: v[:n] for k, v in sec.items()}
        lab = lab[:n]
        ok = measurable(d, sec)

        extra = None
        cpath = os.path.join(root, "connectivity.npz")
        if os.path.isfile(cpath):
            extra = dict(np.load(cpath, allow_pickle=False))
    except (OSError, ValueError, KeyError) as error:
        return f"Error: could not load the cached measurements: {error}"

    picks, title = None, None
    if strut_ids.strip():
        try:
            ids = [int(s) for s in strut_ids.replace(" ", "").split(",") if s]
        except ValueError:
            return f"Error: could not parse strut_ids {strut_ids!r} as integers"
        bad = [i for i in ids if not 0 <= i < n]
        if bad:
            return f"Error: strut id(s) {bad} out of range 0..{n - 1}"
        picks = [(str(lab[i]), i) for i in ids]
        title = (f"{len(ids)} requested struts: lateral view through the axis (left) and "
                 f"cross-sections along it (right)   red = nominal {nominal_um:.0f} um")
    elif classes.strip():
        want = {c.strip() for c in classes.split(",") if c.strip()}
        unknown = want - set(ORDER)
        if unknown:
            return (f"Error: unknown class(es) {sorted(unknown)}. "
                    f"Known classes: {', '.join(ORDER)}")
        rng = np.random.default_rng(seed)
        picks = []
        for name in ORDER:
            if name not in want:
                continue
            idx = np.flatnonzero(ok & (lab == name))
            if idx.size:
                take = rng.choice(idx, size=min(per_class, idx.size), replace=False)
                picks += [(name, int(k)) for k in np.atleast_1d(take)]
        if not picks:
            return (f"Error: no measurable strut in {sorted(want)}. Counts among "
                    f"measurable struts: "
                    + ", ".join(f"{c}={int((ok & (lab == c)).sum())}" for c in ORDER))

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        plot_section_gallery(mask, pos, pairs, r_strut, lab, sec, ok, output_filepath,
                             um, n_total, n_show=n_sections, seed=seed,
                             nominal_um=nominal_um, extra=extra, picks=picks,
                             per_class=per_class, title=title)
    except (OSError, ValueError, KeyError, IndexError) as error:
        return f"Error: could not render the gallery: {error}"

    lines = [f"Wrote {output_filepath}",
             f"  registration correction applied: {corrected}"
             + ("" if corrected else "   <-- NOT APPLIED, the section windows are cut "
                                     "about the design axis and will be off-centre"),
             f"  {n_sections} cross-sections per strut, drawn from the {n_total} the "
             f"cached run measured; nominal {nominal_um:.0f} um circle in red",
             ""]
    if picks is None:
        lines.append(f"Survey view: {per_class} random strut(s) per class, of")
        for c in ORDER:
            lines.append(f"  {c:8s} {int((ok & (lab == c)).sum()):6d} measurable struts")
        lines.append("Pass strut_ids to draw particular struts instead.")
    else:
        lines.append("Struts drawn:")
        for name, sid in picks:
            lines.append(
                f"  #{sid:<6d} {name:8s} median {sec['r_eq_med'][sid] * 2 * um:4.0f} um  "
                f"min {sec['r_eq_min'][sid] * 2 * um:4.0f} um  "
                f"empty {int(sec['empty_sections'][sid])}/{n_total}"
                + ("" if extra is None else
                   ("  no node-to-node path" if not extra["reachable"][sid]
                    else f"  detour {extra['detour'][sid]:.3f}x"))
                + ("" if ok[sid] else "   <-- NOT MEASURABLE (boundary, plate-embedded "
                                      "or window-touching); its class is not trustworthy"))
    return "\n".join(lines)


@mcp.tool()
def validate_against_stl(
    nominal_design_filepath: str,
    stl_filepath: str,
    results_directory: str,
    orientation: str = "",
    absent_tol_mm: float = 0.7,
) -> str:
    """
    Scores the detected missing struts against which struts the CAD actually removed.

    This is the only check here that is not another measurement of the same scan. Three
    instruments agreeing on 0.5% would look identical whether or not those are the right
    struts; `0.5.stl` and `1.stl` are the design with struts deliberately deleted, so the
    list of which ones is a ground truth no reading of the CT can talk itself into.

    A strut is designed-out if no triangle lies within `absent_tol_mm` of its MIDSPAN.
    Sampling near the ends finds the neighbouring junctions, which are still present, so a
    removed strut would look supported.

    PASS THE NOMINAL DESIGN JSON, NOT THE REGISTERED ONE. The STL lives in the design's
    own frame (mm, centred on the origin), not in CT voxels. The registered JSON is in CT
    coordinates and will align with nothing here.

    THE ORIENTATION IS THE SUBTLE PART. Scale and centring come exactly off the bounding
    boxes, leaving one of the cube's 48 signed axis permutations -- and that CANNOT be
    fixed by matching the complete `0.stl`, which is invariant under all 48. This tool
    therefore resolves it against the DEFECT stl, using the detected missing struts as the
    reference pattern, and reports the margin over the runner-up so you can judge the
    identification instead of trusting it. It also runs a check that owes nothing to the
    detector: the winning orientation must put the STL's build-plate axis on the design's
    build axis. Treat the score as provisional if either the margin is small or that check
    fails.

    Because the orientation is chosen to agree with the detector, precision and recall are
    not a fully blind test -- but the COUNT of designed-out struts is independent of the
    detector entirely, and comparing it against the specimen's nominal percentage is.

    Args:
        nominal_design_filepath: The unregistered octet_truss_9x9x9.json (all struts).
        stl_filepath: A defect STL, e.g. data/missing_struts/stls/0.5.stl. Passing the
            complete 0.stl is rejected -- it carries no orientation information.
        results_directory: A detect_lattice_defects run (needs strut_classes.csv,
            struts.csv).
        orientation: Optional "perm|signs" such as "2,0,1|-1,-1,-1" to skip the search
            and use a known alignment.
        absent_tol_mm: Distance from the midspan beyond which a strut counts as absent.

    Returns:
        The designed-out count with its percentage, the orientation and its margin, and
        precision / recall / F1 per predicted class, or an error message.
    """
    for path, what in ((nominal_design_filepath, "design JSON"), (stl_filepath, "STL")):
        if not os.path.isfile(path):
            return f"Error: {what} not found at {path}"
    needed = ("strut_classes.csv", "struts.csv")
    absent_files = [f for f in needed
                    if not os.path.isfile(os.path.join(results_directory or "", f))]
    if absent_files:
        return (f"Error: {results_directory} is missing {', '.join(absent_files)}. Run "
                f"detect_lattice_defects into that directory first.")

    try:
        pos, pairs, _, corrected = lattice_iou.load_design(nominal_design_filepath, None)
        if corrected:
            return "Error: do not pass a registration correction here; the STL is in the design frame."
        extent = float(np.ptp(pos))
        if not 10.0 < extent < 40.0:
            return (f"Error: this design spans {extent:.1f} units, which looks like CT "
                    f"voxels rather than design units. Pass the NOMINAL "
                    f"octet_truss_9x9x9.json, not the registered one.")
        centroids, n_tri = stl_ground_truth.read_stl_centroids(stl_filepath)
        scale, span = stl_ground_truth.design_to_stl_scale(centroids)

        cls = np.genfromtxt(os.path.join(results_directory, "strut_classes.csv"),
                            delimiter=",", names=True, dtype=None, encoding=None)
        st = np.genfromtxt(os.path.join(results_directory, "struts.csv"),
                           delimiter=",", names=True)
        lab = cls["label"]
        n = len(lab)
        usable = np.zeros(len(pairs), bool)
        usable[:n] = (st["is_boundary"][:n] == 0) & (st["embedded"][:n] == 0)
        detected = np.zeros(len(pairs), bool)
        detected[:n] = (lab == "missing") & usable[:n]
    except (OSError, ValueError, KeyError) as error:
        return f"Error: could not load the inputs: {error}"

    lines = [f"{os.path.basename(stl_filepath)}: {n_tri} triangles, "
             f"bbox span {np.round(span, 2).tolist()} mm",
             f"  {scale:.4f} mm per design unit (lattice axes / {int(18)} units)"]

    if orientation.strip():
        try:
            p_txt, s_txt = orientation.split("|")
            perm = tuple(int(v) for v in p_txt.split(","))
            signs = tuple(int(v) for v in s_txt.split(","))
            if sorted(perm) != [0, 1, 2] or set(signs) - {1, -1}:
                raise ValueError
        except ValueError:
            return (f"Error: could not parse orientation {orientation!r}; expected "
                    f'"2,0,1|-1,-1,-1"')
        ranked = None
    else:
        if not detected.any():
            return ("Error: no struts are labelled missing, so there is no pattern to "
                    "align the STL against. The orientation cannot be found from a "
                    "complete lattice -- it is invariant under all 48 symmetries.")
        perm, signs, ranked = stl_ground_truth.find_orientation(
            centroids, pos, pairs, detected, scale, absent_tol_mm)

    worst, truth = stl_ground_truth.strut_support(
        centroids, pos, pairs, perm, signs, scale, absent_tol_mm)

    if not truth.any():
        return (f"{os.path.basename(stl_filepath)} has every strut present -- no strut's "
                f"midspan is further than {absent_tol_mm} mm from the mesh (worst "
                f"{worst.max():.3f} mm).\n"
                "There is nothing to score against, and a complete lattice cannot fix "
                "the orientation either: it is invariant under all 48 cube symmetries. "
                "Pass a defect STL (0.5.stl, 1.stl). Use 0.stl only as the reference "
                "that shows what full support looks like.")

    if ranked is not None:
        best, second = ranked[0][0], ranked[1][0]
        chance = detected.sum() * max(int(truth.sum()), 1) / max(len(pairs), 1)
        lines += [
            f"  orientation perm {tuple(perm)} signs {tuple(signs)}: matches "
            f"{best} of the {int(detected.sum())} detected; runner-up {second}, "
            f"chance {chance:.1f}",
        ]
        if best < 3 * max(second, 1):
            lines.append("  WARNING: the margin over the runner-up is small, so the "
                         "alignment is NOT established. Everything below is unreliable.")
    else:
        lines.append(f"  orientation perm {tuple(perm)} signs {tuple(signs)} (given)")

    axis_name, ok_axis = stl_ground_truth.plate_axis_check(span, perm, signs)
    lines.append(f"  independent check: the STL's plate axis maps to design {axis_name}"
                 + ("  <-- the build axis, as it must" if ok_axis else
                    "  <-- NOT the build axis; the alignment is probably wrong"))

    lines += ["",
              f"DESIGNED OUT: {int(truth.sum())} of {len(pairs)} struts "
              f"({100 * truth.mean():.3f}%)   "
              f"[{int((truth & usable).sum())} measurable, "
              f"{int((truth & ~usable).sum())} boundary/plate-excluded]",
              "  This count owes nothing to the detector -- compare it against the "
              "specimen's nominal percentage as a check on the whole chain.",
              ""]

    lines.append("  class      predicted    TP    FP    FN   precision  recall     F1")
    for name in ("missing", "broken", "thin", "thick"):
        pred = np.zeros(len(pairs), bool)
        pred[:n] = (lab == name)
        s = stl_ground_truth.score(pred, truth, usable)
        lines.append(f"  {name:9s} {s['n_pred']:9d} {s['tp']:5d} {s['fp']:5d} "
                     f"{s['fn']:5d}     {s['precision']:.3f}   {s['recall']:.3f}  "
                     f"{s['f1']:.3f}")

    got = collections.Counter(lab[(truth[:n]) & usable[:n]].tolist())
    lines += ["", "  what the designed-out struts were actually labelled: "
                  + ", ".join(f"{k}={v}" for k, v in got.most_common())]
    return "\n".join(lines)


if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
