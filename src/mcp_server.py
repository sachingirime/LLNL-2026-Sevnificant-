import base64
import io
import json
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


if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
