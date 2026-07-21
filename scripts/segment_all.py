"""
Batch-segment every 3D CT volume under data/ using the Task 1 MCP tool.

This calls ``segment_ct_dataset`` / ``visualize_slice`` from src/mcp_server.py
directly, so the batch run exercises exactly the same code path Codex CLI hits
through MCP -- it is an automation wrapper, not a reimplementation.

Thresholds are chosen per dataset, in this order:
    1. an explicit --threshold NAME=VALUE on the command line,
    2. a pinned value in KNOWN_THRESHOLDS below,
    3. Otsu's method on a strided subsample of the volume.

Usage:
    python scripts/segment_all.py
    python scripts/segment_all.py --threshold unitcell=0.005 --threshold 9x9x9_octet_lattice=32768
    python scripts/segment_all.py --dry-run          # report thresholds, write nothing
    python scripts/segment_all.py --no-previews      # skip the mid-slice PNGs

Note: data/ is read-only. Everything is written under outputs/task1/.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import tifffile
from skimage.filters import threshold_otsu

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.mcp_server import segment_ct_dataset, visualize_slice  # noqa: E402

VOLUME_EXTENSIONS = (".npy", ".tif", ".tiff")

# Thresholds we have already validated against the ground-truth slices. Otsu is
# a reasonable starting point but tends to over-segment the unitcell volume,
# whose density range is narrow (min=-0.0031, max=0.0153).
KNOWN_THRESHOLDS = {
    "unitcell": 0.005,
}

# Otsu on a full 1 GB volume is wasteful; every Nth voxel per axis is plenty to
# recover the same histogram shape.
OTSU_STRIDE = 4


def find_volumes(data_dir: str) -> list[str]:
    """Returns every 3D-volume file under data_dir, sorted for stable runs."""
    volumes = []
    for dirpath, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.lower().endswith(VOLUME_EXTENSIONS):
                volumes.append(os.path.join(dirpath, filename))
    return sorted(volumes)


def file_digest(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_name(path: str, data_dir: str) -> str:
    """`data/unitcell/unitcell.npy` -> `unitcell`; nested files keep their folder."""
    relative = os.path.relpath(path, data_dir)
    parts = relative.split(os.sep)
    return parts[0] if len(parts) > 1 else os.path.splitext(parts[0])[0]


def subsample(path: str, stride: int = OTSU_STRIDE) -> np.ndarray:
    """Loads every `stride`-th voxel per axis without materializing the volume."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        volume = np.load(path, mmap_mode="r")
        return np.asarray(volume[::stride, ::stride, ::stride])

    with tifffile.TiffFile(path) as handle:
        depth = len(handle.pages)
        planes = [handle.pages[i].asarray()[::stride, ::stride] for i in range(0, depth, stride)]
    return np.stack(planes)


def volume_shape(path: str) -> tuple[int, ...]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path, mmap_mode="r").shape
    with tifffile.TiffFile(path) as handle:
        return handle.series[0].shape


def choose_threshold(path: str, name: str, overrides: dict[str, float]) -> tuple[float, str]:
    """Returns (threshold, provenance) so the manifest records where it came from."""
    if name in overrides:
        return overrides[name], "cli"
    if name in KNOWN_THRESHOLDS:
        return KNOWN_THRESHOLDS[name], "pinned"
    sample = subsample(path)
    return float(threshold_otsu(sample)), f"otsu(stride={OTSU_STRIDE})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-segment every CT volume under data/.")
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs", "task1"))
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Pin a threshold for one dataset, e.g. --threshold unitcell=0.005. Repeatable.",
    )
    parser.add_argument("--no-previews", action="store_true", help="Skip the mid-slice PNGs.")
    parser.add_argument("--dry-run", action="store_true", help="Report thresholds without writing.")
    args = parser.parse_args()

    overrides = {}
    for item in args.threshold:
        if "=" not in item:
            parser.error(f"--threshold expects NAME=VALUE, got {item!r}")
        name, _, value = item.partition("=")
        try:
            overrides[name] = float(value)
        except ValueError:
            parser.error(f"--threshold value for {name!r} is not a number: {value!r}")

    volumes = find_volumes(args.data_dir)
    if not volumes:
        print(f"No .npy/.tif volumes found under {args.data_dir}")
        return 1

    records = []
    seen_digests: dict[str, str] = {}

    for path in volumes:
        name = dataset_name(path, args.data_dir)
        relative = os.path.relpath(path, REPO_ROOT)
        print(f"\n=== {relative}")

        # The missing_struts TIFF stack is byte-identical to the 9x9x9 volume;
        # segmenting it twice would just duplicate a 1 GB output.
        digest = file_digest(path)
        if digest in seen_digests:
            print(f"  skipped: identical to {seen_digests[digest]}")
            records.append({"input": relative, "dataset": name, "status": "duplicate",
                            "duplicate_of": seen_digests[digest]})
            continue
        seen_digests[digest] = relative

        shape = volume_shape(path)
        if len(shape) != 3:
            print(f"  skipped: not a 3D volume (shape={shape})")
            records.append({"input": relative, "dataset": name, "status": "not_3d", "shape": list(shape)})
            continue

        threshold, provenance = choose_threshold(path, name, overrides)
        print(f"  shape={shape}  threshold={threshold:g}  ({provenance})")

        if args.dry_run:
            records.append({"input": relative, "dataset": name, "status": "dry_run",
                            "shape": list(shape), "threshold": threshold,
                            "threshold_source": provenance})
            continue

        dataset_dir = os.path.join(args.output_dir, name)
        mask_path = os.path.join(dataset_dir, f"{name}_segmentation_threshold_{threshold:g}.npy")

        message = segment_ct_dataset(path, mask_path, threshold)
        print(f"  {message}")
        if message.startswith("Error"):
            records.append({"input": relative, "dataset": name, "status": "error", "message": message})
            continue

        record = {
            "input": relative,
            "dataset": name,
            "status": "ok",
            "shape": list(shape),
            "threshold": threshold,
            "threshold_source": provenance,
            "mask": os.path.relpath(mask_path, REPO_ROOT),
            "previews": [],
        }

        if not args.no_previews:
            for axis in (0, 1, 2):
                index = shape[axis] // 2
                preview_path = os.path.join(
                    dataset_dir, f"{name}_segmented_slice_{index}_axis{axis}.png"
                )
                preview_message = visualize_slice(mask_path, preview_path, index, axis)
                print(f"  {preview_message}")
                if not preview_message.startswith("Error"):
                    record["previews"].append(os.path.relpath(preview_path, REPO_ROOT))

        records.append(record)

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "segmentation_manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "runs": records},
            handle,
            indent=2,
        )
    print(f"\nWrote manifest to {os.path.relpath(manifest_path, REPO_ROOT)}")

    failures = [r for r in records if r["status"] == "error"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
