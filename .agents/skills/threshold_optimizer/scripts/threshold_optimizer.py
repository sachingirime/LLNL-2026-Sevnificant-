import argparse
import os
import sys
from typing import List

# Ensure this script can import the repository's src package when run from the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.mcp_server import segment_ct_dataset


def parse_thresholds(value: str) -> List[float]:
    thresholds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            thresholds.append(float(part))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid threshold value: {part}")
    if not thresholds:
        raise argparse.ArgumentTypeError("At least one threshold value is required.")
    return thresholds


def format_threshold(threshold: float) -> str:
    return str(int(threshold)) if threshold.is_integer() else str(threshold)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare segmentation thresholds by calling the segment_ct_dataset MCP tool."
    )
    parser.add_argument(
        "input_filepath",
        help="Path to the input .npy file to segment.",
    )
    parser.add_argument(
        "--thresholds",
        default="300,500,700",
        help="Comma-separated list of threshold values to test."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/threshold_optimizer",
        help="Directory where thresholded segmentation files will be saved."
    )
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for threshold in thresholds:
        output_filename = f"segmentation_threshold_{format_threshold(threshold)}.npy"
        output_filepath = os.path.join(args.output_dir, output_filename)
        status = segment_ct_dataset(args.input_filepath, output_filepath, threshold)
        results.append((threshold, output_filepath, status))
        print(status)

    print("\nThreshold comparison summary:")
    print(f"{'threshold':>10}  {'output file':<60}  {'status'}")
    for threshold, output_filepath, status in results:
        summary_status = status.split("\n", 1)[0]
        print(
            f"{format_threshold(threshold):>10}  {output_filepath:<60}  {summary_status}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
