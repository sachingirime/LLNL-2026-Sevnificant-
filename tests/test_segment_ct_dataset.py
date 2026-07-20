"""Tests for the Task 1 MCP CT-segmentation tool."""

import numpy as np

from src.mcp_server import segment_ct_dataset


def test_segment_ct_dataset_creates_binary_uint8_mask(tmp_path):
    input_path = tmp_path / "ct.npy"
    output_path = tmp_path / "results" / "segmentation.npy"
    volume = np.array(
        [[[0.1, 0.5], [0.6, 0.4]], [[0.8, 0.2], [0.5, 0.0]]],
        dtype=np.float32,
    )
    np.save(input_path, volume)

    message = segment_ct_dataset(str(input_path), str(output_path), threshold=0.5)

    assert message.startswith("Saved binary segmentation")
    assert np.array_equal(
        np.load(output_path),
        np.array(
            [[[0, 1], [1, 0]], [[1, 0], [1, 0]]],
            dtype=np.uint8,
        ),
    )


def test_segment_ct_dataset_rejects_non_npy_outputs(tmp_path):
    input_path = tmp_path / "ct.npy"
    np.save(input_path, np.zeros((2, 2, 2), dtype=np.float32))

    message = segment_ct_dataset(str(input_path), str(tmp_path / "mask.tif"), 0.5)

    assert message == "Error: output_filepath must end with '.npy'"
