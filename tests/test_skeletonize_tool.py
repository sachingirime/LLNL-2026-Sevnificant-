"""Tests for the Task 3 MCP skeletonization tool."""

import numpy as np

from src.mcp_server import skeletonize


def test_skeletonize_creates_a_3d_skeleton(tmp_path):
    input_path = tmp_path / "mask.npy"
    output_path = tmp_path / "results" / "skeleton.npy"
    mask = np.zeros((7, 7, 7), dtype=np.uint8)
    mask[2:5, 2:5, 1:6] = 1
    np.save(input_path, mask)

    message = skeletonize(str(input_path), str(output_path))

    result = np.load(output_path)
    assert message.startswith("Saved 3D skeleton")
    assert result.shape == mask.shape
    assert result.dtype == bool
    assert 0 < np.count_nonzero(result) < np.count_nonzero(mask)


def test_skeletonize_rejects_an_empty_mask(tmp_path):
    input_path = tmp_path / "empty_mask.npy"
    np.save(input_path, np.zeros((3, 3, 3), dtype=np.uint8))

    message = skeletonize(str(input_path), str(tmp_path / "skeleton.npy"))

    assert message == "Error: input mask has no foreground voxels to skeletonize"
