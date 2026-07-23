"""Regression tests for JSON-to-volume axis ordering in overlay helpers."""

import numpy as np

from src.mcp_server import _json_position_to_volume_position


def test_json_position_to_volume_position_reorders_xyz_to_zyx():
    json_position = np.array([10.0, 20.0, 30.0])

    converted = _json_position_to_volume_position(json_position)

    assert np.array_equal(converted, np.array([30.0, 20.0, 10.0]))
