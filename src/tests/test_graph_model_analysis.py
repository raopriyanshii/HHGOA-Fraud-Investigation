"""Focused tests for the device-profile composite key logic -- this is the
one piece of Phase C logic with a real, non-obvious policy decision
(require DeviceInfo, use an explicit UNKNOWN token for missing sub-parts).
"""
import pandas as pd

from src.data.graph_model_analysis import build_device_profile_key


def test_missing_device_info_yields_no_profile():
    row = {"DeviceInfo": float("nan"), "id_30": "Windows 10", "id_31": "chrome 63.0", "id_33": "1920x1080"}
    assert build_device_profile_key(row) is None


def test_full_profile_is_pipe_joined():
    row = {"DeviceInfo": "SM-G935F Build/NRD90M", "id_30": "Android 7.0", "id_31": "chrome 62.0 for android", "id_33": "1920x1080"}
    assert build_device_profile_key(row) == "SM-G935F Build/NRD90M|Android 7.0|chrome 62.0 for android|1920x1080"


def test_missing_subparts_become_explicit_unknown_not_dropped():
    row = {"DeviceInfo": "Windows", "id_30": float("nan"), "id_31": "chrome 63.0", "id_33": float("nan")}
    assert build_device_profile_key(row) == "Windows|UNKNOWN|chrome 63.0|UNKNOWN"


def test_two_different_unknown_states_are_not_accidentally_equal():
    # a device with a genuinely unknown OS must not collide with a
    # different device that also has an unknown OS, unless every other
    # component matches too -- the join key itself, not the UNKNOWN token,
    # provides the identity
    row_a = {"DeviceInfo": "Windows", "id_30": float("nan"), "id_31": "chrome 63.0", "id_33": float("nan")}
    row_b = {"DeviceInfo": "MacOS", "id_30": float("nan"), "id_31": "chrome 63.0", "id_33": float("nan")}
    assert build_device_profile_key(row_a) != build_device_profile_key(row_b)
