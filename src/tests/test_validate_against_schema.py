"""
Focused tests for the pre-load type-compatibility checks -- these are the
functions that decide whether a column is safe to load into the GSQL type
proposed for it, so a bug here could let a real format problem through
silently.
"""
import pandas as pd

from src.graph.validate_against_schema import (
    _check_bool_string_column,
    _check_datetime_column,
    _check_nonneg_int_column,
    _check_unique_nonnull,
)


def test_datetime_check_accepts_expected_format():
    s = pd.Series(["2016-07-01 10:00:00", "2016-12-31 23:58:54", None])
    result = _check_datetime_column(s)
    assert result["n_bad_format"] == 0
    assert result["n_checked"] == 2


def test_datetime_check_flags_wrong_format():
    s = pd.Series(["2016-07-01 10:00:00", "07/01/2016", "not a date"])
    result = _check_datetime_column(s)
    assert result["n_bad_format"] == 2
    assert set(result["examples"]) == {"07/01/2016", "not a date"}


def test_nonneg_int_check_accepts_whole_numbers_incl_float_strings():
    # C1-C14 are stored as "1.0" style floats but are conceptually counts
    s = pd.Series(["1.0", "0.0", "42", 3])
    result = _check_nonneg_int_column(s)
    assert result["n_bad_format"] == 0


def test_nonneg_int_check_flags_negative_and_fractional():
    s = pd.Series(["1.0", "-5", "3.7", "not_a_number"])
    result = _check_nonneg_int_column(s)
    assert result["n_bad_format"] == 3


def test_bool_string_check_accepts_only_python_capitalized_literals():
    s = pd.Series(["True", "False", "True"])
    result = _check_bool_string_column(s)
    assert result["unexpected_values"] == []


def test_bool_string_check_flags_lowercase_or_other_values():
    s = pd.Series(["True", "false", "yes", "1"])
    result = _check_bool_string_column(s)
    assert set(result["unexpected_values"]) == {"false", "yes", "1"}


def test_unique_nonnull_detects_duplicates():
    s = pd.Series(["A", "B", "A"])
    result = _check_unique_nonnull(s, "id")
    assert result["is_unique_and_nonnull"] is False
    assert result["n_unique"] == 2


def test_unique_nonnull_detects_nulls():
    s = pd.Series(["A", None, "C"])
    result = _check_unique_nonnull(s, "id")
    assert result["is_unique_and_nonnull"] is False
    assert result["n_null"] == 1


def test_unique_nonnull_passes_clean_column():
    s = pd.Series(["A", "B", "C"])
    result = _check_unique_nonnull(s, "id")
    assert result["is_unique_and_nonnull"] is True
