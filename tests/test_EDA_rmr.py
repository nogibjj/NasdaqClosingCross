"""Tests"""
from src.eda_rmr_script import sanity_add


def test_sanity_add():
    """sanity check"""
    assert sanity_add(2, 3) == 5
