import pytest

from src.data.crawler import *


def test_load_domains():
    domains = load_domains()
    conditions = [
        d.startswith("http") and (d.endswith(".pt/") or d.endswith(".com/"))
        for d in domains
    ]
    assert all(conditions)
