from __future__ import annotations

import os

import pytest

from kondor_crawler.db.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(f"sqlite:///{tmp_path / 'test.db'}")
    s.create_all()
    return s


@pytest.fixture
def pg_store():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    s = Store(url)
    s.create_all()
    return s
