import pytest

from shadho.backend.base.tests.test_base_db import TestBaseBackend
from shadho.backend.json.db import JsonBackend

import json
import os
import shutil
import tempfile


class TestJsonBackend(object):

    def test_init(self):
        """Ensure that initialization sets up the db and filepath."""
        # Test default initialization
        b = JsonBackend()
        assert b.path == os.path.join(os.getcwd(), 'shadho.json')
        assert b.db == {'models': {},
                        'domains': {},
                        'results': {},
                        'values': {}}
        assert b.commit_frequency == 10
        assert b.update_frequency == 10

        # Test custom initialization
        b = JsonBackend(path='foo.bar',
                        commit_frequency=42,
                        update_frequency=42)
        assert b.path == os.path.join(os.getcwd(), 'foo.bar')
        assert b.db == {'models': {},
                        'domains': {},
                        'results': {},
                        'values': {}}
        assert b.commit_frequency == 42
        assert b.update_frequency == 42

        # Test without specifying a file name
        b = JsonBackend(path='/tmp')
        assert b.path == os.path.join('/tmp', 'shadho.json')
        assert b.db == {'models': {},
                        'domains': {},
                        'results': {},
                        'values': {}}
        assert b.commit_frequency == 10
        assert b.update_frequency == 10

    def test_commit(self):
        """Ensure that commit writes to file and the file is loadable."""
        temp = tempfile.mkdtemp()
        fpath = os.path.join(temp, 'shadho.json')

        # Test saving and loading
        b = JsonBackend(path=temp)
        b.commit()
        assert os.path.isfile(fpath)
        with open(fpath, 'r') as f:
            db = json.load(f)
            assert db == {'models': {},
                          'domains': {},
                          'results': {},
                          'values': {}}

        shutil.rmtree(temp)

    def test_count(self):
        """Ensure that the correct counts are returned for object classes"""
        # Test count on empty database
        b = JsonBackend()
        assert b.count('models') == 0
        assert b.count('domains') == 0
        assert b.count('results') == 0
        assert b.count('values') == 0