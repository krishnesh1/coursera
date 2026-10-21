import os, shutil, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_sample_data import generate

_cache = {}


def sample_data_dir():
    """Small stand-in dataset shared by the tests (created once in a temp folder)."""
    if "dir" not in _cache:
        d = tempfile.mkdtemp(prefix="capstone_data_")
        generate(d, start="2018-01-01", end="2018-12-31", scale=0.5)
        _cache["dir"] = d
    return _cache["dir"]


class IsolatedHome:
    """Points CAPSTONE_HOME at a temp folder so tests never touch real models/logs."""
    def __enter__(self):
        self.old = os.environ.get("CAPSTONE_HOME")
        self.home = tempfile.mkdtemp(prefix="capstone_home_")
        os.environ["CAPSTONE_HOME"] = self.home
        return self.home

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("CAPSTONE_HOME", None)
        else:
            os.environ["CAPSTONE_HOME"] = self.old
        shutil.rmtree(self.home, ignore_errors=True)
