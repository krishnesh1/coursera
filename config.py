"""Paths used by the project.  Tests set CAPSTONE_HOME to a temp folder so they
never touch the production models or logs."""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_paths(test=False):
    home = os.environ.get("CAPSTONE_HOME", PROJECT_DIR)
    prefix = "test_" if test else ""
    return {
        "model_dir": os.path.join(home, prefix + "models"),
        "log_dir": os.path.join(home, prefix + "logs"),
        "data_dir": os.environ.get("DATA_DIR", os.path.join(PROJECT_DIR, "data", "cs-train")),
    }
