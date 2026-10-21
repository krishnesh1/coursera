"""Runs every unit test (API, model, logging) with one command:  python run_tests.py
Exit code 0 means all tests passed."""
import os, sys, unittest

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    suite = unittest.defaultTestLoader.discover(os.path.join(here, "unittests"), pattern="test_*.py",
                                                top_level_dir=here)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
