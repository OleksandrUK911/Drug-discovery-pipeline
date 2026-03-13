# conftest.py — pytest root configuration
import sys, os

# Make sure `src/` is always on the Python path when running tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
