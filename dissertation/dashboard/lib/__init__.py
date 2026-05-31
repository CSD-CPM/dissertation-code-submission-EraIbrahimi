"""Dashboard support package. Importing it puts the dissertation root on
sys.path so `from src import config` resolves from any page."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
