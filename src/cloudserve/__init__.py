"""CloudServe ticket resolution workflow (src layout; install with `pip install -e . --no-deps`)."""
from pathlib import Path

# Repository root (src/cloudserve/__init__.py -> parents[2]). Default data, config, prompt and
# .env paths resolve from here, so commands work from any directory in an editable install.
ROOT = Path(__file__).resolve().parents[2]
