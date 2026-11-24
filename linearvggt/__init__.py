"""LinearVGGT: a linear-time online visual geometry transformer with a hybrid architecture."""
import sys
from pathlib import Path

__version__ = "0.1.0"

# The StreamVGGT / VGGT code that LinearVGGT builds on is in third_party/ (packages `streamvggt` and
# `dust3r_utils`). `pip install -e .` installs it; without an installation, a checkout finds it here.
_third_party = Path(__file__).resolve().parents[1] / "third_party"
if _third_party.is_dir() and str(_third_party) not in sys.path:
    sys.path.insert(0, str(_third_party))

_EXPORTS = {
    "LinearVGGT": "linearvggt.models.linearvggt",
    "from_pretrained": "linearvggt.models.linearvggt",
    "load_student": "linearvggt.models.linearvggt",
    "build_student": "linearvggt.models.linearvggt",
    "StreamVGGT": "streamvggt.models.streamvggt",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):  # lazy: importing the package does not load Mamba2 / Triton
    if name in _EXPORTS:
        from importlib import import_module
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'linearvggt' has no attribute {name!r}")
