"""Put the repository's code folders on sys.path.

The tree is folders-only by design: there is no package and no install step, and every
script is still run as a plain file - `python fit/best_fit.py` - importing what it needs
by its plain name, `import xspec`. What that costs is this shim. Python puts only the
SCRIPT'S OWN directory on sys.path, so `import xspec` from analysis/ would not find
fit/xspec.py without help.

One copy sits in each code folder, so `import _path` resolves from whichever folder the
running script lives in. It is idempotent and inserts nothing already present.
"""
import os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (os.path.join(_ROOT, _g) for _g in
           ("core", "targets", "fit", "analysis", "viz")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
