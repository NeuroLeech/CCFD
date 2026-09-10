import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("core", "fit", "targets", "analysis", "viz"):
    sys.path.insert(0, os.path.join(ROOT, d))
