"""Sanity check for rescore --jitter-s: give every input the SAME delay (5 s). A common
delay is a pure time shift of the whole drive, so the score must equal the fitted one."""
import sys, runpy
import numpy as np
sys.path.insert(0, "/home/user/CCFD/fit")
sys.argv = ["rescore.py", "--tag", "nobp_sm6_s3_box", "--seconds", "577", "--draws", "2",
            "--workers", "1", "--jitter-s", "5", "--seed", "777"]
_orig = np.random.default_rng
class _Const:
    def normal(self, mu, sd, n):
        return np.full(n, 5.0)
def _rng(seed=None, *a, **k):
    return _Const() if seed == 777 else _orig(seed, *a, **k)
np.random.default_rng = _rng
runpy.run_path("/home/user/CCFD/fit/rescore.py", run_name="__main__")
