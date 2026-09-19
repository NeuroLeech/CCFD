"""What was run, stored inside the thing it produced.

Every writer in this repo saves the parameters it thought were interesting and drops the
rest. `fit/torch_fit.py` recorded ref, seconds, amp and the nonlinearities but never wa,
lag_taus, lr, map_lr, seed or iters - so whether a checkpoint carried the lagged term in
its objective could not be read back off it. `fit/best_fit.py` records more, but nothing
about which code produced it, and the solver changed repeatedly under runs that share a
tag prefix.

`stamp` takes the argparse namespace and returns four keys to splat into any np.savez.
All four are plain strings, which survive npz round-tripping without pickle:

    np.savez(path, S=S, ..., **provenance.stamp(a))

    argv        the command line as issued
    args_json   the WHOLE namespace, including defaults never typed on the line
    git_commit  HEAD, suffixed +dirty when the tree had uncommitted changes
    run_at      local wall clock, ISO, to the second

argv and args_json are both kept because neither covers the other: argv is what was typed
and shows what was deliberate, args_json is what the run actually used and includes every
default. git_commit is what makes the other two mean anything a week later.
"""
import os, sys, json, subprocess, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    """git output, or "" if git is missing, slow, or this is not a checkout."""
    try:
        return subprocess.run(("git",) + args, cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def stamp(args=None):
    """-> dict of four string keys, ready to splat into np.savez.

    `args` is an argparse.Namespace, or anything vars() accepts; None records the command
    line alone. Never raises: a checkpoint that fails to write because provenance failed
    would be worse than one with a blank commit field."""
    head = _git("rev-parse", "HEAD")
    try:
        aj = json.dumps(vars(args), sort_keys=True, default=str) if args is not None \
            else "{}"
    except TypeError:
        aj = "{}"
    return dict(argv=" ".join(sys.argv),
                args_json=aj,
                git_commit=head + ("+dirty" if head and _git("status", "--porcelain")
                                   else ""),
                run_at=datetime.datetime.now().isoformat(timespec="seconds"))


def read(z):
    """Pull the stamp back out of a loaded npz. -> dict, with args parsed.

    Returns {} for a file written before this existed, which is how a reader tells an
    unrecorded run from one that recorded nothing."""
    if "argv" not in z:
        return {}
    out = {k: str(z[k]) for k in ("argv", "git_commit", "run_at") if k in z}
    try:
        out["args"] = json.loads(str(z["args_json"]))
    except (ValueError, KeyError):
        out["args"] = {}
    return out
