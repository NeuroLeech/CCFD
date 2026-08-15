"""Every path the code uses, derived from this file's own location.

Nothing outside this folder is read or written. Moving or renaming the folder
does not break anything, and there is no dependency on nilearn's download cache
or on any absolute path under a home directory.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

SURF = os.path.join(DATA, "surf")          # fsaverage5/6 inflated left surfaces
ANNOT = os.path.join(DATA, "annot", "lh.HCP-MMP1.annot")
MSC = os.path.join(DATA, "msc")            # MSC subject-01 resting state, CIFTI-1
CACHE = os.path.join(DATA, "cache")        # anything derived and reusable

RESULTS = os.path.join(ROOT, "results")    # GA histories, measure tables
FIELDS = os.path.join(RESULTS, "fields")   # vertex x time simulation output
VIDEOS = os.path.join(RESULTS, "videos")

for _d in (CACHE, RESULTS, FIELDS, VIDEOS):
    os.makedirs(_d, exist_ok=True)
