"""Where everything lives on disk.

Kept in its own module so db.py, pipeline.py and jobs.py can share the paths without importing
each other -- the catalog and the pipeline are independent halves that only meet in main.py.
"""

import os

STORAGE_ROOT = os.environ.get(
    "MINERALBLUER_STORAGE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage"),
)

# Photos of each cosplayer, served over HTTP at /media. One folder per (video, segment).
MEDIA_DIR = os.path.join(STORAGE_ROOT, "media")
# Downloads are cached by (video, segment, quality), not by job: re-analysing the same clip after
# tuning a threshold is the normal case, and every repeat download is minutes and hundreds of MB.
VIDEO_CACHE_DIR = os.path.join(STORAGE_ROOT, "video_cache")
# Job snapshots, so an analysis survives a page reload or a server restart.
JOBS_DIR = os.path.join(STORAGE_ROOT, "jobs")

DB_PATH = os.environ.get("MINERALBLUER_DB", os.path.join(STORAGE_ROOT, "catalog.db"))


def ensure_dirs():
    for path in (STORAGE_ROOT, MEDIA_DIR, VIDEO_CACHE_DIR, JOBS_DIR):
        os.makedirs(path, exist_ok=True)
