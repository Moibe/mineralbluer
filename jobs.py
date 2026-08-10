import json
import os
import threading
import time
import traceback
import uuid

import errors
from config import JOBS_DIR, ensure_dirs
from pipeline import DEFAULT_QUALITY, PipelineCancelled, run_pipeline

JOBS = {}
_lock = threading.Lock()


def _job_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def _persist(job: dict):
    """Jobs outlive the process they were started in: an analysis can run for many minutes and the
    user may reload (or the server may restart) in the meantime. Without this, a reloaded page polls
    a job id the server has never heard of and waits forever."""
    ensure_dirs()
    try:
        with open(_job_path(job["id"]), "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False)
    except OSError:
        pass  # persistence is a convenience; never fail the analysis over it


def create_job(
    url: str,
    start_seconds=None,
    end_seconds=None,
    force_redownload=False,
    quality=DEFAULT_QUALITY,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "url": url,
        "created_at": time.time(),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "quality": quality,
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "stage_progress": 0.0,
        "result": None,
        "error": None,
        "error_code": None,
        "retryable": False,
        "cancelled": False,
    }
    with _lock:
        JOBS[job_id] = job
    _persist(job)
    threading.Thread(
        target=_run_job,
        args=(job_id, url, start_seconds, end_seconds, force_redownload, quality),
        daemon=True,
    ).start()
    return job_id


def get_job(job_id: str):
    with _lock:
        job = JOBS.get(job_id)
    if job is not None:
        return job

    # Not in memory: either this process restarted, or the job predates it. A persisted job that
    # never reached a terminal state was interrupted by that restart -- report it as such rather
    # than leaving the client polling something nothing is working on.
    try:
        with open(_job_path(job_id), encoding="utf-8") as f:
            job = json.load(f)
    except (OSError, ValueError):
        return None

    if job.get("status") in ("queued", "processing"):
        job["status"] = "error"
        job["error"] = (
            "El análisis se interrumpió porque el servidor se reinició. "
            "Vuelve a iniciarlo para continuar."
        )
        job["error_code"] = "interrupted"
        job["retryable"] = True
    return job


def list_active():
    """Jobs still running in this process. Lets a fresh page (or a different browser, where no
    local record exists) discover that an analysis is already underway instead of showing nothing."""
    with _lock:
        active = [j for j in JOBS.values() if j["status"] in ("queued", "processing")]
        return [
            {
                "id": j["id"],
                "url": j["url"],
                "stage": j["stage"],
                "progress": j["progress"],
                "stage_progress": j.get("stage_progress", 0.0),
                "start_seconds": j.get("start_seconds"),
                "end_seconds": j.get("end_seconds"),
                "quality": j.get("quality", DEFAULT_QUALITY),
            }
            for j in active
        ]


def list_history(limit=50):
    """Every job ever run, newest first. Reads straight from disk rather than the in-memory JOBS
    dict, since that's what actually survives across restarts."""
    if not os.path.isdir(JOBS_DIR):
        return []

    items = []
    for filename in os.listdir(JOBS_DIR):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(JOBS_DIR, filename), encoding="utf-8") as f:
                job = json.load(f)
        except (OSError, ValueError):
            continue

        result = job.get("result") or {}
        video = result.get("video") or {}
        items.append(
            {
                "id": job["id"],
                "url": job.get("url"),
                "status": job.get("status"),
                "created_at": job.get("created_at"),
                "title": video.get("title"),
                "video_id": video.get("id"),
                "start_seconds": job.get("start_seconds"),
                "end_seconds": job.get("end_seconds"),
                "quality": job.get("quality", DEFAULT_QUALITY),
                "height": video.get("height"),
                "degraded": video.get("degraded", False),
                "cosplays": len(result["cosplays"]) if "cosplays" in result else None,
                "error": job.get("error"),
            }
        )

    items.sort(key=lambda j: j.get("created_at") or 0, reverse=True)
    return items[:limit]


def cancel_job(job_id: str) -> bool:
    with _lock:
        if job_id not in JOBS:
            return False
        JOBS[job_id]["cancelled"] = True
        return True


def _run_job(
    job_id: str,
    url: str,
    start_seconds=None,
    end_seconds=None,
    force_redownload=False,
    quality=DEFAULT_QUALITY,
):
    def on_progress(stage, fraction, stage_fraction=None):
        with _lock:
            JOBS[job_id]["status"] = "processing"
            JOBS[job_id]["stage"] = stage
            JOBS[job_id]["progress"] = fraction
            if stage_fraction is not None:
                JOBS[job_id]["stage_progress"] = stage_fraction

    def should_cancel():
        with _lock:
            return JOBS[job_id]["cancelled"]

    try:
        result = run_pipeline(
            job_id,
            url,
            progress_cb=on_progress,
            should_cancel=should_cancel,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            force_redownload=force_redownload,
            quality=quality,
        )
        with _lock:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["stage"] = "done"
            JOBS[job_id]["progress"] = 1.0
            JOBS[job_id]["result"] = result
            snapshot = dict(JOBS[job_id])
    except PipelineCancelled:
        with _lock:
            JOBS[job_id]["status"] = "cancelled"
            JOBS[job_id]["stage"] = "cancelled"
            snapshot = dict(JOBS[job_id])
    except Exception as exc:
        # Full detail stays in the server log for debugging; the client gets the friendly version.
        traceback.print_exc()
        friendly = errors.classify(exc)
        with _lock:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = friendly["message"]
            JOBS[job_id]["error_code"] = friendly["code"]
            JOBS[job_id]["retryable"] = friendly["retryable"]
            snapshot = dict(JOBS[job_id])

    _persist(snapshot)
