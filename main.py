from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
import jobs
from config import MEDIA_DIR, ensure_dirs
from pipeline import DEFAULT_QUALITY, QUALITY_PRESETS

app = FastAPI(title="mineralbluer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:7979",
        "http://127.0.0.1:7979",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_dirs()
db.init()
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")


class JobRequest(BaseModel):
    url: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    # Escape hatch for a cached download that turned out low-res (e.g. a rate limit forced the
    # 360p fallback) -- force a fresh attempt instead of silently reusing the bad copy.
    force_redownload: bool = False
    quality: str = DEFAULT_QUALITY


class CosplayUpdate(BaseModel):
    character: str | None = None
    series: str | None = None
    twitter: str | None = None
    # Lets the UI promote one of the alternative frames to be the entry's photo.
    photo: str | None = None
    verified: bool | None = None
    deleted: bool | None = None


# ---------------------------------------------------------------- analysis jobs


@app.post("/jobs")
def create_job(payload: JobRequest):
    if (
        payload.start_seconds is not None
        and payload.end_seconds is not None
        and payload.end_seconds <= payload.start_seconds
    ):
        raise HTTPException(status_code=400, detail="end_seconds must be greater than start_seconds")
    if payload.quality not in QUALITY_PRESETS:
        raise HTTPException(
            status_code=400, detail=f"quality must be one of {sorted(QUALITY_PRESETS)}"
        )
    job_id = jobs.create_job(
        payload.url,
        payload.start_seconds,
        payload.end_seconds,
        payload.force_redownload,
        payload.quality,
    )
    return {"job_id": job_id}


@app.get("/jobs")
def list_active_jobs():
    return {"active": jobs.list_active()}


# Must come before /jobs/{job_id}: FastAPI matches path routes in declaration order, so a
# variable segment declared first would swallow "history" as if it were a job id.
@app.get("/jobs/history")
def get_history():
    return {"jobs": jobs.list_history()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not jobs.cancel_job(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    return {"status": "cancelling"}


# ---------------------------------------------------------------- catalog


@app.get("/cosplays")
def list_cosplays(
    q: str | None = None,
    series: str | None = None,
    video_id: str | None = None,
    verified: bool | None = None,
    include_deleted: bool = False,
    order: str = "recent",
    limit: int = Query(60, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    items, total = db.list_cosplays(
        q=q,
        series=series,
        video_id=video_id,
        verified=verified,
        include_deleted=include_deleted,
        order=order,
        limit=limit,
        offset=offset,
    )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@app.get("/cosplays/{cosplay_id}")
def get_cosplay(cosplay_id: int):
    item = db.get_cosplay(cosplay_id)
    if item is None:
        raise HTTPException(status_code=404, detail="cosplay not found")
    return item


@app.patch("/cosplays/{cosplay_id}")
def update_cosplay(cosplay_id: int, payload: CosplayUpdate):
    existing = db.get_cosplay(cosplay_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="cosplay not found")
    fields = payload.model_dump(exclude_unset=True)
    # A promoted frame has to be one of the frames actually saved for this entry; anything else
    # would let a request point the catalog at an arbitrary path under /media.
    if fields.get("photo") is not None and fields["photo"] not in existing["frames"]:
        raise HTTPException(status_code=400, detail="photo must be one of the entry's frames")
    return db.update_cosplay(cosplay_id, fields)


@app.delete("/cosplays/{cosplay_id}")
def delete_cosplay(cosplay_id: int, hard: bool = False):
    if not db.delete_cosplay(cosplay_id, hard=hard):
        raise HTTPException(status_code=404, detail="cosplay not found")
    return {"status": "deleted", "hard": hard}


@app.get("/series")
def list_series():
    return {"series": db.list_series()}


@app.get("/videos")
def list_videos():
    return {"videos": db.list_videos()}


@app.get("/stats")
def get_stats():
    return db.stats()


@app.get("/health")
def health():
    return {"status": "ok"}
