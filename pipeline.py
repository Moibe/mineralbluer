"""Turn a MineralBlu convention video into catalog rows: photo + character + series + twitter.

Shape of the run: download the (optionally clipped) video once, walk it at ~1 sampled frame per
second reading the burnt-in caption, group consecutive frames that show the same caption into one
cosplayer, then keep the best-looking frames of each group as that cosplayer's photos.
"""

import glob
import json
import os
import re

import cv2
import numpy as np
import yt_dlp

import db
import errors
import overlay
from config import MEDIA_DIR, VIDEO_CACHE_DIR, ensure_dirs

TARGET_SAMPLE_FPS = 1.0
# Hard cap on sampled frames regardless of length. OCR is what this costs -- measured at ~3.4s per
# frame on 1440p convention footage -- so this cap is really a ~70 minute ceiling on a run. Past it
# the sampling rate drops proportionally rather than the video being truncated. Captions stay on
# screen around four seconds, which is what makes a sub-1fps rate still land a couple of readings
# on each one; below roughly 0.4fps they start being missed entirely.
MAX_SAMPLES = 1200
# A caption read in a single frame is far more likely to be a fluke (a banner that happened to
# parse, one frame of motion blur) than a real cosplayer -- the channel holds each label on screen
# long enough that a genuine one is always seen more than once.
MIN_READINGS = 2
MAX_GAP_SECONDS = 3.0
# Candidate frames held in memory per segment, JPEG-encoded (~300KB each at 1080p). Only one
# segment is ever open at a time, so this is the whole frame budget of the run.
MAX_CANDIDATES = 10
FRAMES_KEPT = 3
# Laplacian variance a frame needs to count as fully sharp. Convention footage is handheld, so a
# lot of frames of the same cosplayer differ mostly in how blurred they are.
SHARPNESS_REFERENCE = 120.0

_face_app = None


class PipelineCancelled(Exception):
    pass


def get_face_app():
    """Face detection only -- no recognition model is loaded.

    Faces are used here to answer "which frame of this segment is the best photo, and where is the
    person in it", not "who is this": the identity already came off the caption. Skipping the
    recognition model halves the load time and the memory of the model bundle.
    """
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis

        print("[framing] loading face detector...", flush=True)
        _face_app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=-1, det_size=(640, 640))
        print("[framing] face detector ready", flush=True)
    return _face_app


def extract_video_id(url):
    match = re.search(
        r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/))([a-zA-Z0-9_-]{11})", url
    )
    return match.group(1) if match else None


def _cache_key(video_id, start_seconds, end_seconds, quality):
    start_part = "full" if start_seconds is None else int(start_seconds)
    end_part = "full" if end_seconds is None else int(end_seconds)
    return f"{video_id}_{start_part}_{end_part}_{quality}"


# av01 is excluded outright, not just capped by resolution: some av01 variants are unreadable by
# this OpenCV build (cap.read() blocks forever) while the vp9 variant of the same tier decodes
# fine. "auto" climbs as high as whatever is available short of 4k; "4k" is the explicit opt-in,
# worth it only when the caption is small on screen, since it costs ~4x the download.
QUALITY_PRESETS = {
    "auto": "bestvideo[height<2160][vcodec!^=av01]/bestvideo[height<2160]/best[height<2160]/best",
    "4k": "bestvideo[height<=2160][vcodec!^=av01]/bestvideo[height<=2160]/best[height<=2160]/best",
}
DEFAULT_QUALITY = "auto"


def download_video(
    url,
    job_id,
    start_seconds=None,
    end_seconds=None,
    progress_cb=None,
    force_redownload=False,
    quality=DEFAULT_QUALITY,
):
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"unknown quality preset: {quality!r}")

    ensure_dirs()
    cache_key = _cache_key(extract_video_id(url) or job_id, start_seconds, end_seconds, quality)
    meta_path = os.path.join(VIDEO_CACHE_DIR, f"{cache_key}.json")

    if not force_redownload and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if os.path.exists(meta["path"]):
            print(f"[downloading] usando copia ya descargada ({meta['info'].get('height')}p)", flush=True)
            if progress_cb:
                progress_cb(1.0)
            return meta["path"], meta["info"]
        # Metadata survived but the file didn't (storage cleared by hand); fall through and redo it.

    outtmpl = os.path.join(VIDEO_CACHE_DIR, f"{cache_key}.%(ext)s")

    def on_ydl_progress(d):
        # Downloads run for minutes; without this the bar sits at 0% and a slow download is
        # indistinguishable from a stuck one.
        if not progress_cb or d.get("status") != "downloading":
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        done = d.get("downloaded_bytes") or 0
        if total:
            progress_cb(min(0.99, done / total))

    ydl_opts = {
        "progress_hooks": [on_ydl_progress],
        # Resolution is what decides whether the caption is readable at all: the label is a thin
        # uppercase font maybe 30px tall at 1080p, which is right at the edge of what PP-OCR reads.
        # At 480p the same text is ~13px and comes back as garbage.
        "format": QUALITY_PRESETS[quality],
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # A download that stops receiving data must fail rather than sit there: without these a
        # stalled connection keeps the job "downloading" forever, which is indistinguishable from
        # a slow one and blocks the worker thread with no way out but killing the server.
        "socket_timeout": 30,
        "retries": 2,
        "fragment_retries": 2,
        # Without the EJS challenge solver YouTube only exposes the 360p progressive format --
        # every higher-resolution URL stays obfuscated behind an "n challenge".
        "remote_components": ["ejs:github"],
    }

    # Cookies are what make the default client work, and the default client is the only one that
    # is both high-resolution AND fast (measured: ~1.3MB/s vs ~32KB/s on the embedded client).
    # Two ways in, because reading them live from Chrome fails whenever Chrome is running -- it
    # holds a lock on its cookie DB -- and that is the normal state of the machine.
    cookies_file = os.environ.get("MINERALBLUER_COOKIES_FILE")
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts["cookiefile"] = cookies_file
    cookies_browser = os.environ.get("MINERALBLUER_COOKIES_FROM_BROWSER")
    if cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser,)

    if start_seconds is not None or end_seconds is not None:
        start = float(start_seconds or 0)
        end = float(end_seconds) if end_seconds is not None else None
        ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(
            None, [(start, end if end is not None else float("inf"))]
        )
        ydl_opts["force_keyframes_at_cuts"] = True

    # Ordered by resulting quality, not by reliability.
    # The default client gets two shots because clipped downloads go through ffmpeg, which fails
    # transiently often enough that one crash should not cost the good resolution -- that retry is
    # NOT a quality fallback (is_fallback=False).
    # web_embedded is the one that matters here: when the default client is refused with "sign in
    # to confirm you're not a bot" (routine on this network, no cookies available since a running
    # Chrome locks its cookie DB), it still returns the full 1440p/vp9 ladder -- measured on this
    # exact video, minutes apart, while every other client was either bot-checked or capped. That
    # difference decides whether the run works at all: the caption is unreadable at 360p.
    # android/tv dodges the bot check too but is capped low, so it is the last resort that at
    # least returns something, and the only attempt that counts as an actual degradation.
    attempts = [
        ("default", {}, False),
        ("default (reintento)", {}, False),
        (
            "web_embedded",
            {"extractor_args": {"youtube": {"player_client": ["web_embedded"]}}},
            False,
        ),
        (
            "android/tv",
            {"extractor_args": {"youtube": {"player_client": ["android", "tv", "web"]}}},
            True,
        ),
    ]

    last_error = None
    for label, overrides, is_fallback in attempts:
        # A partial file from a crashed attempt can make the next ffmpeg run fail the same way.
        for stale in glob.glob(os.path.join(VIDEO_CACHE_DIR, f"{cache_key}.*")):
            try:
                os.remove(stale)
            except OSError:
                pass

        opts = {**ydl_opts, **overrides}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                path = ydl.prepare_filename(info)
            print(f"[downloading] cliente '{label}' -> {info.get('height')}p", flush=True)
            degraded = is_fallback
            degraded_reason = errors.classify(last_error) if degraded and last_error else None
            info_subset = {
                "title": info.get("title"),
                "duration": info.get("duration"),
                "height": info.get("height"),
                "degraded": degraded,
                "degraded_reason": degraded_reason,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"path": path, "info": info_subset}, f)
            return path, info_subset
        except Exception as exc:
            print(f"[downloading] cliente '{label}' fallo: {str(exc)[:140]}", flush=True)
            last_error = exc

    raise last_error


def iter_sampled_frames(video_path, target_fps=TARGET_SAMPLE_FPS):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / target_fps))
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                yield frame_idx / fps, frame, frame_idx, total_frames
            frame_idx += 1
    finally:
        cap.release()


def compute_sample_fps(duration_seconds, max_samples=MAX_SAMPLES, default_fps=TARGET_SAMPLE_FPS):
    if not duration_seconds or duration_seconds <= 0:
        return default_fps
    if duration_seconds * default_fps <= max_samples:
        return default_fps
    return max_samples / duration_seconds


def _sharpness(frame):
    small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
    return cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def score_frame(frame):
    """How good a photo this frame is: a big, confidently detected, sharp face.

    Returns (score, face_bbox|None). A frame with no detectable face still scores -- some captions
    land on a shot from behind or a full-body pose -- just far below one where the cosplayer's
    face is clearly visible.
    """
    sharp_term = min(1.0, _sharpness(frame) / SHARPNESS_REFERENCE)
    faces = get_face_app().get(frame)
    if not faces:
        return 0.25 * sharp_term, None

    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    face_height = float(best.bbox[3] - best.bbox[1])
    # Square-rooted so the score keeps separating small faces from medium ones instead of
    # saturating the moment anyone is close to the camera.
    face_term = min(1.0, (face_height / frame.shape[0]) ** 0.5 * 1.8)
    score = (0.65 * face_term + 0.35 * sharp_term) * float(best.det_score)
    return score, [float(v) for v in best.bbox]


def portrait_crop(frame, bbox, aspect=0.62):
    """Full-body-ish vertical crop built out from the face box.

    There is no person detector in the stack, but a face is a reliable ruler: a standing adult is
    roughly seven head-heights tall, so the face box alone is enough to frame the cosplay (which is
    the point -- the costume is below the face, and a face crop would cut off everything worth
    seeing). Falls back to a centred crop when no face was found.
    """
    h, w = frame.shape[:2]
    if bbox is None:
        crop_h = h
        crop_w = min(w, int(crop_h * aspect))
        left = (w - crop_w) // 2
        return frame[0:crop_h, left : left + crop_w]

    x1, y1, x2, y2 = bbox
    face_h = max(1.0, y2 - y1)
    body_h = min(float(h), face_h * 7.2)
    top = max(0.0, y1 - 0.45 * face_h)
    if top + body_h > h:
        top = max(0.0, h - body_h)

    crop_w = min(float(w), body_h * aspect)
    center_x = (x1 + x2) / 2
    left = min(max(0.0, center_x - crop_w / 2), w - crop_w)

    return frame[int(top) : int(top + body_h), int(left) : int(left + crop_w)]


def _segment_id(ts_start):
    return f"seg_{int(ts_start):06d}"


def _save_segment_media(video_id, segment):
    """Writes the kept frames + the portrait crop, best photo first. Returns the media paths."""
    scored = []
    for ts, jpeg in segment["candidates"]:
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        score, bbox = score_frame(frame)
        scored.append((score, ts, frame, bbox))
    if not scored:
        return None, None, []

    scored.sort(key=lambda item: item[0], reverse=True)
    keep = scored[:FRAMES_KEPT]

    seg_id = _segment_id(segment["ts_start"])
    out_dir = os.path.join(MEDIA_DIR, video_id, seg_id)
    os.makedirs(out_dir, exist_ok=True)
    # A re-run of the same stretch of video must not leave last run's extra frames behind.
    for stale in glob.glob(os.path.join(out_dir, "*.jpg")):
        try:
            os.remove(stale)
        except OSError:
            pass

    frames = []
    for position, (_score, _ts, frame, _bbox) in enumerate(keep, start=1):
        name = f"frame_{position}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frames.append(f"/media/{video_id}/{seg_id}/{name}")

    _score, _ts, best_frame, best_bbox = keep[0]
    crop = portrait_crop(best_frame, best_bbox)
    crop_path = None
    if crop is not None and crop.size:
        cv2.imwrite(os.path.join(out_dir, "crop.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        crop_path = f"/media/{video_id}/{seg_id}/crop.jpg"

    return frames[0], crop_path, frames


def _open_segment(timestamp, caption, jpeg):
    return {
        "ts_start": timestamp,
        "ts_end": timestamp,
        "observations": [caption],
        # A frame that failed to encode still counts as a reading of the caption -- it just can't
        # be a candidate photo, and passing None down here would blow up at decode time instead.
        "candidates": [(timestamp, jpeg)] if jpeg else [],
    }


def _extend_segment(segment, timestamp, caption, jpeg):
    segment["ts_end"] = timestamp
    segment["observations"].append(caption)
    if jpeg:
        segment["candidates"].append((timestamp, jpeg))
    if len(segment["candidates"]) > MAX_CANDIDATES:
        # Halve by taking every other one rather than dropping the tail: the goal is candidates
        # spread across the whole appearance, not the first ten frames of it.
        segment["candidates"] = segment["candidates"][::2]


def scan_captions(
    video_path,
    progress_cb=None,
    should_cancel=None,
    target_fps=TARGET_SAMPLE_FPS,
    clip_offset=0.0,
    max_gap_seconds=MAX_GAP_SECONDS,
):
    """Walks the video once, returning (segments, stats). Each segment is one caption on screen."""
    segments = []
    current = None
    frames_scanned = 0
    frames_with_caption = 0
    dropped_short = 0

    def close(segment):
        nonlocal dropped_short
        if segment is None:
            return
        if len(segment["observations"]) < MIN_READINGS:
            dropped_short += 1
            return
        segments.append(segment)

    for timestamp, frame, frame_idx, total_frames in iter_sampled_frames(video_path, target_fps):
        if should_cancel and should_cancel():
            raise PipelineCancelled()
        frames_scanned += 1
        ts = timestamp + clip_offset

        caption = overlay.parse_caption(overlay.read_lines(frame))
        if caption:
            frames_with_caption += 1
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            jpeg = encoded.tobytes() if ok else None

            # Compared against both ends of the open segment: one badly misread frame in the middle
            # would otherwise split a single cosplayer in two, since the next frame is compared to
            # the misreading rather than to what the caption actually says.
            continues = current is not None and ts - current["ts_end"] <= max_gap_seconds and (
                overlay.same_caption(current["observations"][-1], caption)
                or overlay.same_caption(current["observations"][0], caption)
            )
            if continues:
                _extend_segment(current, ts, caption, jpeg)
            else:
                close(current)
                current = _open_segment(ts, caption, jpeg)
        elif current is not None and ts - current["ts_end"] > max_gap_seconds:
            # Tolerating a gap matters: someone walking in front of the caption, or a frame where
            # OCR simply missed it, is not the end of that cosplayer's appearance.
            close(current)
            current = None

        if progress_cb and total_frames:
            progress_cb(min(0.98, frame_idx / total_frames))

    close(current)
    return segments, {
        "frames_scanned": frames_scanned,
        "frames_with_caption": frames_with_caption,
        "segments_dropped_short": dropped_short,
    }


def run_pipeline(
    job_id,
    url,
    progress_cb=None,
    should_cancel=None,
    start_seconds=None,
    end_seconds=None,
    force_redownload=False,
    quality=DEFAULT_QUALITY,
):
    def report(stage, fraction, stage_fraction=None):
        # Two numbers: overall progress drives the bar, while the stage's own fraction is what
        # tells a waiting user that a long download or a long OCR pass is actually moving.
        if progress_cb:
            progress_cb(stage, fraction, stage_fraction)

    ensure_dirs()
    db.init()

    report("downloading", 0.0, 0.0)
    video_path, info = download_video(
        url,
        job_id,
        start_seconds,
        end_seconds,
        force_redownload=force_redownload,
        quality=quality,
        progress_cb=lambda f: report("downloading", f * 0.1, f),
    )

    if should_cancel and should_cancel():
        raise PipelineCancelled()

    full_duration = info.get("duration")
    # A clipped download restarts at t=0, so every timestamp must be shifted back onto the original
    # timeline -- otherwise "jump to this moment" links point at the wrong place.
    clip_offset = float(start_seconds or 0)
    if start_seconds is not None or end_seconds is not None:
        clip_end = float(end_seconds) if end_seconds is not None else (full_duration or 0)
        analysed_duration = max(1.0, clip_end - clip_offset)
    else:
        analysed_duration = full_duration

    sample_fps = compute_sample_fps(analysed_duration)
    # Gap tolerance must scale with the actual sampling interval, or a coarser rate (long videos)
    # would fragment one caption into several entries.
    gap_seconds = max(MAX_GAP_SECONDS, (1.0 / sample_fps) * 2.5)
    print(
        f"[reading] sampling at {sample_fps:.3f} fps "
        f"(analysing {analysed_duration}s of {full_duration}s, offset {clip_offset}s)",
        flush=True,
    )

    report("reading", 0.1, 0.0)
    segments, scan_stats = scan_captions(
        video_path,
        progress_cb=lambda f: report("reading", 0.1 + f * 0.75, f),
        should_cancel=should_cancel,
        target_fps=sample_fps,
        clip_offset=clip_offset,
        max_gap_seconds=gap_seconds,
    )
    print(f"[reading] {len(segments)} captions found", flush=True)

    video_id = extract_video_id(url) or job_id
    report("framing", 0.85, 0.0)

    entries = []
    for index, segment in enumerate(segments):
        if should_cancel and should_cancel():
            raise PipelineCancelled()
        consolidated = overlay.consolidate(segment["observations"])
        photo, crop, frames = _save_segment_media(video_id, segment)
        if not photo:
            continue
        entries.append(
            {
                **consolidated,
                "ts_start": segment["ts_start"],
                "ts_end": segment["ts_end"],
                "photo": photo,
                "crop": crop,
                "frames": frames,
            }
        )
        report("framing", 0.85 + 0.1 * (index + 1) / len(segments), (index + 1) / len(segments))

    report("saving", 0.95, 0.0)
    height = info.get("height")
    db.upsert_video(
        {
            "id": video_id,
            "url": url,
            "title": info.get("title"),
            "duration": full_duration,
            "height": height,
            "analysed_from": clip_offset,
            "analysed_to": clip_offset + (analysed_duration or 0),
            "last_job_id": job_id,
        }
    )
    inserted, protected = db.replace_video_cosplays(video_id, entries)
    saved, _total = db.list_cosplays(video_id=video_id, limit=500, order="timeline")

    warnings = []
    degraded = info.get("degraded", False)
    degraded_reason = info.get("degraded_reason")
    if degraded and degraded_reason:
        warnings.append(
            f"Se descargó a {height}p en vez de la calidad pedida ({quality}): "
            f"{degraded_reason['message']}"
        )
    if height and height < 720:
        # The caption is a thin uppercase font; below 720p it is roughly 15px tall and PP-OCR
        # starts returning nonsense rather than nothing, which is the worse failure of the two.
        warnings.append(
            "Por debajo de 720p el texto del overlay queda demasiado pequeño para leerlo con "
            "confianza: es probable que falten cosplayers o que los nombres salgan mal escritos."
        )
    if protected:
        warnings.append(
            f"{protected} entrada(s) de este video ya estaban editadas o descartadas a mano y se "
            "respetaron tal cual."
        )
    for warning in warnings:
        print(f"[warning] {warning}", flush=True)

    report("done", 1.0, 1.0)
    return {
        "video": {
            "id": video_id,
            "title": info.get("title"),
            "duration": full_duration,
            "source_url": url,
            "analysed_from": clip_offset,
            "analysed_to": clip_offset + (analysed_duration or 0),
            "height": height,
            "quality_requested": quality,
            "degraded": degraded,
            "degraded_reason": degraded_reason,
        },
        "cosplays": saved,
        "warnings": warnings,
        "stats": {
            **scan_stats,
            "segments": len(segments),
            "saved": inserted,
            "protected": protected,
        },
    }
