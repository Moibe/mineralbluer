"""The catalog: every cosplayer ever read off a video, accumulated across runs.

SQLite in a file, written only here. The pipeline produces both the rows and the JPGs they point
at, so keeping the database next to the media (one owner, one directory) is what makes a backup or
a move a single `storage/` copy.
"""

import json
import re
import sqlite3
import time
from contextlib import contextmanager

from config import DB_PATH, ensure_dirs

SCHEMA_VERSION = 2


@contextmanager
def connect():
    """A fresh connection per call, committed and closed on the way out.

    The analysis runs on a worker thread while the API answers on another, and sqlite3 objects are
    not shareable across threads. Per-call connections sidestep that entirely; WAL plus a busy
    timeout is what makes the concurrent readers cheap.

    This is a context manager of its own rather than a bare `sqlite3.connect()` because sqlite3's
    own `with conn:` only wraps the transaction -- it never closes the connection -- so every
    request would leak a file handle until the process ran out.
    """
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init():
    with connect() as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= SCHEMA_VERSION:
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS videos (
                id            TEXT PRIMARY KEY,
                url           TEXT NOT NULL,
                title         TEXT,
                duration      REAL,
                height        INTEGER,
                analysed_from REAL,
                analysed_to   REAL,
                last_job_id   TEXT,
                analysed_at   REAL
            );

            CREATE TABLE IF NOT EXISTS cosplays (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id    TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                "character" TEXT,
                series      TEXT,
                twitter     TEXT,
                handles     TEXT,
                ts_start    REAL NOT NULL,
                ts_end      REAL NOT NULL,
                photo       TEXT,
                crop        TEXT,
                frames      TEXT,
                confidence  REAL,
                readings    INTEGER,
                raw_lines   TEXT,
                verified    INTEGER NOT NULL DEFAULT 0,
                deleted     INTEGER NOT NULL DEFAULT 0,
                created_at  REAL,
                updated_at  REAL
            );

            CREATE INDEX IF NOT EXISTS idx_cosplays_video   ON cosplays(video_id);
            CREATE INDEX IF NOT EXISTS idx_cosplays_twitter ON cosplays(twitter);
            CREATE INDEX IF NOT EXISTS idx_cosplays_series  ON cosplays(series);
            """
        )
        # v1 catalogs stored a single twitter handle and predate `handles`; CREATE TABLE IF NOT
        # EXISTS leaves an existing table alone, so the column has to be added explicitly.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(cosplays)")}
        if "handles" not in columns:
            conn.execute("ALTER TABLE cosplays ADD COLUMN handles TEXT")
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def upsert_video(video):
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO videos (id, url, title, duration, height, analysed_from, analysed_to,
                                last_job_id, analysed_at)
            VALUES (:id, :url, :title, :duration, :height, :analysed_from, :analysed_to,
                    :last_job_id, :analysed_at)
            ON CONFLICT(id) DO UPDATE SET
                url = excluded.url,
                title = COALESCE(excluded.title, videos.title),
                duration = COALESCE(excluded.duration, videos.duration),
                height = excluded.height,
                analysed_from = excluded.analysed_from,
                analysed_to = excluded.analysed_to,
                last_job_id = excluded.last_job_id,
                analysed_at = excluded.analysed_at
            """,
            {**video, "analysed_at": time.time()},
        )


def replace_video_cosplays(video_id, entries):
    """Re-analysing a video replaces its entries -- except the ones you touched by hand.

    A rerun happens for boring reasons (a better quality pass, a wider time range), and it would be
    hostile for it to wipe a correction or resurrect an entry that was discarded on purpose. Rows
    that are verified or deleted are therefore treated as decisions already made: they survive, and
    any freshly read segment landing on the same stretch of video is dropped as a duplicate of one.

    Returns (inserted, kept).
    """
    now = time.time()
    with connect() as conn:
        protected = conn.execute(
            "SELECT ts_start, ts_end FROM cosplays "
            "WHERE video_id = ? AND (verified = 1 OR deleted = 1)",
            (video_id,),
        ).fetchall()
        conn.execute(
            "DELETE FROM cosplays WHERE video_id = ? AND verified = 0 AND deleted = 0",
            (video_id,),
        )

        inserted = 0
        for entry in entries:
            overlaps = any(
                entry["ts_start"] <= row["ts_end"] and entry["ts_end"] >= row["ts_start"]
                for row in protected
            )
            if overlaps:
                continue
            conn.execute(
                """
                INSERT INTO cosplays (video_id, "character", series, twitter, handles,
                                      ts_start, ts_end, photo, crop, frames, confidence,
                                      readings, raw_lines, created_at, updated_at)
                VALUES (:video_id, :character, :series, :twitter, :handles,
                        :ts_start, :ts_end, :photo, :crop, :frames, :confidence,
                        :readings, :raw_lines, :created_at, :updated_at)
                """,
                {
                    "video_id": video_id,
                    "character": entry.get("character"),
                    "series": entry.get("series"),
                    "twitter": entry.get("twitter"),
                    "handles": json.dumps(entry.get("socials") or [], ensure_ascii=False),
                    "ts_start": entry["ts_start"],
                    "ts_end": entry["ts_end"],
                    "photo": entry.get("photo"),
                    "crop": entry.get("crop"),
                    "frames": json.dumps(entry.get("frames") or []),
                    "confidence": entry.get("confidence"),
                    "readings": entry.get("readings"),
                    "raw_lines": json.dumps(entry.get("raw_lines") or [], ensure_ascii=False),
                    "created_at": now,
                    "updated_at": now,
                },
            )
            inserted += 1
        return inserted, len(protected)


def _row_to_cosplay(row):
    item = dict(row)
    item["frames"] = json.loads(item.get("frames") or "[]")
    item["raw_lines"] = json.loads(item.get("raw_lines") or "[]")
    item["handles"] = json.loads(item.get("handles") or "[]")
    item["verified"] = bool(item["verified"])
    item["deleted"] = bool(item["deleted"])
    return item


def list_cosplays(
    q=None,
    series=None,
    video_id=None,
    verified=None,
    include_deleted=False,
    limit=60,
    offset=0,
    order="recent",
):
    where = []
    params = {}
    if not include_deleted:
        where.append("c.deleted = 0")
    if q:
        where.append(
            # handles is searched as raw JSON so looking up an instagram or twitch name works too,
            # not just whichever account was picked as the twitter one.
            '(c."character" LIKE :q OR c.series LIKE :q OR c.twitter LIKE :q '
            'OR c.handles LIKE :q OR v.title LIKE :q)'
        )
        params["q"] = f"%{q}%"
    if series:
        # Compared with spaces stripped, to match how list_series() collapses the facets: picking
        # "MARVEL RIVALS" in the filter has to return the entries stored as "MARVELRIVALS" too.
        where.append("REPLACE(UPPER(c.series), ' ', '') = REPLACE(UPPER(:series), ' ', '')")
        params["series"] = series
    if video_id:
        where.append("c.video_id = :video_id")
        params["video_id"] = video_id
    if verified is not None:
        where.append("c.verified = :verified")
        params["verified"] = 1 if verified else 0

    clause = f"WHERE {' AND '.join(where)}" if where else ""
    orders = {
        "recent": "c.created_at DESC, c.ts_start DESC",
        "timeline": "c.video_id, c.ts_start",
        # Lowest confidence first is the review queue: it puts whatever OCR was least sure about
        # in front of the person who can fix it.
        "confidence": "c.confidence ASC",
        "character": 'c."character" COLLATE NOCASE',
    }
    order_by = orders.get(order, orders["recent"])

    with connect() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM cosplays c JOIN videos v ON v.id = c.video_id {clause}", params
        ).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT c.*, v.title AS video_title, v.url AS video_url
            FROM cosplays c JOIN videos v ON v.id = c.video_id
            {clause}
            ORDER BY {order_by}
            LIMIT :limit OFFSET :offset
            """,
            {**params, "limit": limit, "offset": offset},
        ).fetchall()
    return [_row_to_cosplay(r) for r in rows], total


def get_cosplay(cosplay_id):
    with connect() as conn:
        row = conn.execute(
            """
            SELECT c.*, v.title AS video_title, v.url AS video_url
            FROM cosplays c JOIN videos v ON v.id = c.video_id WHERE c.id = ?
            """,
            (cosplay_id,),
        ).fetchone()
    return _row_to_cosplay(row) if row else None


def update_cosplay(cosplay_id, fields):
    """Editing the text marks the row verified -- that flag is what protects it from a re-analysis.

    Promoting a different frame does NOT: "verified" means a human read the name, series and handle
    and they are right, and picking a nicer photo says nothing about any of that. Letting a photo
    swap set the flag would quietly pull unchecked rows out of the review queue.
    """
    allowed = {"character", "series", "twitter", "photo", "crop", "verified", "deleted"}
    sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not sets:
        return get_cosplay(cosplay_id)
    if any(k in sets for k in ("character", "series", "twitter")):
        sets.setdefault("verified", True)
    sets = {k: (int(v) if isinstance(v, bool) else v) for k, v in sets.items()}

    assignments = ", ".join(f'"{k}" = :{k}' for k in sets)
    with connect() as conn:
        conn.execute(
            f"UPDATE cosplays SET {assignments}, updated_at = :updated_at WHERE id = :id",
            {**sets, "id": cosplay_id, "updated_at": time.time()},
        )
    return get_cosplay(cosplay_id)


def delete_cosplay(cosplay_id, hard=False):
    """Soft by default: a discarded entry has to be remembered, or the next re-analysis of that
    video would read the same bad caption again and put it right back."""
    with connect() as conn:
        if hard:
            cur = conn.execute("DELETE FROM cosplays WHERE id = ?", (cosplay_id,))
        else:
            cur = conn.execute(
                "UPDATE cosplays SET deleted = 1, updated_at = ? WHERE id = ?",
                (time.time(), cosplay_id),
            )
        return cur.rowcount > 0


def list_series():
    """Series facets, collapsing the spacing variants OCR produces.

    "MARVEL RIVALS" and "MARVELRIVALS" are the same show, and listing them separately splits the
    filter so neither option shows all the entries. They are counted together under whichever
    spelling was read most often, with a spaced reading winning ties.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT series, COUNT(*) AS count FROM cosplays "
            "WHERE deleted = 0 AND series IS NOT NULL AND series <> '' "
            "GROUP BY series"
        ).fetchall()

    buckets = {}
    for row in rows:
        key = re.sub(r"[^A-Z0-9]", "", row["series"].upper())
        bucket = buckets.setdefault(key, {"series": row["series"], "count": 0, "_rank": (-1, -1)})
        bucket["count"] += row["count"]
        rank = (row["count"], row["series"].count(" "))
        if rank > bucket["_rank"]:
            bucket["_rank"], bucket["series"] = rank, row["series"]

    facets = [{"series": b["series"], "count": b["count"]} for b in buckets.values()]
    facets.sort(key=lambda f: (-f["count"], f["series"]))
    return facets


def list_videos():
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT v.*, COUNT(c.id) AS cosplays
            FROM videos v LEFT JOIN cosplays c ON c.video_id = v.id AND c.deleted = 0
            GROUP BY v.id ORDER BY v.analysed_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def stats():
    with connect() as conn:
        row = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM cosplays WHERE deleted = 0)                    AS cosplays,
                (SELECT COUNT(*) FROM cosplays WHERE deleted = 0 AND verified = 1)   AS verified,
                -- Spacing-insensitive, exactly like list_series(): otherwise the header counts a
                -- series twice that the filter below it shows as one.
                (SELECT COUNT(DISTINCT REPLACE(UPPER(series), ' ', '')) FROM cosplays
                    WHERE deleted = 0 AND series IS NOT NULL AND series <> '')       AS series,
                -- Keyed on the accounts, falling back to the raw handle list: most entries now
                -- carry a twitch/instagram instead of a twitter, and counting only the twitter
                -- column reported fewer cosplayers than the catalog actually holds.
                (SELECT COUNT(DISTINCT COALESCE(twitter, handles)) FROM cosplays
                    WHERE deleted = 0 AND (COALESCE(twitter, '') <> ''
                        OR COALESCE(handles, '[]') <> '[]'))                         AS cosplayers,
                (SELECT COUNT(*) FROM videos)                                        AS videos
            """
        ).fetchone()
    return dict(row)
