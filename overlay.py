"""Read MineralBlu's burnt-in cosplay caption off a video frame.

The channel labels every cosplayer with a left-aligned stack: the character and series in caps,
then one line per social account. Older videos carry a single bare handle, newer ones name the
platform and often list two or three:

    UBEL                    SUE STORM
    FRIEREN                 MARVEL RIVALS
    @tinechan               twitch @alinity
                            ig @alinity_divine

What identifies the caption is its *geometry*: two or more lines sharing a left edge and a font
size, stacked one directly under the next. A convention video is full of incidental text (booth
banners, badges, signage) but it almost never forms a clean left-aligned column like this, so the
stack is the anchor.

An earlier version anchored on the "@" instead, and it lost more than half of the captions: PP-OCR
drops that thin glyph constantly, returning "calebweekss" and "wanderlustluca" with no "@" at all.
Requiring it meant every one of those frames was thrown away.

Within a stack the split is by content, not by position: lines that look like an account go to
`socials`, the rest are the character (first) and the series (the remainder). Reading it
positionally is what made an early version file "twitch @alinity" as part of the series name.
"""

import re
from difflib import SequenceMatcher

import cv2

# Below this the reading is more likely a hallucination on a busy background than real text.
MIN_LINE_SCORE = 0.45
# Fraction of the frame height skipped before OCR. Zero: the caption's vertical position is NOT
# stable. In one 50-second clip the same channel placed captions starting anywhere from 0.22 to
# 0.51 of the frame height, and every crop tried lopped the character line off some of them --
# which costs the most important field on the entry. What separates a caption from booth signage
# here is the shape of the stack (see _find_stack), not where it sits, so the crop was buying
# speed at the price of recall. Raise this only if profiling says OCR time is the problem.
ROI_TOP_FRACTION = 0.0
# OCR runs on a downscaled copy. Benchmarked over 50 frames of a 1440p video: 1280px read 35 of
# them at 3.4s/frame, against 33 at 4.6s/frame for 1600px and 34 at 4.5s/frame for 1024px. So this
# is not a speed-for-accuracy trade -- 1280 simply won both, and 1024 was slower than 1280 because
# PP-OCR resizes internally anyway and a blurrier input yields more recognition candidates.
OCR_MAX_WIDTH = 1280

_HANDLE_RE = re.compile(r"[@©]\s*([A-Za-z0-9_.]{2,30})")
_WATERMARK = "mineralblu"

_engine = None
_engine_api = None


def get_engine():
    """Loads PP-OCR lazily -- importing it costs seconds and a job may fail before ever needing it."""
    global _engine, _engine_api
    if _engine is None:
        print("[reading] loading OCR model...", flush=True)
        try:
            from rapidocr_onnxruntime import RapidOCR

            _engine, _engine_api = RapidOCR(), "v1"
        except ImportError:
            # The project moved from `rapidocr-onnxruntime` to `rapidocr` upstream and the two
            # return different shapes; support both so a fresh install of either one works.
            from rapidocr import RapidOCR

            _engine, _engine_api = RapidOCR(), "v2"
        print(f"[reading] OCR model ready ({_engine_api})", flush=True)
    return _engine, _engine_api


def _raw_detections(image):
    engine, api = get_engine()
    out = engine(image)
    if api == "v1":
        result = out[0] if isinstance(out, tuple) else out
        return [(item[0], item[1], float(item[2])) for item in (result or [])]
    boxes = getattr(out, "boxes", None)
    if boxes is None:
        return []
    return [
        (box, txt, float(score))
        for box, txt, score in zip(boxes, out.txts or [], out.scores or [])
    ]


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.strip(" .,:;|/\\-_'\"“”‘’·•")


def _is_watermark(text: str) -> bool:
    """The channel logo sits in the corner of every frame and must never enter a caption.

    Matched fuzzily because OCR mangles its wide-tracked lettering a different way on each frame --
    MINERALBLL, MINGRALBLO, MINERALBLD and MINERALBUU all showed up in a single 90-second clip, so
    an exact (or even prefix) match lets most readings of it straight through.
    """
    compact = re.sub(r"[^a-z]", "", (text or "").lower())
    if len(compact) < 6:
        return False
    return SequenceMatcher(None, compact[:12], _WATERMARK).ratio() >= 0.7


def _merge_rows(lines):
    """Rejoins one visual line that OCR returned as two boxes.

    "twitch @alinity" routinely comes back as "twitch" and "@alinity" side by side, and left as
    separate lines the platform half looks like a stray word and the handle half loses its
    platform. Boxes that overlap vertically and nearly touch are the same line.
    """
    ordered = sorted(lines, key=lambda ln: (ln["top"], ln["left"]))
    merged = []
    for line in ordered:
        for target in merged:
            h = min(target["height"], line["height"]) or 1.0
            overlap = min(target["bottom"], line["bottom"]) - max(target["top"], line["top"])
            # Measured between the facing edges rather than assuming the new box is to the right:
            # sorting by top does not order a row left to right, because an "@" reaches higher than
            # the lowercase word beside it, so "@alinity" arrives before the "twitch" it follows.
            gap = max(target["left"], line["left"]) - min(target["right"], line["right"])
            if overlap > 0.5 * h and gap <= 1.5 * h:
                left, right = (
                    (line, target) if line["left"] < target["left"] else (target, line)
                )
                target["text"] = f"{left['text']} {right['text']}"
                target["left"] = min(target["left"], line["left"])
                target["right"] = max(target["right"], line["right"])
                target["top"] = min(target["top"], line["top"])
                target["bottom"] = max(target["bottom"], line["bottom"])
                target["height"] = target["bottom"] - target["top"]
                # The weaker half is what the merged line should be trusted as far as.
                target["score"] = min(target["score"], line["score"])
                break
        else:
            merged.append(dict(line))
    return merged


def read_lines(frame):
    """OCRs the caption band of a frame. Returns dicts in ORIGINAL frame coordinates."""
    h, w = frame.shape[:2]
    roi_top = int(h * ROI_TOP_FRACTION)
    roi = frame[roi_top:, :]

    scale = 1.0
    if w > OCR_MAX_WIDTH:
        scale = OCR_MAX_WIDTH / w
        roi = cv2.resize(roi, (int(w * scale), int(roi.shape[0] * scale)))

    lines = []
    for box, text, score in _raw_detections(roi):
        text = _clean(text)
        if score < MIN_LINE_SCORE or len(text) < 2:
            continue
        if _is_watermark(text):
            continue
        xs = [float(p[0]) / scale for p in box]
        ys = [float(p[1]) / scale + roi_top for p in box]
        lines.append(
            {
                "text": text,
                "score": score,
                "left": min(xs),
                "right": max(xs),
                "top": min(ys),
                "bottom": max(ys),
                "height": max(ys) - min(ys),
            }
        )
    return _merge_rows(lines)


# Platform prefixes normalised so "ig", "insta" and "instagram" don't become three platforms.
_PLATFORM_ALIASES = {
    "ig": "instagram",
    "insta": "instagram",
    "instagram": "instagram",
    "x": "twitter",
    "tw": "twitter",
    "twitter": "twitter",
    "twitch": "twitch",
    "tt": "tiktok",
    "tiktok": "tiktok",
    "yt": "youtube",
    "youtube": "youtube",
}


def _looks_like_handle(text: str) -> bool:
    """Whether a caption line is an account rather than a character or series name.

    The "@" is checked first but cannot be required: OCR loses it on most frames. What survives is
    the shape of a handle -- one word, no spaces, and mostly lowercase -- which is exactly what the
    uppercase character/series lines are not. "Cchristianperera" (a misread "@christianperera")
    still passes; "MARVELRIVALS" does not.
    """
    if _HANDLE_RE.search(text):
        return True
    token = text.strip()
    if " " in token or not (3 <= len(token) <= 30):
        return False
    if not re.fullmatch(r"[A-Za-z0-9._]+", token):
        return False
    letters = [c for c in token if c.isalpha()]
    return bool(letters) and sum(c.islower() for c in letters) / len(letters) >= 0.6


def _parse_social(text):
    """Splits "twitch @alinity" into ("twitch", "alinity"). Platform is None for a bare handle."""
    match = _HANDLE_RE.search(text)
    if match:
        prefix = re.sub(r"[^a-z]", "", text[: match.start()].lower())
        return {
            "platform": _PLATFORM_ALIASES.get(prefix, prefix or None),
            "handle": match.group(1).lower().strip("._"),
        }
    return {"platform": None, "handle": text.strip().lower().strip("._")}


def pick_twitter(socials):
    """Which of the listed accounts is the twitter one.

    A bare handle counts: that is how the channel labelled everyone before it started naming
    platforms, and it is what the older videos in the catalog look like. When there is genuinely no
    twitter, this returns None rather than passing off an instagram as one -- the other accounts
    are still kept in `handles`, so nothing is lost by being accurate here.
    """
    for social in socials:
        if social["platform"] == "twitter":
            return social["handle"]
    for social in socials:
        if social["platform"] is None:
            return social["handle"]
    return None


def _stack_from(lines, index):
    """Grows the column of lines starting at lines[index]. `lines` must be sorted top to bottom."""
    chain = [lines[index]]
    for candidate in lines[index + 1 :]:
        last = chain[-1]
        h = last["height"] or 1.0
        # Side by side with the last line, not below it: a different column of the frame, so it
        # neither joins the stack nor ends it.
        if candidate["top"] < last["bottom"] - 0.3 * h:
            continue
        # Measured against the head of the stack, not the previous line, so small drifts don't
        # accumulate down a four-line caption. The tolerance is wide (two line heights) for one
        # specific reason: when OCR drops the leading "@", that line's box starts a whole character
        # to the right of the others -- 72px against a 71px budget was enough to break the stack.
        aligned = abs(candidate["left"] - chain[0]["left"]) <= 2.0 * h
        same_size = 0.55 * h <= candidate["height"] <= 1.8 * h
        # Generous on the vertical gap because OCR boxes hug the glyphs: an all-caps line has no
        # descenders, so its box is short and the measured gap to the next line runs past 1.2*h.
        # Tightening this is what made the parser miss stacks as obvious as CYCLOPS/MARVEL RIVALS.
        adjacent = -0.2 * h <= candidate["top"] - last["bottom"] <= 1.8 * h
        if aligned and same_size and adjacent:
            chain.append(candidate)
        else:
            # The first line below that doesn't fit ends the stack. Continuing past it would let
            # the parser reach across the frame and glue unrelated text onto the caption.
            break
    return chain


def _find_stack(lines):
    """The most caption-like column in the frame, or None."""
    ordered = sorted(lines, key=lambda ln: ln["top"])
    best = None
    for index in range(len(ordered)):
        chain = _stack_from(ordered, index)
        # Two lines minimum, and at least one of them shaped like an account. A lone line, or a
        # column of pure uppercase, is signage far more often than it is a caption.
        if len(chain) < 2 or not any(_looks_like_handle(ln["text"]) for ln in chain):
            continue
        # Longest wins; on a tie the lower one, since captions sit near the bottom of the frame.
        rank = (len(chain), chain[0]["top"])
        if best is None or rank > (len(best), best[0]["top"]):
            best = chain
    return best


def parse_caption(lines):
    """Picks the caption block out of everything OCR saw in a frame. None when there isn't one."""
    block = _find_stack(lines)
    if not block:
        return None

    socials = []
    texts = []
    for line in block:
        text = line["text"].strip()
        # A platform name on its own is what's left when OCR read "twitch @alinity" and lost the
        # handle half. It is neither an account nor part of the series title, so it is dropped --
        # keeping it appended "TWITCH" to the series name.
        if re.sub(r"[^a-z]", "", text.lower()) in _PLATFORM_ALIASES:
            continue
        if _looks_like_handle(text):
            socials.append(_parse_social(text))
        else:
            texts.append(text.upper())

    # A stack of handles with nothing named above it is not a caption -- it is much more likely a
    # sponsor banner or someone's badge.
    if not texts:
        return None

    character = texts[0]
    # One line above the accounts is the normal case for each of character and series. An extra one
    # means something wrapped, and it is the series that wraps in practice -- character names are
    # short, titles like "FRIEREN BEYOND JOURNEY'S END" are not -- so extras join the series.
    series = " ".join(texts[1:]) if len(texts) > 1 else None

    scores = [ln["score"] for ln in block]
    return {
        "character": character,
        "series": series,
        "socials": socials,
        "twitter": pick_twitter(socials),
        "score": sum(scores) / len(scores),
        "raw_lines": [ln["text"] for ln in block],
        # Kept so the caller can tell where and how big the caption was on screen; useful when
        # debugging a video where the block was misread.
        "left": block[0]["left"],
        "top": block[0]["top"],
        "line_height": block[0]["height"],
    }


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()


def same_caption(a, b) -> bool:
    """Whether two frames are showing the same cosplayer's caption.

    Compared fuzzily on purpose: the same block read one second apart routinely differs by a
    character or two (motion blur, a passer-by crossing the text), and an exact match would split
    one cosplayer into several entries.
    """
    handles_a = [s["handle"] for s in a.get("socials") or []]
    handles_b = [s["handle"] for s in b.get("socials") or []]
    if handles_a and handles_b:
        # Both frames read an account, so the account IS the identity and a mismatch means a
        # different person. An earlier version fell back to the names here, and at a convention
        # where every caption's series line said MARVEL RIVALS that shared line dragged the
        # similarity over the threshold -- Luna Snow and Ultron were filed as one cosplayer.
        return any(_ratio(x, y) >= 0.7 for x in handles_a for y in handles_b)
    # One of the frames lost its account line entirely. Compare on the character name alone: it is
    # the part that actually distinguishes people, while the series repeats all video long.
    return _ratio(a["character"] or "", b["character"] or "") >= 0.8


def consolidate(observations):
    """Vote across every frame of a segment instead of trusting any single reading.

    One frame's OCR is noisy; five readings of the same caption are not, and the disagreement rate
    doubles as the confidence score the UI uses to flag entries worth eyeballing.
    """

    def winner(field, key_of=lambda v: re.sub(r"[^a-z0-9]", "", str(v).lower())):
        weights = {}
        best_value = {}
        for obs in observations:
            value = obs.get(field)
            if not value:
                continue
            key = key_of(value)
            weights[key] = weights.get(key, 0.0) + obs["score"]
            # Best-scoring reading represents the group, with word breaks as the tie-break --
            # "MARVEL RIVALS" is preferred over an equally confident "MARVELRIVALS". Spacing is
            # only cosmetic for grouping, since the key above already ignores it, which is what
            # keeps both spellings of a series counted as one in the catalog.
            rank = (obs["score"], str(value).count(" "))
            if rank > best_value.get(key, ((-1, 0.0), None))[0]:
                best_value[key] = (rank, value)
        if not weights:
            return None, 0.0
        top = max(weights, key=weights.get)
        return best_value[top][1], weights[top] / sum(weights.values())

    character, char_agreement = winner("character")
    series, series_agreement = winner("series")
    twitter, handle_agreement = winner("twitter")
    # The whole account list is voted on as a unit rather than per account: a frame where OCR
    # dropped one of the three lines should lose to the frames that read all of them, not
    # contribute a partial list that then has to be merged with the others.
    socials, _ = winner(
        "socials",
        key_of=lambda v: "|".join(sorted(f"{s['platform']}:{s['handle']}" for s in v)),
    )

    mean_score = sum(o["score"] for o in observations) / len(observations)
    agreements = [a for a in (char_agreement, series_agreement, handle_agreement) if a]
    agreement = sum(agreements) / len(agreements) if agreements else 0.0

    return {
        "character": character,
        "series": series,
        "twitter": twitter,
        "socials": socials or [],
        "confidence": round(mean_score * agreement, 4),
        "readings": len(observations),
        "raw_lines": observations[len(observations) // 2]["raw_lines"],
    }
