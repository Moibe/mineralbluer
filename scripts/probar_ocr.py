"""Debug tool: dump what the caption parser reads, frame by frame, on a short clip.

Tuning the overlay parser means looking at what OCR actually returned, not at the final catalog --
a missing cosplayer can come from a bad reading, a rejected block, or a segment that fell under
MIN_READINGS, and only the raw dump tells those apart.

    venv\\Scripts\\python scripts\\probar_ocr.py <url> [inicio_seg] [fin_seg]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

import overlay
from config import STORAGE_ROOT
from pipeline import download_video, iter_sampled_frames

DEFAULT_URL = "https://www.youtube.com/watch?v=76MzBuoulOg"
# Frames land here so a misreading can be checked against what was actually on screen.
DUMP_DIR = os.path.join(STORAGE_ROOT, "debug")


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    end = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
    force = "--force" in sys.argv
    dump = "--dump" in sys.argv
    if dump:
        os.makedirs(DUMP_DIR, exist_ok=True)

    path, info = download_video(url, "ocr-test", start, end, force_redownload=force)
    print(f"video: {info.get('title')} @ {info.get('height')}p -> {path}\n", flush=True)

    found = 0
    for timestamp, frame, _idx, _total in iter_sampled_frames(path, target_fps=1.0):
        lines = overlay.read_lines(frame)
        caption = overlay.parse_caption(lines)
        stamp = f"{start + timestamp:7.1f}s"
        if dump:
            cv2.imwrite(
                os.path.join(DUMP_DIR, f"{int(start + timestamp):06d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 88],
            )
        if caption:
            found += 1
            socials = " ".join(
                f"{s['platform'] or ''}@{s['handle']}" for s in caption["socials"]
            )
            print(
                f"{stamp}  OK  {caption['character']} | {caption['series']} | "
                f"{socials}  -> twitter={caption['twitter']}  ({caption['score']:.2f})",
                flush=True,
            )
        elif lines:
            texts = " / ".join(f"{ln['text']}({ln['score']:.2f})" for ln in lines[:6])
            print(f"{stamp}  --  {texts}", flush=True)
        else:
            print(f"{stamp}  --", flush=True)

    print(f"\n{found} frames con caption reconocido")


if __name__ == "__main__":
    main()
