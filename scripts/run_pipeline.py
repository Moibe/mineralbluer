"""Run a full analysis from the terminal, without the API or the front.

    venv\\Scripts\\python scripts\\run_pipeline.py <url> [inicio_seg] [fin_seg]

Writes to the same catalog the API reads, so anything analysed here shows up in the gallery.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_pipeline

DEFAULT_URL = "https://www.youtube.com/watch?v=76MzBuoulOg"


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    start = float(sys.argv[2]) if len(sys.argv) > 2 else None
    end = float(sys.argv[3]) if len(sys.argv) > 3 else None

    def on_progress(stage, fraction, stage_fraction=None):
        detail = f" ({stage_fraction:.0%} de la etapa)" if stage_fraction else ""
        print(f"[{stage}] {fraction:.0%}{detail}", flush=True)

    result = run_pipeline("cli", url, progress_cb=on_progress, start_seconds=start, end_seconds=end)

    print(json.dumps(result["stats"], indent=2, ensure_ascii=False))
    for entry in result["cosplays"]:
        accounts = " ".join(
            f"{s['platform'] or ''}@{s['handle']}" for s in entry.get("handles") or []
        )
        print(
            f"  {entry['ts_start']:7.1f}s  {entry['character']} | {entry['series']} | "
            f"{accounts or '(sin cuenta)'}  (conf {entry['confidence']})"
        )


if __name__ == "__main__":
    main()
