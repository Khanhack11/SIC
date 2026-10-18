"""Cooperative pause marker for long-running local project jobs."""

from pathlib import Path

PAUSE_FILE = Path(__file__).resolve().parents[1] / "artifacts" / "PAUSE_REQUESTED"


def check_pause():
    if PAUSE_FILE.exists():
        print(f"Paused safely. Progress is saved. Remove {PAUSE_FILE} and rerun the same command to continue.", flush=True)
        raise SystemExit(75)
