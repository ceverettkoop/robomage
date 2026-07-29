"""Atomic JSON progress sidecars for the resumable drivers.

Every long-running driver in this repo (the PFSP league, an archetype exploiter
run, the AZ league, a curriculum) persists its loop position to a small JSON
sidecar next to its checkpoints so an interrupted run can be resumed. The write
must be crash-safe: a half-written file would break --resume, so it always goes
through write-to-temp + os.replace.

This module is the single home for that read/write pair. It is stdlib-only on
purpose — train.py (torch), az_train.py (torch) and the lightweight tui.py /
curriculum.py all import it, so it can never pull the ML stack into the
launcher.
"""

import json
import os


def write_progress_state(path: str, state: dict, label: str) -> None:
    """Atomically persist a driver's progress dict to its JSON sidecar.

    ``label`` names the driver in the warning printed when the file cannot be
    written (the run itself continues — losing resumability is not worth
    aborting a training session over)."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[{label}] WARNING: could not write progress file {path}: {exc}")


def read_progress_state(path: str, label: str) -> dict | None:
    """Load a driver's progress sidecar, or None if it is absent/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[{label}] WARNING: could not read progress file {path}: {exc}")
        return None
