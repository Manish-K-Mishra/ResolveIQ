"""Runtime kill switch (Governance §5): stop automatic answers immediately, with no deployment.

    python -m cloudserve.killswitch on --reason "wrong answers about billing"
    python -m cloudserve.killswitch status
    python -m cloudserve.killswitch off

The switch is a flag file, `storage/KILL_SWITCH` by default (override with
CLOUDSERVE_KILL_SWITCH_FILE). The decision engine reads it in `finalize` for EVERY ticket, so it
takes effect from the next ticket that finalises, including tickets already in flight; decisions
already finalised are not changed. While it is engaged no answer is released, even with
--auto-send: auto-respond decisions are held for an agent. Anyone with write access to the
deployment's storage directory can operate it; that access is the authorisation control.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from . import ROOT

DEFAULT_FILE = ROOT / "storage" / "KILL_SWITCH"


def flag_path() -> Path:
    return Path(os.getenv("CLOUDSERVE_KILL_SWITCH_FILE") or DEFAULT_FILE)


def display_path(path: Path) -> str:
    """Repository-relative when possible, so records never carry a local absolute path."""
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def state(path: Path | None = None) -> dict:
    path = Path(path or flag_path())
    if not path.exists():
        return {"engaged": False, "flag_file": display_path(path)}
    try:
        info = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError):
        info = {}           # any file at the path engages the switch, even an empty or odd one
    return {"engaged": True, "flag_file": display_path(path),
            "engaged_at": info.get("engaged_at"), "reason": info.get("reason")}


def engage(reason: str = "", path: Path | None = None) -> dict:
    path = Path(path or flag_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"engaged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "reason": reason[:300]}) + "\n", encoding="utf-8")
    return state(path)


def release(path: Path | None = None) -> dict:
    path = Path(path or flag_path())
    path.unlink(missing_ok=True)
    return state(path)


def main(argv=None):
    p = argparse.ArgumentParser(description="Engage, release or inspect the CloudServe kill switch")
    p.add_argument("command", choices=["on", "off", "status"])
    p.add_argument("--reason", default="", help="why the switch is engaged (stored in the flag file)")
    a = p.parse_args(argv)
    s = engage(a.reason) if a.command == "on" else release() if a.command == "off" else state()
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
