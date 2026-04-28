"""Background PDF-ingest job state.

Each job lives under `ingest_jobs/{user_id}/{job_id}/` with two files:

* `source.pdf` — the uploaded PDF, kept until the job succeeds so we can
  restart on failure. Deleted on `done`.
* `state.json` — phase + progress + counters, polled by the home page.

A job's `phase` is one of:

* `rendering` — pages being rasterized
* `detecting` — boards being detected and saved
* `done`      — finished successfully (terminal)
* `error`     — runner raised; `error` field has the message (terminal)

Stalled jobs are computed at list-time: a non-terminal job whose
`updated_at` is older than `STALL_AFTER_SECONDS` is reported with
`stalled=true`. The user can then click Restart, which re-runs ingest
from the staged PDF; the existing `problem_exists` short-circuit makes
the resumed run skip already-saved boards.
"""

from __future__ import annotations

import calendar
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from .paths import ingest_job_dir, ingest_jobs_dir

log = logging.getLogger(__name__)

STALL_AFTER_SECONDS = 90
TERMINAL_PHASES = {"done", "error"}


class JobDismissed(Exception):
    """Raised by `save_state` when the job directory has been deleted
    out from under the runner — i.e. the user clicked Dismiss while the
    runner was still mid-flight. The runner catches this and exits
    quietly, leaving the dismissal sticky."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _state_path(user_id: str, job_id: str) -> Path:
    return ingest_job_dir(user_id, job_id) / "state.json"


def source_pdf_path(user_id: str, job_id: str) -> Path:
    return ingest_job_dir(user_id, job_id) / "source.pdf"


def new_job_id() -> str:
    return uuid.uuid4().hex


def create_job(user_id: str, job_id: str, source: str) -> dict[str, Any]:
    job_dir = ingest_job_dir(user_id, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "job_id": job_id,
        "user_id": user_id,
        "source": source,
        "phase": "rendering",
        "started_at": _now(),
        "updated_at": _now(),
        "total_pages": None,
        "pages_rendered": 0,
        "pages_detected": 0,
        "total_saved": 0,
        "skipped": 0,
        "error": None,
    }
    save_state(user_id, state)
    return state


def load_state(user_id: str, job_id: str) -> dict[str, Any] | None:
    p = _state_path(user_id, job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        log.warning("ingest_jobs: bad state file %s: %s", p, e)
        return None


def save_state(user_id: str, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    p = _state_path(user_id, state["job_id"])
    if not p.parent.exists():
        # User dismissed the job; don't resurrect it.
        raise JobDismissed(state["job_id"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(p)


def list_jobs(user_id: str) -> list[dict[str, Any]]:
    """Every job for the user, newest first, with `stalled` annotated."""
    root = ingest_jobs_dir(user_id)
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        state = load_state(user_id, child.name)
        if state is None:
            continue
        out.append(_annotate(state))
    out.sort(key=lambda s: s.get("started_at", ""), reverse=True)
    return out


def _annotate(state: dict[str, Any]) -> dict[str, Any]:
    state = dict(state)
    if state.get("phase") in TERMINAL_PHASES:
        state["stalled"] = False
        return state
    updated_at = state.get("updated_at") or state.get("started_at") or ""
    state["stalled"] = _seconds_since(updated_at) > STALL_AFTER_SECONDS
    return state


def _seconds_since(iso: str) -> float:
    """Seconds elapsed since `iso`, which is a UTC timestamp produced by
    `_now()`. Uses `calendar.timegm` so the conversion is correct under
    DST — `time.mktime` would interpret the struct as local time, and
    naive corrections via `time.timezone` ignore DST and skew the result
    by an hour during summer."""
    try:
        t = time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")
        return time.time() - calendar.timegm(t)
    except Exception:
        return float("inf")


def apply_event(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Mutate state in place from a `pdf_ingest.iter_ingest_events` payload."""
    et = event.get("event")
    if et == "start":
        state["phase"] = "rendering"
        state["total_pages"] = event.get("total_pages")
        state["source"] = event.get("source", state.get("source"))
    elif et == "page_rendered":
        state["pages_rendered"] = event.get("page", state.get("pages_rendered", 0))
        if (
            state.get("total_pages")
            and state["pages_rendered"] >= state["total_pages"]
        ):
            state["phase"] = "detecting"
    elif et == "page_detected":
        state["pages_detected"] = event.get("page", state.get("pages_detected", 0))
    elif et == "board_saved":
        state["total_saved"] = event.get("total_saved", state.get("total_saved", 0))
    elif et == "done":
        state["phase"] = "done"
        state["total_saved"] = event.get("total_saved", state.get("total_saved", 0))
        state["skipped"] = event.get("skipped", state.get("skipped", 0))
        state["source"] = event.get("source", state.get("source"))
    elif et == "error":
        state["phase"] = "error"
        state["error"] = event.get("detail") or event.get("message")
    return state


def mark_error(user_id: str, job_id: str, message: str) -> None:
    state = load_state(user_id, job_id)
    if state is None:
        return
    state["phase"] = "error"
    state["error"] = message
    try:
        save_state(user_id, state)
    except JobDismissed:
        # Dismissed between load and save; nothing to record.
        return


def delete_job(user_id: str, job_id: str) -> bool:
    """Remove the job directory entirely (state + staged PDF)."""
    job_dir = ingest_job_dir(user_id, job_id)
    if not job_dir.exists():
        return False
    for child in job_dir.iterdir():
        try:
            child.unlink()
        except FileNotFoundError:
            pass
    try:
        job_dir.rmdir()
    except OSError:
        pass
    return True


def cleanup_source_pdf(user_id: str, job_id: str) -> None:
    p = source_pdf_path(user_id, job_id)
    try:
        p.unlink()
    except FileNotFoundError:
        pass
