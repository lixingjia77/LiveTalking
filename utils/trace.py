from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from threading import Lock

from utils.logger import logger


TRACE_PREFIX = "_trace_"
MAX_TRACES = 512
MAX_EVENTS_PER_TRACE = 128

_trace_lock = Lock()
_trace_events = OrderedDict()


def _append_event(trace_event: dict):
    tid = trace_event["trace_id"]
    with _trace_lock:
        events = _trace_events.setdefault(tid, [])
        events.append(trace_event)
        if len(events) > MAX_EVENTS_PER_TRACE:
            del events[:-MAX_EVENTS_PER_TRACE]
        _trace_events.move_to_end(tid)
        while len(_trace_events) > MAX_TRACES:
            _trace_events.popitem(last=False)


def new_trace(sessionid: str = "", route: str = "human") -> dict:
    now = time.perf_counter()
    trace_id = f"{route}-{sessionid or 'none'}-{uuid.uuid4().hex[:8]}"
    return {
        "_trace_id": trace_id,
        "_trace_start": now,
        "_trace_last": now,
        "_trace_logged": set(),
    }


def trace_id(datainfo: dict) -> str:
    return str(datainfo.get("_trace_id", "no-trace"))


def mark(datainfo: dict, stage: str, detail: str = "", once_key: str = ""):
    if not datainfo or "_trace_start" not in datainfo:
        return
    logged = datainfo.setdefault("_trace_logged", set())
    if once_key:
        if once_key in logged:
            return
        logged.add(once_key)

    now = time.perf_counter()
    start = float(datainfo.get("_trace_start", now))
    last = float(datainfo.get("_trace_last", start))
    datainfo["_trace_last"] = now
    elapsed_ms = (now - start) * 1000
    delta_ms = (now - last) * 1000

    _append_event(
        {
            "trace_id": trace_id(datainfo),
            "stage": stage,
            "elapsed_ms": round(elapsed_ms, 3),
            "delta_ms": round(delta_ms, 3),
            "detail": detail,
            "timestamp": time.time(),
        }
    )

    suffix = f" {detail}" if detail else ""
    logger.info(
        "[trace:%s] %s elapsed=%.1fms delta=%.1fms%s",
        trace_id(datainfo),
        stage,
        elapsed_ms,
        delta_ms,
        suffix,
    )


def get_trace_events(tid: str) -> list[dict]:
    with _trace_lock:
        events = _trace_events.get(tid, [])
        return [dict(event) for event in events]


def strip_trace(datainfo: dict) -> dict:
    if not datainfo:
        return datainfo
    for key in list(datainfo.keys()):
        if key.startswith(TRACE_PREFIX):
            datainfo.pop(key, None)
    return datainfo
