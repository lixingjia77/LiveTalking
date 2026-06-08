import time
import uuid

from utils.logger import logger


TRACE_PREFIX = "_trace_"


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

    suffix = f" {detail}" if detail else ""
    logger.info(
        "[trace:%s] %s elapsed=%.1fms delta=%.1fms%s",
        trace_id(datainfo),
        stage,
        (now - start) * 1000,
        (now - last) * 1000,
        suffix,
    )


def strip_trace(datainfo: dict) -> dict:
    if not datainfo:
        return datainfo
    for key in list(datainfo.keys()):
        if key.startswith(TRACE_PREFIX):
            datainfo.pop(key, None)
    return datainfo
