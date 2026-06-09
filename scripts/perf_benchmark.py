#!/usr/bin/env python3
"""
Run repeatable LiveTalking first-frame latency benchmarks.

Default mode is in-process: the script loads the avatar/model, creates one
session, starts render threads, submits text directly, and reads structured
trace events from memory. This is suitable for CI and does not require an HTTP
server or a manually-created session.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from utils.trace import get_trace_events, mark, new_trace, trace_id


PRE_SUBMIT_STAGES = [
    ("avatar.put_msg_txt", "avatar 收到文本", "avatar.put_msg_txt"),
    ("low_latency.clear_backlog", "清理低延迟 backlog", "清掉旧帧和队列积压"),
    ("tts.queue.enqueue", "TTS 入队", "TTS 任务进入队列"),
]

POST_SUBMIT_STAGES = [
    ("tts.worker.dequeue", "TTS worker 取任务", "TTS worker 开始处理"),
    ("tts.qwen.first_audio_frame_to_avatar", "Qwen 首音频帧到 avatar", "TTS 首包耗时"),
    ("avatar.first_audio_frame_received", "avatar 收到首音频帧", "首个音频 chunk"),
    ("asr.queue.first_audio_enqueue", "ASR 首音频入队", "进入 ASR 队列"),
    ("asr.queue.first_audio_dequeue", "ASR 首音频出队", "render loop 消费等待"),
    ("asr.whisper.first_feat_enqueue", "Whisper 首特征完成", "音频特征完成"),
    ("avatar.infer.first_audio_batch", "Avatar 拿到首音频 batch", "inference 调度等待"),
    ("avatar.infer.first_model_start", "MuseTalk 小 batch 推理开始", "模型推理开始"),
    ("avatar.infer.first_model_done", "MuseTalk 小 batch 推理完成", "模型推理耗时"),
    ("avatar.infer.first_frame_enqueue", "首帧入结果队列", "推理结果入队"),
    ("avatar.output.first_frame_dequeue", "output 取到首帧", "output 消费结果"),
    ("avatar.output.first_video_push", "首视频 push", "服务端推送视频"),
    ("avatar.output.first_audio_push", "首音频 push", "服务端首帧链路完成"),
]

INPROCESS_STAGES = [
    ("benchmark.request", "benchmark 收到请求", "脚本内提交测试样本"),
    ("benchmark.interrupt", "interrupt", "打断旧播放"),
    ("benchmark.dispatch_echo", "dispatch echo", "分发到 avatar"),
    *PRE_SUBMIT_STAGES,
    ("benchmark.submit_done", "提交完成", "脚本提交路径完成"),
    *POST_SUBMIT_STAGES,
]

HTTP_STAGES = [
    ("http.human.request", "/human 收到请求", "HTTP 解析"),
    ("http.human.interrupt", "interrupt", "打断旧播放"),
    ("http.human.dispatch_echo", "dispatch echo", "分发到 echo"),
    *PRE_SUBMIT_STAGES,
    ("http.human.response_ok", "HTTP 返回 OK", "API 已返回"),
    *POST_SUBMIT_STAGES,
]


def build_runtime_options(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        fps=args.fps,
        l=args.l,
        m=args.m,
        r=args.r,
        model=args.model,
        avatar_id=args.avatar_id,
        batch_size=args.batch_size,
        low_latency=args.low_latency,
        low_latency_batch_size=args.low_latency_batch_size,
        modelres=args.modelres,
        modelfile=args.modelfile,
        customvideo_config=args.customvideo_config,
        customopt=load_customopt(args.customvideo_config),
        tts=args.tts,
        REF_FILE=args.ref_file,
        REF_TEXT=args.ref_text,
        TTS_SERVER=args.tts_server,
        transport="null",
        push_url="",
        max_session=1,
        listenport=0,
        sessionid=args.sessionid,
        qwen_tts_model=args.qwen_tts_model,
        qwen_tts_url=args.qwen_tts_url,
        dashscope_api_key=args.dashscope_api_key,
    )


def load_customopt(path: str) -> list[Any]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_inprocess_avatar(opt: argparse.Namespace):
    module_names = {
        "musetalk": "avatars.musetalk_avatar",
        "musetalk_fused": "avatars.musetalk_avatar",
        "wav2lip": "avatars.wav2lip_avatar",
        "ultralight": "avatars.ultralight_avatar",
    }
    avatar_mod = importlib.import_module(module_names[opt.model])
    importlib.import_module("streamout.null")

    if opt.model in ("musetalk", "musetalk_fused"):
        model = avatar_mod.load_model()
        avatar = avatar_mod.load_avatar(opt.avatar_id)
        avatar_mod.warm_up(opt.batch_size, model)
    elif opt.model == "wav2lip":
        model_path = opt.modelfile or "./models/wav2lip.pth"
        model = avatar_mod.load_model(model_path)
        avatar = avatar_mod.load_avatar(opt.avatar_id)
        avatar_mod.warm_up(opt.batch_size, model, opt.modelres)
    elif opt.model == "ultralight":
        model = avatar_mod.load_model(opt)
        avatar = avatar_mod.load_avatar(opt.avatar_id)
        avatar_mod.warm_up(opt.batch_size, avatar, 160)
    else:
        raise ValueError(f"unsupported model: {opt.model}")

    import registry

    return registry.create("avatar", opt.model, opt=opt, model=model, avatar=avatar)


def start_render_thread(avatar_session) -> tuple[Event, Thread]:
    quit_event = Event()
    thread = Thread(
        target=avatar_session.render,
        args=(quit_event,),
        name="perf-benchmark-render",
        daemon=True,
    )
    thread.start()
    return quit_event, thread


def submit_inprocess(avatar_session, args: argparse.Namespace) -> str:
    datainfo = new_trace(sessionid=args.sessionid, route="benchmark")
    mark(datainfo, "benchmark.request", detail=f"type=echo text_len={len(args.text)}")
    if not args.no_interrupt:
        mark(datainfo, "benchmark.interrupt")
        avatar_session.flush_talk()
    mark(datainfo, "benchmark.dispatch_echo")
    avatar_session.put_msg_txt(args.text, datainfo)
    mark(datainfo, "benchmark.submit_done", detail=f"trace_id={trace_id(datainfo)}")
    return trace_id(datainfo)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: float) -> dict[str, Any] | None:
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def submit_http(args: argparse.Namespace) -> str:
    payload = {
        "sessionid": args.sessionid,
        "type": args.type,
        "text": args.text,
        "interrupt": not args.no_interrupt,
    }
    response = post_json(f"{args.base_url.rstrip('/')}/human", payload, timeout=args.timeout)
    if response.get("code") != 0:
        raise RuntimeError(f"/human failed: {response}")
    trace_id_value = response.get("data", {}).get("trace_id")
    if not trace_id_value:
        raise RuntimeError("server did not return data.trace_id")
    return trace_id_value


def wait_trace(
    args: argparse.Namespace,
    trace_id_value: str,
    complete_stage: str,
) -> list[dict[str, Any]]:
    deadline = time.time() + args.timeout
    last_events: list[dict[str, Any]] = []

    while time.time() < deadline:
        if args.mode == "http":
            data = get_json(f"{args.base_url.rstrip('/')}/api/traces/{trace_id_value}", timeout=5)
            if data and data.get("code") == 0:
                last_events = data.get("data", {}).get("events", [])
        else:
            last_events = get_trace_events(trace_id_value)

        if any(event.get("stage") == complete_stage for event in last_events):
            return last_events
        time.sleep(args.poll_interval)

    seen = ", ".join(event.get("stage", "") for event in last_events)
    raise TimeoutError(f"trace {trace_id_value} did not reach {complete_stage}; seen: {seen}")


def event_map(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for event in events:
        result.setdefault(event["stage"], event)
    return result


def summarize(runs: list[dict[str, Any]], stages: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    rows = []
    for index, (stage, label, description) in enumerate(stages):
        elapsed_values = []
        phase_values = []
        present = 0

        for run in runs:
            events = run["events_by_stage"]
            if stage not in events:
                continue
            present += 1
            elapsed = float(events[stage]["elapsed_ms"])
            elapsed_values.append(elapsed)

            previous_elapsed = None
            for prev_stage, _, _ in reversed(stages[:index]):
                if prev_stage in events:
                    previous_elapsed = float(events[prev_stage]["elapsed_ms"])
                    break
            if previous_elapsed is not None:
                phase_values.append(elapsed - previous_elapsed)

        if elapsed_values:
            rows.append(
                {
                    "stage": stage,
                    "label": label,
                    "present": present,
                    "avg_elapsed_ms": statistics.mean(elapsed_values),
                    "avg_phase_ms": statistics.mean(phase_values) if phase_values else None,
                    "min_elapsed_ms": min(elapsed_values),
                    "max_elapsed_ms": max(elapsed_values),
                    "std_elapsed_ms": statistics.pstdev(elapsed_values) if len(elapsed_values) > 1 else 0.0,
                    "description": description,
                }
            )
    return rows


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}ms"


def render_markdown(rows: list[dict[str, Any]], total_runs: int) -> str:
    lines = [
        "| 阶段 | 样本 | 平均时间点 | 平均阶段耗时 | min/max 时间点 | 说明 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {present}/{total} | {elapsed} | {phase} | {minv}/{maxv} | {description} |".format(
                label=row["label"],
                present=row["present"],
                total=total_runs,
                elapsed=fmt_ms(row["avg_elapsed_ms"]),
                phase=fmt_ms(row["avg_phase_ms"]),
                minv=fmt_ms(row["min_elapsed_ms"]),
                maxv=fmt_ms(row["max_elapsed_ms"]),
                description=row["description"],
            )
        )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path | None,
    rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    markdown: str,
):
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text(markdown + "\n", encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps({"summary": rows, "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["stage"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LiveTalking first-frame latency.")

    parser.add_argument("--mode", choices=["inprocess", "http"], default="inprocess")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=0.02)
    parser.add_argument("--complete-stage", default="avatar.output.first_audio_push")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--text", default="你好，这是一次低延迟性能测试。")
    parser.add_argument("--sessionid", default="bench")
    parser.add_argument("--no-interrupt", action="store_true")

    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--type", default="echo", choices=["echo", "chat"])

    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("-l", type=int, default=10)
    parser.add_argument("-m", type=int, default=8)
    parser.add_argument("-r", type=int, default=10)
    parser.add_argument("--model", default="musetalk", choices=["musetalk", "musetalk_fused", "wav2lip", "ultralight"])
    parser.add_argument("--avatar-id", default="musetalk_avatar1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--low-latency", action="store_true")
    parser.add_argument("--low-latency-batch-size", type=int, default=2)
    parser.add_argument("--modelres", type=int, default=192)
    parser.add_argument("--modelfile", default="")
    parser.add_argument("--customvideo-config", default="")
    parser.add_argument("--tts", default="qwentts")
    parser.add_argument("--ref-file", default="Cherry")
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--tts-server", default="http://127.0.0.1:9880")
    parser.add_argument("--qwen-tts-model", default="qwen3-tts-flash-realtime")
    parser.add_argument("--qwen-tts-url", default="wss://dashscope.aliyuncs.com/api-ws/v1/realtime")
    parser.add_argument("--dashscope-api-key", default=None)
    return parser.parse_args()


def maybe_load_dotenv():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


def main():
    args = parse_args()
    maybe_load_dotenv()

    avatar_session = None
    quit_event = None
    render_thread = None
    if args.mode == "inprocess":
        opt = build_runtime_options(args)
        avatar_session = load_inprocess_avatar(opt)
        quit_event, render_thread = start_render_thread(avatar_session)
        time.sleep(0.2)

    stages = HTTP_STAGES if args.mode == "http" else INPROCESS_STAGES
    measured_runs = []
    total = args.warmup + args.runs

    try:
        for index in range(total):
            is_warmup = index < args.warmup
            if args.mode == "http":
                tid = submit_http(args)
            else:
                tid = submit_inprocess(avatar_session, args)

            events = wait_trace(args, tid, args.complete_stage)
            label = "warmup" if is_warmup else "run"
            print(f"{label} {index + 1}/{total}: trace_id={tid}, events={len(events)}")

            if not is_warmup:
                measured_runs.append(
                    {
                        "trace_id": tid,
                        "events": events,
                        "events_by_stage": event_map(events),
                    }
                )
            time.sleep(args.interval)

        rows = summarize(measured_runs, stages)
        markdown = render_markdown(rows, total_runs=len(measured_runs))
        print()
        print(markdown)
        write_outputs(args.output_dir, rows, measured_runs, markdown)
    finally:
        if quit_event is not None:
            quit_event.set()
        if render_thread is not None:
            render_thread.join(timeout=10)
            if render_thread.is_alive():
                print("warning: render thread did not stop within 10s", file=sys.stderr)


if __name__ == "__main__":
    main()
