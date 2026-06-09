###############################################################################
#  配置解析 — CLI 参数 + YAML 配置
###############################################################################

import argparse
import json
import os


def str_or_int(value):
    """尝试转换为 int，失败则返回 str"""
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LiveTalking Digital Human Server")

    # ─── 音频 ──────────────────────────────────────────────────────────
    parser.add_argument('--fps', type=int, default=25, help="video fps, must be 25")
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    # ─── 画面 ──────────────────────────────────────────────────────────
    # parser.add_argument('--W', type=int, default=450, help="GUI width")
    # parser.add_argument('--H', type=int, default=450, help="GUI height")

    # ─── 数字人模型 ────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='wav2lip',
                        help="avatar model: musetalk/musetalk_fused/wav2lip/ultralight")
    parser.add_argument('--avatar_id', type=str, default='wav2lip256_avatar1',
                        help="avatar id in data/avatars")
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")
    parser.add_argument('--low_latency', action='store_true',
                        help="optimize first-frame latency with small first batches and shorter output backpressure sleeps")
    parser.add_argument('--low_latency_batch_size', type=int, default=2,
                        help="batch size used while low-latency mode is waiting for the first speaking frame")
    parser.add_argument('--modelres', type=int, default=192)
    parser.add_argument('--modelfile', type=str, default='')

    # ─── 自定义动作和多形象 ────────────────────────────────────────────
    parser.add_argument('--customvideo_config', type=str, default='',
                        help="custom action json")

    # ─── TTS ───────────────────────────────────────────────────────────
    parser.add_argument('--tts', type=str, default='edgetts',
                        help="tts plugin: edgetts/gpt-sovits/cosyvoice/fishtts/tencent/doubao/indextts2/azuretts/qwentts/qwen3vllm")
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",
                        help="参考文件名或语音模型ID")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')
    parser.add_argument('--qwen3_vllm_model', type=str, default='',
                        help="vLLM-Omni Qwen3-TTS model name/path, optional when server has a single model")
    parser.add_argument('--qwen3_vllm_task_type', type=str, default='CustomVoice',
                        help="Qwen3-TTS task type: CustomVoice/VoiceDesign/Base")
    parser.add_argument('--qwen3_vllm_language', type=str, default='Chinese',
                        help="Qwen3-TTS language: Auto/Chinese/English/etc.")
    parser.add_argument('--qwen3_vllm_api_key', type=str, default='EMPTY',
                        help="Bearer token for vLLM-Omni server")
    parser.add_argument('--qwen3_vllm_timeout', type=float, default=120.0,
                        help="read timeout seconds for streaming PCM response")
    parser.add_argument('--qwen3_vllm_read_chunk_bytes', type=int, default=960,
                        help="PCM read chunk bytes, 960 is 20ms at 24kHz s16 mono")
    parser.add_argument('--qwen3_vllm_max_new_tokens', type=int, default=4096,
                        help="max audio tokens for Qwen3-TTS generation; too small truncates audio")

    # ─── 传输 ─────────────────────────────────────────────────────────
    parser.add_argument('--transport', type=str, default='webrtc',
                        help="output: rtcpush/webrtc/rtmp/virtualcam")
    parser.add_argument('--push_url', type=str,
                        default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument('--max_session', type=int, default=1)
    parser.add_argument('--listenport', type=int, default=8010,
                        help="web listen port")

    opt = parser.parse_args()

    # ─── 后处理 ────────────────────────────────────────────────────────
    opt.customopt = []
    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r') as f:
            opt.customopt = json.load(f)

    return opt
