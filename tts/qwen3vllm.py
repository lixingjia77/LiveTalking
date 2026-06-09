import time

import numpy as np
import requests
import resampy

from registry import register
from utils.logger import logger
from utils.trace import mark
from .base_tts import BaseTTS, State


SRC_SR = 24000


@register("tts", "qwen3vllm")
class Qwen3VllmTTS(BaseTTS):
    """
    vLLM-Omni Qwen3-TTS PCM streaming adapter.

    The server returns 24 kHz mono signed 16-bit PCM bytes. LiveTalking consumes
    16 kHz float32 frames, so this adapter incrementally resamples and pushes
    20 ms frames into the avatar pipeline.
    """

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.server_url = opt.TTS_SERVER.rstrip("/")
        self.endpoint = f"{self.server_url}/v1/audio/speech"
        self.model = getattr(opt, "qwen3_vllm_model", "") or None
        self.task_type = getattr(opt, "qwen3_vllm_task_type", "CustomVoice")
        self.language = getattr(opt, "qwen3_vllm_language", "Chinese")
        self.api_key = getattr(opt, "qwen3_vllm_api_key", "") or "EMPTY"
        self.timeout = float(getattr(opt, "qwen3_vllm_timeout", 120.0))
        self.read_chunk_bytes = int(getattr(opt, "qwen3_vllm_read_chunk_bytes", 960))
        self.max_new_tokens = int(getattr(opt, "qwen3_vllm_max_new_tokens", 4096))
        self.session = requests.Session()

        self._pcm_remainder = b""
        self._sample_remainder = np.array([], dtype=np.float32)

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        voice = textevent.get("tts", {}).get("ref_file", self.opt.REF_FILE)
        instructions = textevent.get("tts", {}).get("ref_text", self.opt.REF_TEXT)
        language = textevent.get("tts", {}).get("language", self.language)
        task_type = textevent.get("tts", {}).get("task_type", self.task_type)

        self._pcm_remainder = b""
        self._sample_remainder = np.array([], dtype=np.float32)
        first = True
        first_pcm = True
        start = time.perf_counter()

        payload = {
            "input": text,
            "voice": voice,
            "language": language,
            "task_type": task_type,
            "stream": True,
            "response_format": "pcm",
            "max_new_tokens": self.max_new_tokens,
        }
        if self.model:
            payload["model"] = self.model
        if instructions:
            payload["instructions"] = instructions

        tts_opts = textevent.get("tts", {})
        for key in ("ref_audio", "ref_text", "x_vector_only_mode", "max_new_tokens"):
            if key in tts_opts and tts_opts[key] is not None:
                payload[key] = tts_opts[key]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        mark(
            textevent,
            "tts.qwen3vllm.start",
            detail=(
                f"text_len={len(text)} voice={voice} language={language} "
                f"task_type={task_type} max_new_tokens={payload.get('max_new_tokens')}"
            ),
        )

        try:
            with self.session.post(
                self.endpoint,
                json=payload,
                headers=headers,
                stream=True,
                timeout=(10.0, self.timeout),
            ) as res:
                post_done = time.perf_counter()
                logger.info("qwen3vllm POST ready in %.3fs", post_done - start)
                mark(
                    textevent,
                    "tts.qwen3vllm.post_ready",
                    detail=f"cost_ms={(post_done - start) * 1000:.1f}",
                )

                if res.status_code != 200:
                    logger.error("qwen3vllm error %s: %s", res.status_code, res.text)
                    mark(textevent, "tts.qwen3vllm.error", detail=f"status={res.status_code}")
                    return

                for chunk in res.iter_content(chunk_size=self.read_chunk_bytes):
                    if self.state != State.RUNNING:
                        break
                    if not chunk:
                        continue
                    if first_pcm:
                        first_audio = time.perf_counter()
                        logger.info("qwen3vllm first PCM chunk in %.3fs", first_audio - start)
                        mark(
                            textevent,
                            "tts.qwen3vllm.first_pcm_chunk",
                            detail=f"cost_ms={(first_audio - start) * 1000:.1f} bytes={len(chunk)}",
                        )
                        first_pcm = False
                    first = self._push_pcm_chunk(chunk, msg, first)

            self._flush_remainders(msg, first)
            mark(
                textevent,
                "tts.qwen3vllm.done",
                detail=f"cost_ms={(time.perf_counter() - start) * 1000:.1f}",
            )
        except Exception:
            logger.exception("qwen3vllm txt_to_audio failed")
            mark(textevent, "tts.qwen3vllm.exception")

    def _push_pcm_chunk(self, pcm_data: bytes, msg: tuple[str, dict], first: bool) -> bool:
        pcm_data = self._pcm_remainder + pcm_data
        aligned_len = len(pcm_data) - (len(pcm_data) % 2)
        self._pcm_remainder = pcm_data[aligned_len:]
        if aligned_len <= 0:
            return first

        samples = np.frombuffer(pcm_data[:aligned_len], dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return first

        samples = resampy.resample(x=samples, sr_orig=SRC_SR, sr_new=self.sample_rate)
        if self._sample_remainder.size > 0:
            samples = np.concatenate([self._sample_remainder, samples])

        first = self._push_samples(samples, msg, first)
        whole_len = (samples.shape[0] // self.chunk) * self.chunk
        self._sample_remainder = samples[whole_len:] if whole_len < samples.shape[0] else np.array([], dtype=np.float32)
        return first

    def _push_samples(self, samples: np.ndarray, msg: tuple[str, dict], first: bool) -> bool:
        text, textevent = msg
        idx = 0
        while samples.shape[0] - idx >= self.chunk and self.state == State.RUNNING:
            eventpoint = {}
            if first:
                eventpoint = {"status": "start", "text": text}
                first = False
            eventpoint.update(**textevent)
            frame = samples[idx:idx + self.chunk].astype(np.float32, copy=False)
            mark(
                eventpoint,
                "tts.qwen3vllm.first_audio_frame_to_avatar",
                detail=f"samples={frame.shape[0]}",
                once_key="qwen3vllm_first_audio_frame_to_avatar",
            )
            self.parent.put_audio_frame(frame, eventpoint)
            idx += self.chunk
        return first

    def _flush_remainders(self, msg: tuple[str, dict], first: bool):
        if self.state == State.RUNNING and self._sample_remainder.size >= self.chunk:
            first = self._push_samples(self._sample_remainder, msg, first)

        text, textevent = msg
        eventpoint = {"status": "end", "text": text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

        self._pcm_remainder = b""
        self._sample_remainder = np.array([], dtype=np.float32)

    def stop_tts(self):
        self.session.close()
        logger.info("qwen3vllm stopped")
