from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import Settings


log = logging.getLogger(__name__)

# Spark-TTS is not a pip package; the Dockerfile sets PYTHONPATH, but for local
# development honour ``SPARKTTS_ROOT`` as a fallback so ``from cli.SparkTTS ...``
# and ``from sparktts.* ...`` resolve.
_spark_root = os.environ.get("SPARKTTS_ROOT")
if _spark_root and _spark_root not in sys.path:
    sys.path.insert(0, _spark_root)


def _resolve_model_dir(repo_or_path: str) -> str:
    """Return a local model directory.

    If ``repo_or_path`` is an existing directory, use it. Otherwise treat it
    as a HuggingFace repo id and fetch a full snapshot. Spark-TTS is not
    published to ModelScope, so HF is the only remote source.
    """
    if os.path.isdir(repo_or_path):
        return repo_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_or_path)


class TTSEngine:
    """Thin async wrapper around :class:`cli.SparkTTS.SparkTTS`.

    Spark-TTS exposes two inference paths inside a single ``inference()``
    method — we re-enter it for the design (controllable) path, but the clone
    path is re-implemented here so the wav2vec2 + BiCodec tokenize step can
    be cached across requests using the same reference voice.
    """

    def __init__(self, settings: Settings) -> None:
        import torch

        from cli.SparkTTS import SparkTTS
        from sparktts.utils.token_parser import TASK_TOKEN_MAP

        self._settings = settings
        self._device_str = settings.resolved_device
        self._torch_device = torch.device(self._device_str)
        self._TASK = TASK_TOKEN_MAP

        self._resolved_model_dir = _resolve_model_dir(settings.sparktts_model)
        self._spark = SparkTTS(Path(self._resolved_model_dir), device=self._torch_device)

        self._apply_dtype()

        # key: (ref_audio_path, ref_mtime) -> (global_ids_cpu, semantic_ids_cpu)
        import torch as _t  # local alias for typing clarity below

        self._prompt_cache: dict[tuple[str, float], tuple[_t.Tensor, _t.Tensor]] = {}
        self._prompt_cache_order: list[tuple[str, float]] = []
        self._prompt_cache_max = settings.sparktts_prompt_cache_size
        self._prompt_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public attributes

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def dtype_str(self) -> str:
        # When we had to force float32 on CPU, report the effective dtype.
        if self._device_str.startswith("cpu"):
            return "float32"
        return self._settings.sparktts_dtype

    @property
    def sample_rate(self) -> int:
        return int(self._spark.sample_rate)

    @property
    def model_id(self) -> str:
        return self._settings.sparktts_model

    @property
    def builtin_voices_list(self) -> list[str]:
        return []  # Spark-TTS ships no built-in SFT voices.

    # ------------------------------------------------------------------
    # Dtype handling

    def _apply_dtype(self) -> None:
        """Cast just the LLM (HF CausalLM) to the requested dtype.

        We deliberately leave BiCodec and wav2vec2 in their default dtype —
        lightweight modules with occasional numerical sensitivity to fp16.
        CPU unconditionally runs fp32.
        """
        import torch

        if self._device_str.startswith("cpu"):
            if self._settings.sparktts_dtype != "float32":
                log.warning(
                    "CPU device: overriding SPARKTTS_DTYPE=%s -> float32",
                    self._settings.sparktts_dtype,
                )
            self._spark.model.to(torch.float32)
            return

        dmap = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self._spark.model.to(dmap[self._settings.sparktts_dtype])

    # ------------------------------------------------------------------
    # Prompt embedding cache

    def _get_or_tokenize(self, ref_audio: str, ref_mtime: Optional[float]):
        """Return ``(global_token_ids, semantic_token_ids)`` for ``ref_audio``.

        CPU copies are cached so GPU memory does not accumulate across
        cached entries. ``ref_mtime=None`` (one-shot upload) skips the cache.
        """
        if ref_mtime is None:
            return self._spark.audio_tokenizer.tokenize(ref_audio)

        key = (ref_audio, ref_mtime)
        with self._prompt_lock:
            cached = self._prompt_cache.get(key)
            if cached is not None:
                try:
                    self._prompt_cache_order.remove(key)
                except ValueError:
                    pass
                self._prompt_cache_order.append(key)
                return cached

        g, s = self._spark.audio_tokenizer.tokenize(ref_audio)
        g_cpu = g.detach().cpu()
        s_cpu = s.detach().cpu()

        with self._prompt_lock:
            self._prompt_cache[key] = (g_cpu, s_cpu)
            self._prompt_cache_order.append(key)
            while len(self._prompt_cache_order) > self._prompt_cache_max:
                old = self._prompt_cache_order.pop(0)
                self._prompt_cache.pop(old, None)

        return g_cpu, s_cpu

    # ------------------------------------------------------------------
    # Sync inference paths

    def _clone_sync(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        ref_mtime: Optional[float],
        temperature: float,
        top_k: int,
        top_p: float,
        max_new_tokens: int,
    ) -> np.ndarray:
        import torch

        g_ids_cached, s_ids_cached = self._get_or_tokenize(ref_audio, ref_mtime)
        g_ids_dev = g_ids_cached.to(self._torch_device)
        s_ids_dev = s_ids_cached.to(self._torch_device)

        global_tokens = "".join(
            f"<|bicodec_global_{i}|>" for i in g_ids_dev.squeeze()
        )
        semantic_tokens = "".join(
            f"<|bicodec_semantic_{i}|>" for i in s_ids_dev.squeeze()
        )

        prompt = (
            f"{self._TASK['tts']}<|start_content|>{ref_text}{text}<|end_content|>"
            f"<|start_global_token|>{global_tokens}<|end_global_token|>"
            f"<|start_semantic_token|>{semantic_tokens}"
        )

        tok = self._spark.tokenizer
        mdl = self._spark.model
        mi = tok([prompt], return_tensors="pt").to(self._torch_device)

        with torch.no_grad():
            out = mdl.generate(
                **mi,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )

        out = [o[len(i):] for i, o in zip(mi.input_ids, out)]
        predicts = tok.batch_decode(out, skip_special_tokens=True)[0]

        pred_sem_ids = (
            torch.tensor(
                [int(t) for t in re.findall(r"bicodec_semantic_(\d+)", predicts)]
            )
            .long()
            .unsqueeze(0)
            .to(self._torch_device)
        )

        with torch.no_grad():
            wav = self._spark.audio_tokenizer.detokenize(
                g_ids_dev.squeeze(0), pred_sem_ids
            )

        return np.asarray(wav, dtype=np.float32)

    def _design_sync(
        self,
        text: str,
        gender: str,
        pitch: str,
        speed_level: str,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> np.ndarray:
        # Spark-TTS' inference() with gender != None handles the whole
        # controllable path (including globals sampling). Re-use it.
        wav = self._spark.inference(
            text=text,
            gender=gender,
            pitch=pitch,
            speed=speed_level,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return np.asarray(wav, dtype=np.float32)

    # ------------------------------------------------------------------
    # Async API

    async def synthesize_clone(
        self,
        text: str,
        *,
        ref_audio: str,
        ref_text: str,
        ref_mtime: Optional[float] = None,
        instructions: Optional[str] = None,
        speed: float = 1.0,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        **_: object,
    ) -> np.ndarray:
        if instructions:
            log.warning(
                "instructions ignored: Spark-TTS clone path does not accept instruct prompts"
            )
        if speed != 1.0:
            log.warning(
                "speed=%.3f ignored on clone path: Spark-TTS has no speed knob for zero-shot; "
                "use /v1/audio/design for structured speed control",
                speed,
            )

        t = temperature if temperature is not None else self._settings.sparktts_temperature
        p = top_p if top_p is not None else self._settings.sparktts_top_p
        k = top_k if top_k is not None else self._settings.sparktts_top_k
        mnt = (
            max_new_tokens
            if max_new_tokens is not None
            else self._settings.sparktts_max_new_tokens
        )

        return await asyncio.to_thread(
            self._clone_sync,
            text,
            ref_audio,
            ref_text,
            ref_mtime,
            t,
            k,
            p,
            mnt,
        )

    async def synthesize_design(
        self,
        text: str,
        *,
        instruct: Optional[str] = None,
        gender: Optional[str] = None,
        pitch: Optional[str] = None,
        speed: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **_: object,
    ) -> np.ndarray:
        if instruct:
            log.info(
                "design: instruct received but not parsed (use gender/pitch/speed "
                "extension fields for control); instruct=%r",
                instruct[:80],
            )

        g = gender or self._settings.sparktts_default_gender
        pi = pitch or self._settings.sparktts_default_pitch
        sp = speed or self._settings.sparktts_default_speed

        t = temperature if temperature is not None else self._settings.sparktts_temperature
        p = top_p if top_p is not None else self._settings.sparktts_top_p
        k = top_k if top_k is not None else self._settings.sparktts_top_k

        return await asyncio.to_thread(
            self._design_sync,
            text,
            g,
            pi,
            sp,
            t,
            k,
            p,
        )
