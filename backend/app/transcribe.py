"""Transcrição local com timestamps por palavra (faster-whisper).

O Claude não recebe áudio, então a transcrição é feita aqui e ele trabalha em cima do texto.
"""

import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from . import config
from .models import Word

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            logger.info("Carregando Whisper %s (%s)", config.WHISPER_MODEL, config.WHISPER_DEVICE)
            _model = WhisperModel(
                config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )
        return _model


def extract_audio(video: Path):
    """Áudio mono 16 kHz em float32, decodificado pelo FFmpeg (evita depender da versão do PyAV)."""
    import numpy as np

    proc = subprocess.run(
        [config.ffmpeg_exe(), "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32)


def transcribe(
    video: Path,
    work_dir: Path,
    language: str = "",
    on_progress: Optional[Callable[[float], None]] = None,
) -> dict:
    """Retorna {"language", "words": [...], "segments": [...]} e guarda em cache no work_dir."""
    cache = work_dir / "transcript.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    segments_iter, info = _get_model().transcribe(
        extract_audio(video),
        language=language or None,
        word_timestamps=True,
        vad_filter=True,
    )

    words: list[dict] = []
    segments: list[dict] = []
    for seg in segments_iter:
        first = len(words)
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append({"i": len(words), "text": text, "start": round(w.start, 3), "end": round(w.end, 3)})
        if len(words) > first:
            segments.append({"first_word": first, "last_word": len(words) - 1})
        if on_progress and info.duration:
            on_progress(min(1.0, seg.end / info.duration))

    data ={"language": info.language, "words": words, "segments": segments}
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def load_words(data: dict) -> list[Word]:
    return [Word(**w) for w in data["words"]]
