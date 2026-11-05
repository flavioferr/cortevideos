"""Biblioteca de músicas de fundo: importação, análise (librosa) e escolha por clipe.

Análise com algoritmos consolidados de MIR (Music Information Retrieval):
- BPM e batidas: beat tracking por programação dinâmica (Ellis, 2007) — librosa.beat.beat_track
- Tom e modo (maior/menor): perfis de Krumhansl-Schmuckler correlacionados com o cromagrama
- Energia: RMS; brilho: centroide espectral; ritmo: taxa de ataques (onset detection)
Esses descritores viram uma posição no modelo circumplexo de Russell (arousal × valence),
o mesmo espaço em que o Claude descreve o clima de cada clipe.
"""

import json
import logging
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)

MUSIC_DIR = config.DATA_DIR / "music"
LIBRARY_FILE = MUSIC_DIR / "library.json"
STATE_FILE = MUSIC_DIR / "import_state.json"
ANALYSIS_SR = 22050
AUDIO_EXTS = {".m4a", ".webm", ".opus", ".mp3", ".ogg", ".wav", ".aac", ".flac"}

# Perfis de Krumhansl-Kessler para tonalidades maiores e menores (Dó como tônica).
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_lock = threading.RLock()
MUSIC_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def load_library() -> list[dict]:
    with _lock:
        if not LIBRARY_FILE.exists():
            return []
        return json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))


def save_library(tracks: list[dict]) -> None:
    with _lock:
        _write_json(LIBRARY_FILE, tracks)


def get_track(track_id: str) -> Optional[dict]:
    return next((t for t in load_library() if t["id"] == track_id), None)


def delete_track(track_id: str) -> None:
    with _lock:
        tracks = load_library()
        for t in tracks:
            if t["id"] == track_id:
                (MUSIC_DIR / t["file"]).unlink(missing_ok=True)
        save_library([t for t in tracks if t["id"] != track_id])


def import_state() -> dict:
    with _lock:
        if not STATE_FILE.exists():
            return {"status": "idle", "progress": 0, "message": ""}
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def _set_state(**fields) -> None:
    with _lock:
        state = import_state()
        state.update(fields)
        _write_json(STATE_FILE, state)


def recover_interrupted() -> None:
    if import_state().get("status") == "running":
        _set_state(status="error", message="Importação interrompida (servidor reiniciado).")


# ---------------------------------------------------------------------------
# Análise
# ---------------------------------------------------------------------------

def decode(path: Path, sr: int = ANALYSIS_SR) -> np.ndarray:
    proc = subprocess.run(
        [config.ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def estimate_key(chroma_mean: np.ndarray) -> tuple[str, str, float, float]:
    """Krumhansl-Schmuckler: correlaciona o cromagrama com os 24 perfis rotacionados."""
    scores = []
    for shift in range(12):
        scores.append(("major", shift, np.corrcoef(chroma_mean, np.roll(MAJOR_PROFILE, shift))[0, 1]))
        scores.append(("minor", shift, np.corrcoef(chroma_mean, np.roll(MINOR_PROFILE, shift))[0, 1]))
    scores.sort(key=lambda s: s[2], reverse=True)
    mode, tonic, best = scores[0]
    best_major = max(s[2] for s in scores if s[0] == "major")
    best_minor = max(s[2] for s in scores if s[0] == "minor")
    return KEY_NAMES[tonic], mode, float(best), float(best_major - best_minor)


def analyze(path: Path) -> dict:
    import librosa

    y = decode(path)
    if len(y) < ANALYSIS_SR * 5:
        raise RuntimeError("Áudio curto demais para analisar.")
    duration = len(y) / ANALYSIS_SR

    tempo, beats = librosa.beat.beat_track(y=y, sr=ANALYSIS_SR, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    rms = librosa.feature.rms(y=y)[0]
    rms_db = float(20 * np.log10(np.mean(rms) + 1e-9))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=ANALYSIS_SR)))
    onsets = librosa.onset.onset_detect(y=y, sr=ANALYSIS_SR, units="time")
    onset_rate = len(onsets) / duration
    chroma = librosa.feature.chroma_stft(y=y, sr=ANALYSIS_SR)
    key, mode, key_strength, mode_margin = estimate_key(chroma.mean(axis=1))

    # Curva de energia por segundo (para escolher o melhor trecho da música).
    hop = 512
    frames_per_sec = ANALYSIS_SR / hop
    energy_curve = []
    for s in range(int(duration)):
        chunk = rms[int(s * frames_per_sec) : int((s + 1) * frames_per_sec)]
        energy_curve.append(round(float(chunk.mean()) if chunk.size else 0.0, 5))

    # Mapeamento para o circumplexo de Russell (descritores normalizados para 0..1).
    tempo_n = _clip01((tempo - 60) / 120)
    loud_n = _clip01((rms_db + 30) / 22)
    onset_n = _clip01(onset_rate / 5)
    bright_n = _clip01((centroid - 800) / 3200)
    mode_n = _clip01(0.5 + 4 * mode_margin)  # >0.5 = tende a maior
    arousal = 0.35 * loud_n + 0.25 * tempo_n + 0.25 * onset_n + 0.15 * bright_n
    valence = 0.5 * mode_n + 0.25 * tempo_n + 0.25 * bright_n

    return {
        "duration": round(duration, 2),
        "bpm": round(tempo, 1),
        "beats": [round(float(b), 3) for b in beats],
        "key": key,
        "mode": mode,
        "key_strength": round(key_strength, 3),
        "rms_db": round(rms_db, 2),
        "brightness": round(centroid, 1),
        "onset_rate": round(onset_rate, 2),
        "energy_curve": energy_curve,
        "algo_arousal": round(_clip01(arousal), 3),
        "algo_valence": round(_clip01(valence), 3),
    }


# ---------------------------------------------------------------------------
# Importação
# ---------------------------------------------------------------------------

def _download_audio(url: str, on_item: Callable[[int, int, str], None]) -> list[dict]:
    import yt_dlp

    from .download import _ffmpeg_dir

    with yt_dlp.YoutubeDL({"extract_flat": "in_playlist", "quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = [e for e in (info.get("entries") or [info]) if e and e.get("id")]

    known = {t["id"] for t in load_library()}
    downloaded = []
    for n, entry in enumerate(entries):
        title = entry.get("title") or entry["id"]
        on_item(n, len(entries), title)
        if entry["id"] in known or title in ("[Deleted video]", "[Private video]"):
            continue
        options = {
            "format": "bestaudio[ext=m4a]/bestaudio",
            "outtmpl": str(MUSIC_DIR / "%(id)s.%(ext)s"),
            "ffmpeg_location": _ffmpeg_dir(),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                meta = ydl.extract_info(f"https://www.youtube.com/watch?v={entry['id']}", download=True)
        except Exception as exc:
            logger.warning("Falha ao baixar %s: %s", title, exc)
            continue
        files = [p for p in MUSIC_DIR.glob(f"{entry['id']}.*") if p.suffix in AUDIO_EXTS]
        if files:
            downloaded.append({
                "id": entry["id"],
                "title": meta.get("title") or title,
                "artist": meta.get("artist") or meta.get("channel") or "",
                "file": files[0].name,
                "source_url": meta.get("webpage_url", ""),
            })
    return downloaded


def import_playlist(url: str) -> None:
    """Baixa, analisa e classifica as músicas. Roda em thread."""
    from . import claude_ai

    try:
        _set_state(status="running", progress=0.0, message="lendo a playlist")

        def on_item(n: int, total: int, title: str) -> None:
            _set_state(progress=0.5 * n / max(1, total), message=f"baixando {n + 1}/{total}: {title}")

        new_tracks = _download_audio(url, on_item)
        for n, track in enumerate(new_tracks):
            _set_state(progress=0.5 + 0.35 * n / max(1, len(new_tracks)), message=f"analisando {track['title']}")
            try:
                track.update(analyze(MUSIC_DIR / track["file"]))
            except Exception as exc:
                logger.warning("Falha ao analisar %s: %s", track["title"], exc)
                track["error"] = str(exc)

        tracks = load_library() + [t for t in new_tracks if "error" not in t]
        untagged = [t for t in tracks if "moods" not in t]
        if untagged:
            _set_state(progress=0.87, message="Claude classificando o clima das músicas")
            for tag in claude_ai.tag_tracks(untagged):
                track = next((t for t in tracks if t["id"] == tag["id"]), None)
                if track:
                    track.update(
                        moods=tag["moods"], genre=tag["genre"], vocals=tag["vocals"],
                        good_for=tag["good_for"],
                        claude_arousal=_clip01(tag["arousal"]), claude_valence=_clip01(tag["valence"]),
                    )
        save_library(tracks)
        _set_state(status="done", progress=1.0, message=f"{len(new_tracks)} música(s) nova(s), {len(tracks)} no total")
    except Exception as exc:
        logger.exception("Falha ao importar playlist")
        _set_state(status="error", message=str(exc))


# ---------------------------------------------------------------------------
# Escolha por clipe
# ---------------------------------------------------------------------------

def track_mood(track: dict) -> tuple[float, float]:
    """Combina a análise do sinal com a avaliação do Claude (quando houver)."""
    a, v = track.get("algo_arousal", 0.5), track.get("algo_valence", 0.5)
    if "claude_arousal" in track:
        a = 0.5 * a + 0.5 * track["claude_arousal"]
        v = 0.5 * v + 0.5 * track["claude_valence"]
    return a, v


def score_track(track: dict, target: dict, used: set[str]) -> float:
    """Nota de 0 a 1: distância no plano arousal×valence, afinidade de clima, voz e repetição."""
    a, v = track_mood(track)
    distance = math.hypot(a - target.get("arousal", 0.5), v - target.get("valence", 0.5))
    score = 1 - distance / math.sqrt(2)
    wanted = {m.lower() for m in target.get("moods", [])}
    if wanted:
        score += 0.12 * len(wanted & {m.lower() for m in track.get("moods", [])}) / len(wanted)
    if track.get("vocals"):
        score -= 0.12  # voz cantada disputa atenção com a fala
    if track["id"] in used:
        score -= 0.35
    return round(score, 3)


def rank_tracks(target: dict, used: set[str], limit: int = 6) -> list[tuple[dict, float]]:
    tracks = [t for t in load_library() if "bpm" in t]
    ranked = sorted(((t, score_track(t, target, used)) for t in tracks), key=lambda x: x[1], reverse=True)
    return ranked[:limit]


def best_offset(track: dict, clip_seconds: float, arousal: float) -> float:
    """Escolhe onde a música começa: janela com energia adequada ao clima, alinhada a uma batida."""
    curve = np.array(track.get("energy_curve") or [0.0])
    window = max(1, int(round(clip_seconds)))
    duration = track.get("duration", len(curve))
    if len(curve) <= window + 2:
        return 0.0
    sums = np.convolve(curve, np.ones(window), mode="valid") / window
    skip = min(len(sums) - 1, 4)  # pula o comecinho (silêncio/intro)
    candidates = sums[skip:]
    # Clima agitado: trecho mais forte (refrão/drop). Clima calmo: trecho de energia mediana.
    quantile = 0.5 + 0.5 * _clip01(arousal)
    target = np.quantile(candidates, quantile)
    start = skip + int(np.argmin(np.abs(candidates - target)))
    beats = [b for b in track.get("beats", []) if b <= duration - clip_seconds]
    if beats:
        start = min(beats, key=lambda b: abs(b - start))
    return round(float(start), 3)


def gain_db(track: dict, volume: float) -> float:
    """Normaliza a loudness da faixa e aplica o volume relativo escolhido."""
    reference = -26.0  # RMS alvo da música em volume 0.15, abaixo da fala
    return round(reference - track.get("rms_db", -20.0) + 20 * math.log10(max(volume, 0.01) / 0.15), 2)
