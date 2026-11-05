"""Download de vídeos por URL (YouTube, Twitch, Kick e tudo que o yt-dlp suporta)."""

import os
import shutil
from pathlib import Path
from typing import Callable, Optional

from . import config


def _ffmpeg_dir() -> str:
    """O yt-dlp procura um executável chamado 'ffmpeg'; garantimos que ele exista com esse nome."""
    exe = Path(config.ffmpeg_exe())
    if exe.stem == "ffmpeg":
        return str(exe.parent)
    bin_dir = config.DATA_DIR / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not target.exists():
        shutil.copy2(exe, target)
    return str(bin_dir)


def download(url: str, work_dir: Path, on_progress: Optional[Callable[[float], None]] = None) -> tuple[Path, str]:
    import yt_dlp

    last_reported = -1.0

    def hook(d: dict) -> None:
        nonlocal last_reported
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                fraction = d.get("downloaded_bytes", 0) / total
                if fraction - last_reported >= 0.02:  # o yt-dlp chama isto muitas vezes por segundo
                    last_reported = fraction
                    on_progress(fraction)

    options = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/bv*[height<=1080]+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(work_dir / "source.%(ext)s"),
        "ffmpeg_location": _ffmpeg_dir(),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    cookies = os.getenv("YTDLP_COOKIES_FILE")
    if cookies:
        options["cookiefile"] = cookies

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    files = sorted(work_dir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError("O download não gerou nenhum arquivo.")
    return files[0], info.get("title") or url
