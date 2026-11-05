"""Configuração central, lida do .env na raiz do projeto."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def _default_data_dir() -> Path:
    # Vídeos não devem ficar em pastas sincronizadas (OneDrive bloqueia arquivos e envia gigabytes).
    local = os.getenv("LOCALAPPDATA")
    return Path(local) / "cortevideos" / "data" if os.name == "nt" and local else ROOT_DIR / "data"


DATA_DIR = Path(os.getenv("DATA_DIR") or _default_data_dir())
JOBS_DIR = DATA_DIR / "jobs"
FONTS_DIR = ROOT_DIR / "fonts"
FRONTEND_DIR = ROOT_DIR / "frontend"

# Como falar com o Claude:
#   claude-code = usa o Claude Code instalado no PC, com o login da sua assinatura (Pro/Max)
#   api         = usa a API da Anthropic com ANTHROPIC_API_KEY (cobrança por uso)
CLAUDE_BACKEND = os.getenv("CLAUDE_BACKEND", "claude-code")
CLAUDE_CODE_PATH = os.getenv("CLAUDE_CODE_PATH", "claude")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
# Esforço de raciocínio do Claude: low | medium | high | xhigh | max
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "high")

# Modelo do faster-whisper: tiny | base | small | medium | large-v3
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

JOBS_DIR.mkdir(parents=True, exist_ok=True)
FONTS_DIR.mkdir(parents=True, exist_ok=True)


def ffmpeg_exe() -> str:
    """Usa o ffmpeg do sistema se existir; senão, o binário do imageio-ffmpeg."""
    custom = os.getenv("FFMPEG_PATH")
    if custom:
        return custom
    import shutil

    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
