"""Estruturas de dados: o "documento" de cada clipe é o que o Claude lê e edita."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

CaptionAnimation = Literal["highlight", "karaoke", "pop", "static"]
CaptionPosition = Literal["top", "middle", "bottom"]
LayoutMode = Literal["face", "center", "blur", "gameplay"]


class Word(BaseModel):
    i: int
    text: str
    start: float
    end: float


class CaptionWord(BaseModel):
    """Palavra da legenda (texto já corrigido pelo Claude)."""

    i: int
    text: str
    start: float
    end: float
    emphasis: bool = False
    emoji: str = ""


class Cut(BaseModel):
    """Trecho removido do clipe, em segundos do vídeo original."""

    start: float
    end: float
    reason: str = ""


class Virality(BaseModel):
    hook: int = 0
    engagement: int = 0
    value: int = 0
    shareability: int = 0
    total: int = 0
    hook_type: str = "none"
    reasoning: str = ""


class CaptionStyle(BaseModel):
    font: str = "Arial Black"
    font_size: int = 78
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFE600"
    outline_color: str = "#000000"
    outline_width: int = 6
    uppercase: bool = True
    words_per_line: int = 3
    position: CaptionPosition = "bottom"
    animation: CaptionAnimation = "highlight"
    show_emojis: bool = False
    hook_title_enabled: bool = True
    hook_title_seconds: float = 3.5
    hook_title_bg: str = "#FFFFFF"
    hook_title_color: str = "#000000"


CAPTION_PRESETS: dict[str, CaptionStyle] = {
    "classico": CaptionStyle(),
    "impacto": CaptionStyle(
        font="Impact", font_size=96, highlight_color="#39FF14", words_per_line=2,
        animation="pop", outline_width=8,
    ),
    "minimal": CaptionStyle(
        font="Arial", font_size=64, uppercase=False, highlight_color="#FFFFFF",
        outline_width=3, words_per_line=5, animation="static", show_emojis=False,
    ),
    "karaoke": CaptionStyle(
        font="Arial Black", font_size=74, highlight_color="#00D1FF",
        words_per_line=4, animation="karaoke",
    ),
}


class RenderState(BaseModel):
    status: Literal["pending", "rendering", "done", "error"] = "pending"
    file: str = ""
    error: str = ""
    version: int = 0


class MusicChoice(BaseModel):
    """Música de fundo de um clipe."""

    track_id: str
    title: str = ""
    offset: float = 0.0  # segundo da música onde o clipe começa
    volume: float = 0.15  # 0..1, volume relativo à voz
    ducking: bool = True  # abaixa a música enquanto alguém fala
    reason: str = ""


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ClipDocument(BaseModel):
    id: str
    job_id: str
    index: int
    start: float
    end: float
    title: str = ""
    hook_title: str = ""
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)
    virality: Virality = Field(default_factory=Virality)
    cuts: list[Cut] = Field(default_factory=list)
    words: list[CaptionWord] = Field(default_factory=list)
    style: CaptionStyle = Field(default_factory=CaptionStyle)
    layout: LayoutMode = "face"
    cam_box: Optional[list[float]] = None  # webcam no layout gameplay [x, y, w, h] 0..1; None = não calculada
    face_track: Optional[list[list[float]]] = None  # câmera virtual [[segundo, centro x 0..1]]; None = não calculada
    music: Optional[MusicChoice] = None
    music_target: dict = Field(default_factory=dict)  # clima pedido pelo Claude (arousal, valence, moods)
    render: RenderState = Field(default_factory=RenderState)
    chat: list[ChatMessage] = Field(default_factory=list)


class JobSettings(BaseModel):
    num_clips: int = 5
    clip_length: Literal["curto", "medio", "longo"] = "medio"
    language: str = ""  # vazio = detectar automaticamente
    caption_preset: str = "classico"
    layout: LayoutMode = "face"
    remove_silences: bool = True
    add_music: bool = True
    instructions: str = ""  # orientação livre para o Claude


class Job(BaseModel):
    id: str
    created_at: float
    source_url: str = ""
    source_name: str = ""
    settings: JobSettings = Field(default_factory=JobSettings)
    status: Literal["queued", "running", "done", "error"] = "queued"
    stage: str = "na fila"
    progress: float = 0.0
    error: str = ""
    summary: str = ""
    language: str = ""
    duration: float = 0.0
    clip_ids: list[str] = Field(default_factory=list)


CLIP_LENGTH_BOUNDS = {
    # (mínimo, ideal_min, ideal_max, máximo) em segundos
    "curto": (12, 15, 30, 35),
    "medio": (20, 30, 55, 65),
    "longo": (45, 60, 90, 120),
}
