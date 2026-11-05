"""Gera legendas .ass (libass) sincronizadas palavra a palavra, mais o título-gancho."""

import os
from typing import Optional

from .models import CaptionStyle, CaptionWord, ClipDocument

EMOJI_FONT = "Segoe UI Emoji" if os.name == "nt" else "Noto Color Emoji"
CHUNK_GAP_SECONDS = 0.7
BRIDGE_GAP_SECONDS = 0.35


def ass_color(hex_color: str, alpha: int = 0) -> str:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        value = "FFFFFF"
    r, g, b = value[0:2], value[2:4], value[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def tag_color(hex_color: str) -> str:
    """Cor para tags inline (\\c), formato &HBBGGRR&."""
    return ass_color(hex_color).replace("&H00", "&H", 1) + "&"


def ass_time(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def map_time(t: float, ranges: list[tuple[float, float]]) -> Optional[float]:
    """Converte um tempo do vídeo original para o tempo do clipe já cortado."""
    elapsed = 0.0
    for start, end in ranges:
        if t < start - 1e-3:
            return None
        if t <= end + 1e-3:
            return elapsed + (min(t, end) - start)
        elapsed += end - start
    return None


class TimedWord:
    def __init__(self, word: CaptionWord, start: float, end: float):
        self.word = word
        self.start = start
        self.end = end


def _visible_words(doc: ClipDocument, ranges: list[tuple[float, float]]) -> list[TimedWord]:
    result = []
    for w in doc.words:
        mid = (w.start + w.end) / 2
        if map_time(mid, ranges) is None:
            continue  # palavra cortada
        start = map_time(w.start, ranges)
        end = map_time(w.end, ranges)
        start = start if start is not None else map_time(mid, ranges)
        end = end if end is not None else map_time(mid, ranges)
        result.append(TimedWord(w, start, max(end, start + 0.05)))
    return result


def _chunks(words: list[TimedWord], per_line: int) -> list[list[TimedWord]]:
    chunks: list[list[TimedWord]] = []
    current: list[TimedWord] = []
    for tw in words:
        if current and (
            len(current) >= per_line
            or tw.start - current[-1].end > CHUNK_GAP_SECONDS
            or current[-1].word.text.endswith((".", "!", "?", ","))
        ):
            chunks.append(current)
            current = []
        current.append(tw)
    if current:
        chunks.append(current)
    return chunks


def _layout(style: CaptionStyle, height: int) -> tuple[int, int]:
    """(alinhamento ASS, margem vertical)"""
    return {
        "bottom": (2, int(height * 0.24)),
        "middle": (5, 0),
        "top": (8, int(height * 0.2)),
    }[style.position]


def _word_text(tw: TimedWord, style: CaptionStyle) -> str:
    text = tw.word.text.upper() if style.uppercase else tw.word.text
    return escape(text)


def build_ass(
    doc: ClipDocument, ranges: list[tuple[float, float]], width: int, height: int, seam: Optional[int] = None
) -> str:
    """seam: altura da divisa jogo/webcam; a legenda fica sobre ela para não cobrir o rosto."""
    style = doc.style
    alignment, margin_v = _layout(style, height)
    if seam is not None and style.position != "top":
        alignment, margin_v = 2, height - seam - int(style.font_size * 0.55)
    primary = ass_color(style.primary_color)
    highlight = ass_color(style.highlight_color)
    outline = ass_color(style.outline_color)
    hook_bg = ass_color(style.hook_title_bg)
    hook_fg = ass_color(style.hook_title_color)
    hook_size = max(40, int(style.font_size * 0.8))
    is_karaoke = style.animation == "karaoke"
    tag_primary = tag_color(style.primary_color)
    tag_highlight = tag_color(style.highlight_color)

    # No karaokê o \kf preenche da cor secundária para a primária.
    caption_primary, caption_secondary = (highlight, primary) if is_karaoke else (primary, highlight)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{style.font},{style.font_size},{caption_primary},{caption_secondary},{outline},&H80000000,-1,0,0,0,100,100,0,0,1,{style.outline_width},2,{alignment},80,80,{margin_v},1
Style: Hook,{style.font},{hook_size},{hook_fg},{hook_fg},{hook_bg},{hook_bg},-1,0,0,0,100,100,0,0,3,18,0,8,90,90,{int(height * 0.11)},1
Style: Emoji,{EMOJI_FONT},{int(style.font_size * 1.3)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []

    def add(layer: int, start: float, end: float, style_name: str, text: str) -> None:
        if end - start >= 0.02:
            events.append(f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},{style_name},,0,0,0,,{text}")

    if style.hook_title_enabled and doc.hook_title.strip():
        title = escape(doc.hook_title.strip())
        add(2, 0.0, style.hook_title_seconds, "Hook", r"{\fad(150,250)}" + title)

    words = _visible_words(doc, ranges)
    chunks = _chunks(words, max(1, style.words_per_line))
    for idx, chunk in enumerate(chunks):
        chunk_start = chunk[0].start
        chunk_end = chunk[-1].end
        if idx + 1 < len(chunks) and chunks[idx + 1][0].start - chunk_end < BRIDGE_GAP_SECONDS:
            chunk_end = chunks[idx + 1][0].start

        if is_karaoke:
            parts = []
            for k, tw in enumerate(chunk):
                nxt = chunk[k + 1].start if k + 1 < len(chunk) else chunk_end
                parts.append(rf"{{\kf{max(1, round((nxt - tw.start) * 100))}}}{_word_text(tw, style)}")
            add(1, chunk_start, chunk_end, "Caption", " ".join(parts))
        elif style.animation == "static":
            parts = [
                (rf"{{\c{tag_highlight}}}{_word_text(tw, style)}{{\c{tag_primary}}}" if tw.word.emphasis else _word_text(tw, style))
                for tw in chunk
            ]
            add(1, chunk_start, chunk_end, "Caption", " ".join(parts))
        else:  # highlight / pop: a palavra falada ganha cor (e animação no "pop")
            for k, tw in enumerate(chunk):
                start = chunk_start if k == 0 else tw.start
                end = chunk[k + 1].start if k + 1 < len(chunk) else chunk_end
                parts = []
                for j, other in enumerate(chunk):
                    text = _word_text(other, style)
                    if j == k and len(chunk) == 1 and not other.word.emphasis:
                        # Uma palavra por vez: ela já é a falada, então usa a cor principal.
                        anim = r"\fscx75\fscy75\t(0,90,\fscx112\fscy112)\t(90,160,\fscx105\fscy105)" if style.animation == "pop" else ""
                        parts.append(rf"{{{anim}}}{text}" if anim else text)
                    elif j == k:
                        anim = (
                            r"\fscx75\fscy75\t(0,90,\fscx112\fscy112)\t(90,160,\fscx105\fscy105)"
                            if style.animation == "pop"
                            else r"\fscx108\fscy108"
                        )
                        parts.append(rf"{{\c{tag_highlight}{anim}}}{text}{{\c{tag_primary}\fscx100\fscy100}}")
                    elif other.word.emphasis:
                        parts.append(rf"{{\c{tag_highlight}}}{text}{{\c{tag_primary}}}")
                    else:
                        parts.append(text)
                add(1, start, end, "Caption", " ".join(parts))

        if style.show_emojis:
            emoji = next((tw.word.emoji for tw in chunk if tw.word.emoji), "")
            if emoji:
                cx = width // 2
                if style.position == "bottom":
                    y = height - margin_v - int(style.font_size * 2.4)
                elif style.position == "middle":
                    y = height // 2 - int(style.font_size * 1.2)
                else:
                    y = margin_v + int(style.font_size * 3.6)
                add(3, chunk_start, chunk_end, "Emoji", rf"{{\an2\pos({cx},{y})\fad(80,80)}}{escape(emoji)}")

    return header + "\n".join(events) + "\n"
