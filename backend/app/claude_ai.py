"""Tudo que usa o Claude: escolher os cortes, editar/legendar cada um e o editor por chat."""

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Callable

import anthropic

from . import config, media, music
from .models import CLIP_LENGTH_BOUNDS, CaptionStyle, ClipDocument, Cut, JobSettings, MusicChoice, Word

logger = logging.getLogger(__name__)

# Se um classificador de segurança recusar a requisição, a API refaz a chamada
# automaticamente no modelo de fallback recomendado para aquela categoria.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=4)
    return _client


class ClaudeError(RuntimeError):
    pass


CLI_TIMEOUT_SECONDS = 900


def _ask_json(system: str, user: str, schema: dict, effort: str | None = None) -> dict:
    """Uma chamada com saída estruturada (JSON garantido pelo schema)."""
    effort = effort or config.CLAUDE_EFFORT
    if config.CLAUDE_BACKEND == "api":
        return _ask_json_api(system, user, schema, effort)
    return _ask_json_claude_code(system, user, schema, effort)


def _claude_code_exe() -> str:
    exe = shutil.which(config.CLAUDE_CODE_PATH)
    if not exe:
        raise ClaudeError(
            "Não encontrei o Claude Code. Instale-o (https://claude.com/claude-code) e faça login, "
            "ou defina CLAUDE_CODE_PATH no .env."
        )
    return exe


def _ask_json_claude_code(system: str, user: str, schema: dict, effort: str) -> dict:
    """Roda `claude -p` sem ferramentas; usa o login da assinatura em vez da API."""
    cmd = [
        _claude_code_exe(), "-p",
        "--output-format", "json",
        "--json-schema", json.dumps(schema, ensure_ascii=False),
        "--system-prompt", system,
        "--tools", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--model", config.CLAUDE_MODEL,
        "--effort", effort,
    ]
    # Sem a chave no ambiente o Claude Code usa o login da assinatura, não a API paga.
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "CLAUDECODE")}
    try:
        proc = subprocess.run(
            cmd, input=user.encode("utf-8"), capture_output=True, env=env,
            cwd=config.DATA_DIR, timeout=CLI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise ClaudeError("O Claude Code demorou demais para responder.")

    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        detail = (proc.stderr.decode("utf-8", errors="replace") or stdout).strip()
        raise ClaudeError(f"Falha ao chamar o Claude Code: {detail[-800:]}")
    if result.get("is_error") or result.get("structured_output") is None:
        raise ClaudeError(f"Claude Code: {str(result.get('result') or result.get('subtype'))[:800]}")
    return result["structured_output"]


def _ask_json_api(system: str, user: str, schema: dict, effort: str, max_tokens: int = 32000) -> dict:
    with client().beta.messages.stream(
        model=config.CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        thinking={"type": "adaptive"},
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": schema},
        },
        betas=[FALLBACK_BETA],
        fallbacks="default",
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise ClaudeError("O Claude recusou a requisição.")
    if message.stop_reason == "max_tokens":
        raise ClaudeError("A resposta do Claude foi cortada (max_tokens).")
    text = "".join(block.text for block in message.content if block.type == "text")
    return json.loads(text)


def _fmt(seconds: float) -> str:
    return f"{seconds:.2f}"


def format_transcript(words: list[Word], segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        chunk = words[seg["first_word"] : seg["last_word"] + 1]
        if not chunk:
            continue
        text = " ".join(w.text for w in chunk)
        lines.append(f"[{_fmt(chunk[0].start)}-{_fmt(chunk[-1].end)}] {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Seleção dos cortes
# ---------------------------------------------------------------------------

SELECT_SYSTEM = """Você é um editor sênior de vídeos curtos (TikTok, Reels, Shorts). Recebe a transcrição \
de um vídeo longo e escolhe os trechos com maior potencial de viralizar como clipes independentes.

Cada linha da transcrição tem o formato [início-fim] texto, com tempos em segundos.

Como escolher:
- Bons cortes: afirmações polêmicas ou contraintuitivas, erros e lições, exemplos concretos, histórias com \
desfecho, números surpreendentes, reações emocionais, respostas completas a perguntas interessantes.
- Evite: introduções, patrocínios e chamadas para ação, contexto vago, repetições, falas que dependem de algo \
que o espectador não viu.
- O clipe precisa fazer sentido sozinho. Comece na primeira palavra de uma frase que um espectador novo \
entende sem contexto (nunca no meio da frase, num "é", "tipo", "então", ou num pronome sem referência). \
Termine na frase que fecha o raciocínio.
- Use tempos que existem na transcrição: start = início de uma linha, end = fim de uma linha.
- Os clipes não podem se sobrepor.
- Qualidade acima de quantidade: se só houver 2 trechos realmente bons, devolva 2.

Pontuação de viralidade (cada nota de 0 a 25):
- hook_score: força dos primeiros 3 segundos para parar o scroll
- engagement_score: quão envolvente/emocionante é
- value_score: valor informativo ou de entretenimento
- shareability_score: vontade de mandar para alguém

hook_title: manchete curta (3 a 9 palavras) que aparece no topo do vídeo nos primeiros segundos. Deve criar \
curiosidade sem prometer o que o clipe não entrega, sem hashtags, sem emojis, sem aspas, no idioma da transcrição.
title: título para postar. description: legenda do post (1 a 2 frases). hashtags: 3 a 6, sem o #.
Escreva tudo no idioma da transcrição."""

SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "hook_title": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                    "reasoning": {"type": "string"},
                    "hook_type": {
                        "type": "string",
                        "enum": ["pergunta", "afirmacao", "estatistica", "historia", "contraste", "nenhum"],
                    },
                    "hook_score": {"type": "integer"},
                    "engagement_score": {"type": "integer"},
                    "value_score": {"type": "integer"},
                    "shareability_score": {"type": "integer"},
                },
                "required": [
                    "start", "end", "hook_title", "title", "description", "hashtags", "reasoning",
                    "hook_type", "hook_score", "engagement_score", "value_score", "shareability_score",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "clips"],
    "additionalProperties": False,
}


def select_clips(words: list[Word], segments: list[dict], duration: float, settings: JobSettings) -> dict:
    minimum, ideal_min, ideal_max, maximum = CLIP_LENGTH_BOUNDS[settings.clip_length]
    user = (
        f"Duração total do vídeo: {duration:.0f} segundos.\n"
        f"Quero até {settings.num_clips} clipes. Cada clipe deve ter entre {minimum} e {maximum} segundos, "
        f"idealmente {ideal_min} a {ideal_max}.\n"
    )
    if settings.instructions.strip():
        user += f"\nOrientação do criador do conteúdo:\n{settings.instructions.strip()}\n"
    user += f"\n<transcricao>\n{format_transcript(words, segments)}\n</transcricao>"

    result = _ask_json(SELECT_SYSTEM, user, SELECT_SCHEMA)
    result["clips"] = _validate_clips(result["clips"], words, settings)
    return result


def _clamp(value: int, low: int = 0, high: int = 25) -> int:
    return max(low, min(high, int(value)))


def _nearest_word(words: list[Word], t: float, attr: str) -> int:
    return min(range(len(words)), key=lambda i: abs(getattr(words[i], attr) - t))


def _validate_clips(raw: list[dict], words: list[Word], settings: JobSettings) -> list[dict]:
    """Encaixa os tempos nas palavras reais e descarta clipes inválidos ou sobrepostos."""
    minimum, _, _, maximum = CLIP_LENGTH_BOUNDS[settings.clip_length]
    accepted: list[dict] = []
    for clip in raw:
        start_i = _nearest_word(words, clip["start"], "start")
        end_i = _nearest_word(words, clip["end"], "end")
        if end_i <= start_i:
            continue
        start, end = words[start_i].start, words[end_i].end
        if not (minimum * 0.75 <= end - start <= maximum * 1.25):
            logger.info("Clipe descartado pela duração: %.1fs", end - start)
            continue
        if any(start < other["end"] and end > other["start"] for other in accepted):
            continue
        scores = [_clamp(clip[k]) for k in ("hook_score", "engagement_score", "value_score", "shareability_score")]
        clip.update(start=start, end=end, start_i=start_i, end_i=end_i, scores=scores, total=sum(scores))
        accepted.append(clip)
    accepted.sort(key=lambda c: c["total"], reverse=True)
    return accepted[: settings.num_clips]


# ---------------------------------------------------------------------------
# 2. Edição e legenda de cada clipe
# ---------------------------------------------------------------------------

EDIT_SYSTEM = """Você é um editor de vídeos curtos refinando UM clipe já escolhido. Você recebe as palavras \
transcritas (formato índice|tempo|palavra), incluindo algumas palavras de contexto antes e depois do clipe.

Sua tarefa:
1. start_i / end_i: confirme ou ajuste os limites do clipe (índices de palavra) para começar numa frase \
compreensível sem contexto e terminar fechando o raciocínio. Só mude se melhorar.
2. fixes: corrija erros da transcrição automática (palavras mal ouvidas, nomes próprios, acentuação, \
pontuação). Devolva apenas as palavras alteradas. Não reescreva o que foi dito, apenas corrija.
3. remove: intervalos de palavras a cortar para deixar o clipe mais dinâmico: vícios de linguagem \
("é...", "tipo", "né", "hum"), gaguejadas, repetições, frases começadas e abandonadas. Não corte nada que \
quebre o sentido. Seja conservador; na dúvida, mantenha.
4. emphasis: índices das palavras-chave que devem ganhar destaque visual na legenda (cerca de 1 a cada 8 \
palavras: números, palavras de impacto, o conceito central).
5. music_target: o clima da música de fundo ideal para este clipe, no modelo de Russell:
   arousal de 0 (calmo, contemplativo) a 1 (agitado, intenso) e valence de 0 (triste, sombrio, tenso)
   a 1 (alegre, otimista). moods: 2 a 4 palavras em inglês (ex.: epic, motivational, dark, chill,
   suspense, funny, emotional, aggressive, dreamy, confident). A música deve reforçar a emoção do
   conteúdo sem competir com a fala."""

EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "start_i": {"type": "integer"},
        "end_i": {"type": "integer"},
        "fixes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"i": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["i", "text"],
                "additionalProperties": False,
            },
        },
        "remove": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_i": {"type": "integer"},
                    "to_i": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["from_i", "to_i", "reason"],
                "additionalProperties": False,
            },
        },
        "emphasis": {"type": "array", "items": {"type": "integer"}},
        "music_target": {
            "type": "object",
            "properties": {
                "arousal": {"type": "number"},
                "valence": {"type": "number"},
                "moods": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["arousal", "valence", "moods", "reason"],
            "additionalProperties": False,
        },
    },
    "required": ["start_i", "end_i", "fixes", "remove", "emphasis", "music_target"],
    "additionalProperties": False,
}

CONTEXT_WORDS = 25


def edit_clip(words: list[Word], start_i: int, end_i: int, title: str) -> dict:
    lo = max(0, start_i - CONTEXT_WORDS)
    hi = min(len(words) - 1, end_i + CONTEXT_WORDS)
    listing = "\n".join(f"{w.i}|{_fmt(w.start)}|{w.text}" for w in words[lo : hi + 1])
    user = (
        f"Clipe: \"{title}\". Limites atuais: start_i={start_i}, end_i={end_i}. "
        f"Você pode ajustar os limites dentro de {lo}..{hi}.\n\n<palavras>\n{listing}\n</palavras>"
    )
    result = _ask_json(EDIT_SYSTEM, user, EDIT_SCHEMA)
    result["start_i"] = max(lo, min(result["start_i"], hi))
    result["end_i"] = max(result["start_i"] + 1, min(result["end_i"], hi))
    return result


# ---------------------------------------------------------------------------
# Música de fundo
# ---------------------------------------------------------------------------

TAG_SYSTEM = """Você é um curador musical para vídeos curtos. Para cada faixa você recebe o título e \
descritores calculados por análise de sinal (BPM, tom, energia em dB, brilho espectral, ataques por segundo, \
e uma estimativa algorítmica de arousal/valence de 0 a 1).

Use o que você sabe sobre a música (pelo título/artista) junto com os descritores para classificar:
- moods: 3 a 5 palavras em inglês (ex.: epic, motivational, dark, chill, suspense, funny, emotional, \
aggressive, dreamy, confident, romantic, nostalgic, hype)
- genre: gênero principal
- vocals: true se a faixa tem voz cantada/rap em boa parte dela
- arousal e valence (0 a 1) no modelo de Russell, corrigindo a estimativa algorítmica quando ela estiver \
claramente errada (ex.: BPM detectado pela metade)
- good_for: tipo de vídeo em que a faixa funciona bem de fundo (frase curta, em português)"""

TAG_SCHEMA = {
    "type": "object",
    "properties": {
        "tracks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "moods": {"type": "array", "items": {"type": "string"}},
                    "genre": {"type": "string"},
                    "vocals": {"type": "boolean"},
                    "arousal": {"type": "number"},
                    "valence": {"type": "number"},
                    "good_for": {"type": "string"},
                },
                "required": ["id", "moods", "genre", "vocals", "arousal", "valence", "good_for"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tracks"],
    "additionalProperties": False,
}


def tag_tracks(tracks: list[dict]) -> list[dict]:
    lines = [
        f"{t['id']} | {t['title']} | {t.get('artist', '')} | {t['bpm']} BPM | {t['key']} {t['mode']} | "
        f"{t['rms_db']} dB | brilho {t['brightness']:.0f} Hz | {t['onset_rate']} ataques/s | "
        f"arousal {t['algo_arousal']} | valence {t['algo_valence']}"
        for t in tracks
    ]
    user = "Classifique todas as faixas abaixo (id | título | artista | descritores):\n\n" + "\n".join(lines)
    return _ask_json(TAG_SYSTEM, user, TAG_SCHEMA, effort="medium")["tracks"]


CHOOSE_MUSIC_SYSTEM = """Você escolhe a música de fundo de cada clipe de um vídeo. Para cada clipe há \
candidatas já ranqueadas por um algoritmo (nota de compatibilidade de 0 a 1, baseada na distância no \
modelo arousal × valence, nos climas em comum, na presença de voz cantada e em repetição).

Escolha uma faixa por clipe entre as candidatas dele. Considere a nota, mas use seu julgamento sobre o \
tema: a música deve reforçar a emoção do conteúdo sem competir com a fala. Evite repetir a mesma faixa em \
clipes diferentes quando houver alternativa boa. volume: de 0.08 (bem discreto, para conteúdo sério ou \
falas rápidas) a 0.25 (mais presente, para conteúdo de energia alta). reason: uma frase em português."""

CHOOSE_MUSIC_SCHEMA = {
    "type": "object",
    "properties": {
        "choices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clip": {"type": "integer"},
                    "track_id": {"type": "string"},
                    "volume": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["clip", "track_id", "volume", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["choices"],
    "additionalProperties": False,
}


def choose_music(clips: list[dict]) -> list[dict]:
    """clips: [{clip, hook_title, text, target, candidates: [(track, score)]}]"""
    blocks = []
    for c in clips:
        cands = "\n".join(
            f"  - {t['id']} | nota {score} | {t['title']} | {', '.join(t.get('moods', []))} | "
            f"{t.get('genre', '')} | {t['bpm']} BPM | voz: {'sim' if t.get('vocals') else 'não'}"
            for t, score in c["candidates"]
        )
        target = c["target"]
        blocks.append(
            f"Clipe {c['clip']}: \"{c['hook_title']}\"\n"
            f"Clima pedido: arousal {target.get('arousal')}, valence {target.get('valence')}, "
            f"{', '.join(target.get('moods', []))} ({target.get('reason', '')})\n"
            f"Trecho: {c['text'][:600]}\nCandidatas:\n{cands}"
        )
    return _ask_json(CHOOSE_MUSIC_SYSTEM, "\n\n".join(blocks), CHOOSE_MUSIC_SCHEMA, effort="medium")["choices"]


# ---------------------------------------------------------------------------
# 3. Editor por chat (o Claude edita o documento do clipe usando ferramentas)
# ---------------------------------------------------------------------------

STYLE_PROPERTIES = {
    "font": {"type": "string", "description": "Nome da fonte instalada (ex.: Arial Black, Impact, Montserrat)"},
    "font_size": {"type": "integer", "description": "Tamanho em px num vídeo 1080x1920 (40-140)"},
    "primary_color": {
        "type": "string",
        "description": "Cor base do texto da legenda, #RRGGBB. 'Legenda verde' = mudar esta cor.",
    },
    "highlight_color": {
        "type": "string",
        "description": "Cor da palavra sendo falada e das palavras de destaque, #RRGGBB. "
        "Se o usuário pedir a legenda toda de uma cor, ajuste esta também para combinar.",
    },
    "outline_color": {"type": "string", "description": "Cor do contorno, #RRGGBB"},
    "outline_width": {"type": "integer", "description": "Espessura do contorno (0-12)"},
    "uppercase": {"type": "boolean"},
    "words_per_line": {"type": "integer", "description": "Palavras por bloco de legenda (1-7)"},
    "position": {"type": "string", "enum": ["top", "middle", "bottom"]},
    "animation": {"type": "string", "enum": ["highlight", "karaoke", "pop", "static"]},
    "show_emojis": {"type": "boolean"},
    "hook_title_enabled": {"type": "boolean"},
    "hook_title_seconds": {"type": "number"},
    "hook_title_bg": {"type": "string", "description": "#RRGGBB"},
    "hook_title_color": {"type": "string", "description": "#RRGGBB"},
}

CHAT_TOOLS = [
    {
        "name": "set_style",
        "description": "Altera o estilo da legenda e do título-gancho. Envie só os campos que mudam.",
        "input_schema": {"type": "object", "properties": STYLE_PROPERTIES, "additionalProperties": False},
    },
    {
        "name": "set_texts",
        "description": "Altera o título-gancho (aparece no vídeo), o título do post, a descrição ou as hashtags.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hook_title": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "hashtags": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "set_layout",
        "description": "face = recorte 9:16 que acompanha o rosto; center = recorte central; "
        "blur = vídeo inteiro com fundo desfocado; gameplay = jogo em cima e webcam do streamer embaixo "
        "(para lives/gameplay com a câmera num canto da tela).",
        "input_schema": {
            "type": "object",
            "properties": {"layout": {"type": "string", "enum": ["face", "center", "blur", "gameplay"]}},
            "required": ["layout"],
            "additionalProperties": False,
        },
    },
    {
        "name": "set_music",
        "description": "Troca, ajusta ou remove a música de fundo. track_id = id de uma faixa listada em "
        "<musicas>, ou 'none' para tirar a música. Envie só os campos que mudam. Sem offset, um algoritmo "
        "escolhe o trecho da música com energia adequada, alinhado à batida.",
        "input_schema": {
            "type": "object",
            "properties": {
                "track_id": {"type": "string"},
                "volume": {"type": "number", "description": "0.05 (bem baixo) a 0.4 (alto); padrão 0.15"},
                "ducking": {"type": "boolean", "description": "Abaixar a música quando alguém fala"},
                "offset": {"type": "number", "description": "Segundo da música em que o clipe começa"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "cut_words",
        "description": "Remove do vídeo o intervalo de palavras from_i..to_i (inclusive).",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_i": {"type": "integer"},
                "to_i": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["from_i", "to_i"],
            "additionalProperties": False,
        },
    },
    {
        "name": "restore_words",
        "description": "Desfaz cortes que cobrem o intervalo de palavras from_i..to_i.",
        "input_schema": {
            "type": "object",
            "properties": {"from_i": {"type": "integer"}, "to_i": {"type": "integer"}},
            "required": ["from_i", "to_i"],
            "additionalProperties": False,
        },
    },
    {
        "name": "set_bounds",
        "description": "Muda onde o clipe começa e termina, por índice de palavra do vídeo original. "
        "Pode usar palavras fora do clipe atual para estendê-lo.",
        "input_schema": {
            "type": "object",
            "properties": {"start_i": {"type": "integer"}, "end_i": {"type": "integer"}},
            "required": ["start_i", "end_i"],
            "additionalProperties": False,
        },
    },
    {
        "name": "edit_words",
        "description": "Edita palavras da legenda: texto, destaque e emoji. Envie só os campos que mudam.",
        "input_schema": {
            "type": "object",
            "properties": {
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "i": {"type": "integer"},
                            "text": {"type": "string"},
                            "emphasis": {"type": "boolean"},
                            "emoji": {"type": "string", "description": "Vazio remove o emoji"},
                        },
                        "required": ["i"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["changes"],
            "additionalProperties": False,
        },
    },
]

CHAT_SYSTEM = """Você é o editor de vídeo de um app que gera cortes verticais. O usuário pede mudanças em \
um clipe e você as aplica usando as ferramentas. Depois de aplicar, responda em uma ou duas frases curtas, \
em português, dizendo o que mudou. Se o pedido não for possível com as ferramentas, explique o que dá para \
fazer. Não invente palavras que não foram ditas: cortes e limites só usam palavras existentes.

O estado atual do clipe vem no formato:
- estilo e textos em JSON
- palavras: índice|início|fim|texto|flags, onde flags pode ter E (destaque), C (cortada) e um emoji
- contexto: palavras do vídeo original logo antes e depois do clipe, que podem ser usadas em set_bounds"""


def _clip_state(doc: ClipDocument, all_words: list[Word]) -> str:
    cut_ranges = [(c.start, c.end) for c in doc.cuts]

    def is_cut(w) -> bool:
        return any(s <= w.start and w.end <= e + 1e-3 for s, e in cut_ranges)

    lines = []
    for w in doc.words:
        flags = ("E" if w.emphasis else "") + ("C" if is_cut(w) else "") + (w.emoji or "")
        lines.append(f"{w.i}|{_fmt(w.start)}|{_fmt(w.end)}|{w.text}|{flags}")
    first = doc.words[0].i if doc.words else 0
    last = doc.words[-1].i if doc.words else 0
    before = all_words[max(0, first - 30) : first]
    after = all_words[last + 1 : last + 31]
    meta = {
        "duracao_segundos": round(doc.end - doc.start, 1),
        "hook_title": doc.hook_title,
        "title": doc.title,
        "description": doc.description,
        "hashtags": doc.hashtags,
        "layout": doc.layout,
        "style": doc.style.model_dump(),
        "musica": doc.music.model_dump() if doc.music else None,
    }
    tracks = "\n".join(
        f"{t['id']} | {t['title']} | {', '.join(t.get('moods', []))} | {t['bpm']} BPM | "
        f"voz: {'sim' if t.get('vocals') else 'não'}"
        for t in music.load_library() if "bpm" in t
    )
    ctx = lambda ws: " ".join(f"{w.i}:{w.text}" for w in ws)  # noqa: E731
    return (
        f"<estado>\n{json.dumps(meta, ensure_ascii=False, indent=1)}\n</estado>\n"
        f"<musicas>\n{tracks or '(biblioteca vazia)'}\n</musicas>\n"
        f"<palavras>\n" + "\n".join(lines) + "\n</palavras>\n"
        f"<contexto_antes>{ctx(before)}</contexto_antes>\n<contexto_depois>{ctx(after)}</contexto_depois>"
    )


class ClipEditor:
    """Aplica as chamadas de ferramenta do Claude ao documento do clipe."""

    def __init__(self, doc: ClipDocument, all_words: list[Word], rebuild_words: Callable[[ClipDocument, int, int], None]):
        self.doc = doc
        self.all_words = all_words
        self.rebuild_words = rebuild_words

    def _word(self, i: int) -> Word:
        if not 0 <= i < len(self.all_words):
            raise ValueError(f"Índice de palavra inválido: {i}")
        return self.all_words[i]

    def run(self, name: str, args: dict[str, Any]) -> str:
        doc = self.doc
        if name == "set_style":
            doc.style = CaptionStyle.model_validate({**doc.style.model_dump(), **args})
            return "Estilo atualizado."
        if name == "set_texts":
            for key, value in args.items():
                setattr(doc, key, value)
            return "Textos atualizados."
        if name == "set_layout":
            doc.layout = args["layout"]
            return f"Layout: {doc.layout}."
        if name == "set_music":
            return self._set_music(args)
        if name == "cut_words":
            a, b = sorted((args["from_i"], args["to_i"]))
            doc.cuts.append(Cut(start=self._word(a).start, end=self._word(b).end, reason=args.get("reason", "chat")))
            return f"Palavras {a}..{b} cortadas."
        if name == "restore_words":
            a, b = sorted((args["from_i"], args["to_i"]))
            s, e = self._word(a).start, self._word(b).end
            before = len(doc.cuts)
            doc.cuts = [c for c in doc.cuts if c.end <= s or c.start >= e]
            return f"{before - len(doc.cuts)} corte(s) desfeito(s)."
        if name == "set_bounds":
            a, b = sorted((args["start_i"], args["end_i"]))
            self.rebuild_words(doc, a, b)
            return f"Clipe agora vai de {doc.start:.2f}s a {doc.end:.2f}s ({doc.end - doc.start:.1f}s)."
        if name == "edit_words":
            by_index = {w.i: w for w in doc.words}
            changed = 0
            for change in args["changes"]:
                word = by_index.get(change["i"])
                if word is None:
                    continue
                for key in ("text", "emphasis", "emoji"):
                    if key in change:
                        setattr(word, key, change[key])
                changed += 1
            return f"{changed} palavra(s) editada(s)."
        raise ValueError(f"Ferramenta desconhecida: {name}")

    def _set_music(self, args: dict[str, Any]) -> str:
        doc = self.doc
        track_id = args.get("track_id")
        if track_id == "none":
            doc.music = None
            return "Música removida."
        if track_id and (doc.music is None or doc.music.track_id != track_id):
            track = music.get_track(track_id)
            if track is None or "bpm" not in track:
                raise ValueError(f"Faixa desconhecida: {track_id}")
            clip_seconds = sum(b - a for a, b in media.keep_ranges(doc))
            arousal = doc.music_target.get("arousal", music.track_mood(track)[0])
            doc.music = MusicChoice(
                track_id=track_id, title=track["title"],
                offset=music.best_offset(track, clip_seconds, arousal),
                volume=doc.music.volume if doc.music else 0.15, reason="escolhida no chat",
            )
        if doc.music is None:
            raise ValueError("O clipe não tem música; informe um track_id.")
        if "volume" in args:
            doc.music.volume = min(0.6, max(0.02, float(args["volume"])))
        if "ducking" in args:
            doc.music.ducking = bool(args["ducking"])
        if "offset" in args:
            doc.music.offset = max(0.0, float(args["offset"]))
        return f"Música: {doc.music.title} (volume {doc.music.volume:.2f}, começa em {doc.music.offset:.1f}s)."


MAX_CHAT_STEPS = 8


CHAT_OPS_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": [t["name"] for t in CHAT_TOOLS]},
                    "args_json": {"type": "string", "description": "Argumentos da operação como objeto JSON"},
                },
                "required": ["tool", "args_json"],
                "additionalProperties": False,
            },
        },
        "reply": {"type": "string"},
    },
    "required": ["operations", "reply"],
    "additionalProperties": False,
}


def _chat_ops_system() -> str:
    tools = "\n".join(
        f"- {t['name']}: {t['description']}\n  argumentos (JSON Schema): {json.dumps(t['input_schema'], ensure_ascii=False)}"
        for t in CHAT_TOOLS
    )
    return (
        CHAT_SYSTEM
        + "\n\nResponda com a lista de operações a aplicar, em ordem, e a resposta curta para o usuário. "
        "Em args_json escreva um objeto JSON com os argumentos da operação. Operações disponíveis:\n"
        + tools
    )


def chat_edit(editor: ClipEditor, instruction: str) -> str:
    if config.CLAUDE_BACKEND == "api":
        return _chat_edit_api(editor, instruction)
    return _chat_edit_claude_code(editor, instruction)


def _chat_edit_claude_code(editor: ClipEditor, instruction: str) -> str:
    """Uma chamada só: o Claude devolve as operações e nós as aplicamos ao clipe."""
    doc = editor.doc
    history = "\n".join(
        f"{'Usuário' if m.role == 'user' else 'Editor'}: {m.content}" for m in doc.chat[-10:]
    )
    user = (
        (f"<conversa_anterior>\n{history}\n</conversa_anterior>\n\n" if history else "")
        + f"{_clip_state(doc, editor.all_words)}\n\nPedido: {instruction}"
    )
    result = _ask_json(_chat_ops_system(), user, CHAT_OPS_SCHEMA, effort="medium")

    errors = []
    for op in result["operations"]:
        try:
            args = json.loads(op["args_json"] or "{}")
            if not isinstance(args, dict):
                raise ValueError("argumentos precisam ser um objeto JSON")
            editor.run(op["tool"], args)
        except Exception as exc:
            errors.append(f"{op['tool']}: {exc}")
    reply = result["reply"].strip() or "Feito."
    if errors:
        reply += "\n\n(Não consegui aplicar: " + "; ".join(errors) + ")"
    return reply


def _chat_edit_api(editor: ClipEditor, instruction: str) -> str:
    """Roda o loop de ferramentas até o Claude terminar; devolve a resposta em texto."""
    doc = editor.doc
    messages: list[dict[str, Any]] = []
    for past in doc.chat[-10:]:
        messages.append({"role": past.role, "content": past.content})
    messages.append(
        {"role": "user", "content": f"{_clip_state(doc, editor.all_words)}\n\nPedido: {instruction}"}
    )

    for _ in range(MAX_CHAT_STEPS):
        response = client().beta.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=16000,
            system=CHAT_SYSTEM,
            tools=CHAT_TOOLS,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return "Não consegui processar esse pedido."
        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            return "".join(b.text for b in response.content if b.type == "text").strip() or "Feito."

        results = []
        for block in tool_uses:
            try:
                output = editor.run(block.name, dict(block.input))
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
            except Exception as exc:  # erro de validação volta para o Claude corrigir
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": str(exc), "is_error": True}
                )
        messages.append({"role": "user", "content": results})

    return "Apliquei as mudanças possíveis."
