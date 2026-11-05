"""Orquestra o processamento: download → transcrição → Claude escolhe → Claude edita → render."""

import logging
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import claude_ai, config, media, music, store, transcribe
from .download import download
from .models import CAPTION_PRESETS, CaptionWord, ChatMessage, ClipDocument, Cut, MusicChoice, Virality, Word

logger = logging.getLogger(__name__)

PAUSE_THRESHOLD_SECONDS = 0.6
PAUSE_KEEP_SECONDS = 0.15

job_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")
render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
music_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="music")


def _transcript_words(job_id: str) -> list[Word]:
    data = transcribe.transcribe(store.find_source(job_id), store.job_dir(job_id))
    return transcribe.load_words(data)


def set_clip_bounds(doc: ClipDocument, all_words: list[Word], start_i: int, end_i: int) -> None:
    """Redefine as palavras do clipe preservando correções/destaques já feitos."""
    start_i = max(0, min(start_i, len(all_words) - 1))
    end_i = max(start_i, min(end_i, len(all_words) - 1))
    previous = {w.i: w for w in doc.words}
    doc.words = [
        previous.get(w.i) or CaptionWord(i=w.i, text=w.text, start=w.start, end=w.end)
        for w in all_words[start_i : end_i + 1]
    ]
    doc.start = all_words[start_i].start
    doc.end = all_words[end_i].end
    doc.cuts = [c for c in doc.cuts if c.end > doc.start and c.start < doc.end]
    doc.face_track = None  # o enquadramento é recalculado para o novo trecho
    doc.cam_box = None


def silence_cuts(doc: ClipDocument) -> list[Cut]:
    cuts = []
    for cur, nxt in zip(doc.words, doc.words[1:]):
        if nxt.start - cur.end > PAUSE_THRESHOLD_SECONDS:
            cuts.append(Cut(start=cur.end + PAUSE_KEEP_SECONDS, end=nxt.start - PAUSE_KEEP_SECONDS, reason="pausa"))
    return cuts


def _apply_edit(doc: ClipDocument, all_words: list[Word], edit: dict, remove_silences: bool) -> None:
    set_clip_bounds(doc, all_words, edit["start_i"], edit["end_i"])
    by_index = {w.i: w for w in doc.words}
    for fix in edit["fixes"]:
        if fix["i"] in by_index and fix["text"].strip():
            by_index[fix["i"]].text = fix["text"].strip()
    for i in edit["emphasis"]:
        if i in by_index:
            by_index[i].emphasis = True
    for rem in edit["remove"]:
        a, b = sorted((rem["from_i"], rem["to_i"]))
        if a in by_index and b in by_index and not (a == doc.words[0].i and b == doc.words[-1].i):
            doc.cuts.append(Cut(start=by_index[a].start, end=by_index[b].end, reason=rem["reason"]))
    if remove_silences:
        doc.cuts.extend(silence_cuts(doc))


def assign_music(docs: list[ClipDocument]) -> None:
    """Algoritmo ranqueia as faixas para cada clipe; o Claude escolhe entre as melhores."""
    library = [t for t in music.load_library() if "bpm" in t]
    if not library:
        return
    requests = []
    for doc in docs:
        target = doc.music_target or {"arousal": 0.5, "valence": 0.5, "moods": []}
        text = " ".join(w.text for w in doc.words)
        requests.append({
            "clip": doc.index, "hook_title": doc.hook_title, "text": text, "target": target,
            "candidates": music.rank_tracks(target, used=set()),
        })

    try:
        choices = {c["clip"]: c for c in claude_ai.choose_music(requests)}
    except Exception:
        logger.exception("Claude não escolheu as músicas; usando só o algoritmo")
        choices = {}

    used: set[str] = set()
    for doc, req in zip(docs, requests):
        choice = choices.get(doc.index)
        track = music.get_track(choice["track_id"]) if choice else None
        if track is None:  # sem resposta válida do Claude: melhor nota, sem repetir
            track, _ = music.rank_tracks(req["target"], used, limit=1)[0]
            choice = {"volume": 0.15, "reason": "melhor nota no algoritmo de compatibilidade"}
        used.add(track["id"])
        clip_seconds = sum(b - a for a, b in media.keep_ranges(doc))
        doc.music = MusicChoice(
            track_id=track["id"], title=track["title"],
            offset=music.best_offset(track, clip_seconds, req["target"].get("arousal", 0.5)),
            volume=min(0.4, max(0.05, float(choice["volume"]))), reason=choice["reason"],
        )


def render(job_id: str, clip_id: str) -> None:
    """Renderiza (ou re-renderiza) um clipe. Roda no render_executor."""
    doc = store.load_clip(job_id, clip_id)
    doc.render.status = "rendering"
    doc.render.error = ""
    store.save_clip(doc)
    try:
        source = store.find_source(job_id)
        info = media.probe(source)
        if doc.layout == "gameplay" and doc.cam_box is None:
            doc.cam_box = media.detect_cam_box(source, doc.start, doc.end)
        # Sem webcam encontrada, o layout gameplay vira "seguir rosto".
        if (doc.layout == "face" or (doc.layout == "gameplay" and not doc.cam_box)) and doc.face_track is None:
            doc.face_track = media.detect_face_track(source, doc.start, doc.end)
        output = media.render_clip(doc, source, info, store.clips_dir(job_id))
        latest = store.load_clip(job_id, clip_id)  # o usuário pode ter editado durante o render
        latest.render = doc.render
        latest.face_track = doc.face_track
        latest.cam_box = doc.cam_box
        latest.render.status = "done"
        latest.render.file = output.name
        store.save_clip(latest)
    except Exception as exc:
        logger.exception("Falha ao renderizar %s", clip_id)
        doc.render.status = "error"
        doc.render.error = str(exc)
        store.save_clip(doc)


def queue_render(job_id: str, clip_id: str) -> None:
    doc = store.load_clip(job_id, clip_id)
    doc.render.status = "pending"
    store.save_clip(doc)
    render_executor.submit(render, job_id, clip_id)


def process_job(job_id: str) -> None:
    job = store.update_job(job_id, status="running", stage="preparando", progress=0.01)
    settings = job.settings
    work = store.job_dir(job_id)
    try:
        if job.source_url:
            store.update_job(job_id, stage="baixando vídeo")
            _, title = download(
                job.source_url, work, lambda f: store.update_job(job_id, progress=0.01 + 0.09 * f)
            )
            store.update_job(job_id, source_name=title)
        source = store.find_source(job_id)
        info = media.probe(source)
        store.update_job(job_id, duration=info.duration, stage="transcrevendo áudio", progress=0.1)
        if not info.has_audio:
            raise RuntimeError("O vídeo não tem áudio para transcrever.")

        data = transcribe.transcribe(
            source, work, settings.language,
            lambda f: store.update_job(job_id, progress=0.1 + 0.35 * f),
        )
        words = transcribe.load_words(data)
        if len(words) < 20:
            raise RuntimeError("A transcrição ficou curta demais para gerar cortes.")
        store.update_job(job_id, language=data["language"], stage="Claude escolhendo os melhores momentos", progress=0.47)

        selection = claude_ai.select_clips(words, data["segments"], info.duration, settings)
        if not selection["clips"]:
            raise RuntimeError("O Claude não encontrou trechos que funcionem como clipe.")
        store.update_job(job_id, summary=selection["summary"], stage="Claude editando e legendando os cortes", progress=0.55)

        def edit(clip: dict) -> dict:
            return claude_ai.edit_clip(words, clip["start_i"], clip["end_i"], clip["hook_title"])

        # No modo Claude Code cada edição é um processo `claude`; menos paralelismo poupa a cota da assinatura.
        workers = 4 if config.CLAUDE_BACKEND == "api" else 2
        with ThreadPoolExecutor(max_workers=workers) as pool:
            edits = list(pool.map(edit, selection["clips"]))

        style = CAPTION_PRESETS.get(settings.caption_preset, CAPTION_PRESETS["classico"])
        clip_ids: list[str] = []
        docs: list[ClipDocument] = []
        for index, (clip, edit_result) in enumerate(zip(selection["clips"], edits)):
            hook, engagement, value, share = clip["scores"]
            doc = ClipDocument(
                id=uuid.uuid4().hex[:10], job_id=job_id, index=index,
                start=clip["start"], end=clip["end"],
                title=clip["title"], hook_title=clip["hook_title"],
                description=clip["description"], hashtags=[h.lstrip("#") for h in clip["hashtags"]],
                virality=Virality(
                    hook=hook, engagement=engagement, value=value, shareability=share,
                    total=clip["total"], hook_type=clip["hook_type"], reasoning=clip["reasoning"],
                ),
                style=style.model_copy(deep=True), layout=settings.layout,
            )
            _apply_edit(doc, words, edit_result, settings.remove_silences)
            doc.music_target = edit_result.get("music_target", {})
            docs.append(doc)

        if settings.add_music:
            store.update_job(job_id, stage="escolhendo músicas de fundo")
            assign_music(docs)
        for doc in docs:
            store.save_clip(doc)
            clip_ids.append(doc.id)
        store.update_job(job_id, clip_ids=clip_ids, stage="renderizando clipes", progress=0.6)

        for n, clip_id in enumerate(clip_ids):
            store.update_job(job_id, stage=f"renderizando clipe {n + 1} de {len(clip_ids)}")
            render(job_id, clip_id)
            store.update_job(job_id, progress=0.6 + 0.4 * (n + 1) / len(clip_ids))

        store.update_job(job_id, status="done", stage="pronto", progress=1.0)
    except Exception as exc:
        logger.error("Job %s falhou:\n%s", job_id, traceback.format_exc())
        store.update_job(job_id, status="error", error=str(exc), stage="erro")


def chat(job_id: str, clip_id: str, message: str) -> tuple[str, ClipDocument]:
    """O usuário pede uma mudança; o Claude edita o clipe e um novo render entra na fila."""
    words = _transcript_words(job_id)
    doc = store.load_clip(job_id, clip_id)
    editor = claude_ai.ClipEditor(doc, words, lambda d, a, b: set_clip_bounds(d, words, a, b))
    reply = claude_ai.chat_edit(editor, message)
    doc.chat.append(ChatMessage(role="user", content=message))
    doc.chat.append(ChatMessage(role="assistant", content=reply))
    doc = ClipDocument.model_validate(doc.model_dump())  # revalida o que as ferramentas mudaram
    store.save_clip(doc)
    queue_render(job_id, clip_id)
    return reply, store.load_clip(job_id, clip_id)
