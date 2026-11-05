"""API HTTP + frontend estático. Rode com: uvicorn app.main:app (dentro de backend/)."""

import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, music, pipeline, store
from .models import CAPTION_PRESETS, CaptionStyle, Job, JobSettings, LayoutMode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ALLOWED_UPLOADS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.recover_interrupted()
    music.recover_interrupted()
    yield


app = FastAPI(title="CorteVideos", lifespan=lifespan)


def _job_or_404(job_id: str) -> Job:
    try:
        return store.load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, "Job não encontrado")


def _clip_or_404(job_id: str, clip_id: str):
    try:
        return store.load_clip(job_id, clip_id)
    except FileNotFoundError:
        raise HTTPException(404, "Clipe não encontrado")


@app.get("/api/presets")
def presets():
    return {name: style.model_dump() for name, style in CAPTION_PRESETS.items()}


@app.post("/api/jobs")
async def create_job(
    url: str = Form(""),
    file: Optional[UploadFile] = File(None),
    num_clips: int = Form(5),
    clip_length: str = Form("medio"),
    language: str = Form(""),
    caption_preset: str = Form("classico"),
    layout: str = Form("face"),
    remove_silences: bool = Form(True),
    add_music: bool = Form(True),
    instructions: str = Form(""),
):
    if not url.strip() and not (file and file.filename):
        raise HTTPException(400, "Envie uma URL ou um arquivo de vídeo.")
    settings = JobSettings(
        num_clips=max(1, min(num_clips, 15)), clip_length=clip_length, language=language.strip(),
        caption_preset=caption_preset, layout=layout, remove_silences=remove_silences,
        add_music=add_music, instructions=instructions,
    )
    job = Job(id=uuid.uuid4().hex[:12], created_at=time.time(), source_url=url.strip(), settings=settings)

    if file and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_UPLOADS:
            raise HTTPException(400, f"Formato não suportado: {suffix}")
        job.source_url = ""
        job.source_name = file.filename
        store.job_dir(job.id).mkdir(parents=True, exist_ok=True)
        with open(store.job_dir(job.id) / f"source{suffix}", "wb") as out:
            await run_in_threadpool(shutil.copyfileobj, file.file, out, 1024 * 1024)

    store.save_job(job)
    pipeline.job_executor.submit(pipeline.process_job, job.id)
    return job


@app.get("/api/jobs")
def list_jobs():
    return [{**job.model_dump(), "size": store.job_size(job.id)} for job in store.list_jobs()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _job_or_404(job_id)
    clips = []
    for clip_id in job.clip_ids:
        try:
            clips.append(store.load_clip(job_id, clip_id))
        except FileNotFoundError:
            continue
    return {"job": job, "clips": sorted(clips, key=lambda c: c.index)}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = _job_or_404(job_id)
    if job.status in ("running", "queued"):
        raise HTTPException(409, "Espere o processamento terminar.")
    store.delete_job(job_id)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/clips/{clip_id}")
def get_clip(job_id: str, clip_id: str):
    return _clip_or_404(job_id, clip_id)


class ChatRequest(BaseModel):
    message: str


@app.post("/api/jobs/{job_id}/clips/{clip_id}/chat")
async def chat(job_id: str, clip_id: str, body: ChatRequest):
    _clip_or_404(job_id, clip_id)
    if not body.message.strip():
        raise HTTPException(400, "Mensagem vazia.")
    reply, doc = await run_in_threadpool(pipeline.chat, job_id, clip_id, body.message.strip())
    return {"reply": reply, "clip": doc}


class ManualEdit(BaseModel):
    style: Optional[CaptionStyle] = None
    hook_title: Optional[str] = None
    layout: Optional[LayoutMode] = None


@app.patch("/api/jobs/{job_id}/clips/{clip_id}")
def patch_clip(job_id: str, clip_id: str, body: ManualEdit):
    doc = _clip_or_404(job_id, clip_id)
    if body.style is not None:
        doc.style = body.style
    if body.hook_title is not None:
        doc.hook_title = body.hook_title
    if body.layout is not None:
        doc.layout = body.layout
    store.save_clip(doc)
    pipeline.queue_render(job_id, clip_id)
    return store.load_clip(job_id, clip_id)


@app.post("/api/jobs/{job_id}/clips/{clip_id}/render")
def rerender(job_id: str, clip_id: str):
    _clip_or_404(job_id, clip_id)
    pipeline.queue_render(job_id, clip_id)
    return {"ok": True}


MUSIC_PUBLIC_FIELDS = (
    "id", "title", "artist", "duration", "bpm", "key", "mode", "moods", "genre", "vocals", "good_for", "source_url",
)


@app.get("/api/music")
def music_library():
    tracks = []
    for t in music.load_library():
        item = {k: t.get(k) for k in MUSIC_PUBLIC_FIELDS}
        item["arousal"], item["valence"] = (round(x, 2) for x in music.track_mood(t))
        tracks.append(item)
    return {"state": music.import_state(), "tracks": tracks}


class MusicImport(BaseModel):
    url: str


@app.post("/api/music/import")
def music_import(body: MusicImport):
    if music.import_state().get("status") == "running":
        raise HTTPException(409, "Já existe uma importação em andamento.")
    if not body.url.strip():
        raise HTTPException(400, "Informe o link da playlist ou do vídeo.")
    music._set_state(status="running", progress=0.0, message="na fila")
    pipeline.music_executor.submit(music.import_playlist, body.url.strip())
    return {"ok": True}


@app.delete("/api/music/{track_id}")
def music_delete(track_id: str):
    music.delete_track(track_id)
    return {"ok": True}


@app.get("/media/{job_id}/{filename}")
def media(job_id: str, filename: str):
    path = (store.clips_dir(job_id) / filename).resolve()
    if path.parent != store.clips_dir(job_id).resolve() or path.suffix != ".mp4" or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4")


app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
