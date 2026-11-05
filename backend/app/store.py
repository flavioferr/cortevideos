"""Persistência simples em disco: data/jobs/<job>/job.json e clips/<clip>.json."""

import shutil
import threading
import time
from pathlib import Path

from . import config
from .models import ClipDocument, Job

_lock = threading.RLock()


def job_dir(job_id: str) -> Path:
    return config.JOBS_DIR / job_id


def clips_dir(job_id: str) -> Path:
    path = job_dir(job_id) / "clips"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    # No Windows, OneDrive/antivírus podem segurar o arquivo por um instante ("Acesso negado").
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def save_job(job: Job) -> None:
    with _lock:
        job_dir(job.id).mkdir(parents=True, exist_ok=True)
        _write(job_dir(job.id) / "job.json", job.model_dump_json(indent=1))


def load_job(job_id: str) -> Job:
    with _lock:
        return Job.model_validate_json((job_dir(job_id) / "job.json").read_text(encoding="utf-8"))


def update_job(job_id: str, **fields) -> Job:
    with _lock:
        job = load_job(job_id)
        for key, value in fields.items():
            setattr(job, key, value)
        save_job(job)
        return job


def list_jobs() -> list[Job]:
    jobs = []
    for path in config.JOBS_DIR.glob("*/job.json"):
        try:
            jobs.append(load_job(path.parent.name))
        except Exception:
            continue
    return sorted(jobs, key=lambda j: j.created_at, reverse=True)


def job_size(job_id: str) -> int:
    """Bytes ocupados pelo projeto (vídeo original, áudio, clipes)."""
    return sum(f.stat().st_size for f in job_dir(job_id).rglob("*") if f.is_file())


def delete_job(job_id: str) -> None:
    with _lock:
        shutil.rmtree(job_dir(job_id), ignore_errors=True)


def save_clip(doc: ClipDocument) -> None:
    with _lock:
        _write(clips_dir(doc.job_id) / f"{doc.id}.json", doc.model_dump_json(indent=1))


def load_clip(job_id: str, clip_id: str) -> ClipDocument:
    with _lock:
        return ClipDocument.model_validate_json((clips_dir(job_id) / f"{clip_id}.json").read_text(encoding="utf-8"))


def find_source(job_id: str) -> Path:
    files = [p for p in job_dir(job_id).glob("source.*") if p.suffix not in (".part", ".ytdl", ".tmp")]
    if not files:
        raise FileNotFoundError("Vídeo original não encontrado.")
    return files[0]


def recover_interrupted() -> None:
    """Jobs/renders que estavam rodando quando o servidor caiu viram erro."""
    for job in list_jobs():
        if job.status in ("queued", "running"):
            update_job(job.id, status="error", error="Processamento interrompido (servidor reiniciado).")
        for clip_id in job.clip_ids:
            try:
                doc = load_clip(job.id, clip_id)
            except Exception:
                continue
            if doc.render.status in ("pending", "rendering"):
                doc.render.status = "error"
                doc.render.error = "Renderização interrompida."
                save_clip(doc)
