"""Análise do vídeo (tamanho, rosto) e renderização dos clipes com FFmpeg."""

import logging
import os
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config, music
from .captions import build_ass
from .models import ClipDocument

logger = logging.getLogger(__name__)

MIN_RANGE_SECONDS = 0.08


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    has_audio: bool


def probe(video: Path) -> VideoInfo:
    proc = subprocess.run([config.ffmpeg_exe(), "-hide_banner", "-i", str(video)], capture_output=True, text=True, errors="replace")
    out = proc.stderr
    duration = 0.0
    if m := re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", out):
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"Stream #.*Video:.*?(\d{2,5})x(\d{2,5})", out)
    if not m:
        raise RuntimeError("Não encontrei uma trilha de vídeo no arquivo.")
    width, height = int(m.group(1)), int(m.group(2))
    rotation = re.search(r"rotation of (-?\d+(?:\.\d+)?)", out) or re.search(r"rotate\s*:\s*(-?\d+)", out)
    if rotation and abs(round(float(rotation.group(1)))) % 180 == 90:
        width, height = height, width
    return VideoInfo(width, height, duration, "Audio:" in out)


FACE_SAMPLE_SECONDS = 0.2  # 5 detecções por segundo
CAMERA_SMOOTHING_SECONDS = 0.4  # constante de tempo da panorâmica
DEADZONE = 0.15  # fração do quadro que o rosto pode andar sem mover a câmera
SWITCH_JUMP = 0.6  # salto maior que isso (fração do quadro) vira corte seco


def _find_faces(cascades, gray) -> list:
    import cv2

    frontal, profile = cascades
    faces = list(frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40)))
    if not faces:  # rosto virado: tenta perfil dos dois lados
        w = gray.shape[1]
        faces = list(profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40)))
        flipped = profile.detectMultiScale(cv2.flip(gray, 1), scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
        faces += [(w - x - fw, y, fw, fh) for x, y, fw, fh in flipped]
    return faces


def _smooth(xs: list[float], dts: list[float]) -> list[float]:
    """Média exponencial ida e volta: suaviza sem atrasar a câmera."""
    out = list(xs)
    for k in range(1, len(out)):
        alpha = dts[k] / (CAMERA_SMOOTHING_SECONDS + dts[k])
        out[k] = out[k - 1] + alpha * (out[k] - out[k - 1])
    for k in range(len(out) - 2, -1, -1):
        alpha = dts[k + 1] / (CAMERA_SMOOTHING_SECONDS + dts[k + 1])
        out[k] = out[k + 1] + alpha * (out[k] - out[k + 1])
    return out


def _simplify(points: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker: mantém só os pontos onde a câmera muda de direção."""
    if len(points) < 3:
        return points
    (t0, x0), (t1, x1) = points[0], points[-1]
    worst, worst_k = 0.0, 0
    for k in range(1, len(points) - 1):
        t, x = points[k]
        dev = abs(x - (x0 + (x1 - x0) * (t - t0) / (t1 - t0)))
        if dev > worst:
            worst, worst_k = dev, k
    if worst <= tol:
        return [points[0], points[-1]]
    return _simplify(points[: worst_k + 1], tol)[:-1] + _simplify(points[worst_k:], tol)


def detect_face_track(video: Path, start: float, end: float) -> list[list[float]]:
    """Trajetória da câmera virtual: [[segundo no original, centro x 0..1], ...]; vazia se não houver rosto."""
    import cv2

    cascades = (
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"),
    )
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, round(fps * FACE_SAMPLE_SECONDS))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    times: list[float] = []
    detections: list[list[tuple[float, float]]] = []  # (centro x, área) de cada rosto
    crop_frac = 1.0
    frame_i = 0
    try:
        while start + frame_i / fps < end:
            t = start + frame_i / fps
            frame_i += 1
            if (frame_i - 1) % every:
                if not cap.grab():  # leitura sequencial: bem mais rápida que um seek por amostra
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            crop_frac = min(1.0, h * config.OUTPUT_WIDTH / config.OUTPUT_HEIGHT / w)
            scale = 640 / w if w > 640 else 1.0
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            sw = small.shape[1]
            times.append(t)
            detections.append([((x + fw / 2) / sw, fw * fh) for x, y, fw, fh in _find_faces(cascades, gray)])
    finally:
        cap.release()

    if sum(1 for d in detections if d) < 3:
        return []

    # Segue a mesma pessoa: fica no rosto mais próximo, a não ser que outro seja bem maior.
    xs: list[Optional[float]] = []
    cur = None
    for faces in detections:
        if not faces:
            xs.append(None)
            continue
        biggest = max(faces, key=lambda f: f[1])
        pick = biggest
        if cur is not None:
            near = min(faces, key=lambda f: abs(f[0] - cur))
            if abs(near[0] - cur) < crop_frac / 2 and near[1] * 1.6 >= biggest[1]:
                pick = near
        cur = pick[0]
        xs.append(cur)

    # Sem detecção: mantém a última posição (ou a primeira conhecida, no começo).
    last = next(x for x in xs if x is not None)
    filled = []
    for x in xs:
        last = x if x is not None else last
        filled.append(last)
    # Mediana móvel de 5 amostras (1 s) elimina detecções falsas isoladas.
    filled = [statistics.median(filled[max(0, k - 2) : k + 3]) for k in range(len(filled))]

    # Zona morta: a câmera só se move quando o rosto sai do miolo do quadro.
    segments: list[list[int]] = [[0]]
    targets, cam = [filled[0]], filled[0]
    for k in range(1, len(filled)):
        x = filled[k]
        if abs(x - cam) > SWITCH_JUMP * crop_frac:
            segments.append([])  # trocou de pessoa ou de câmera: corte seco
            cam = x
        elif abs(x - cam) > DEADZONE * crop_frac:
            cam = x
        targets.append(cam)
        segments[-1].append(k)

    half = crop_frac / 2
    points: list[tuple[float, float]] = []
    for seg in segments:
        dts = [FACE_SAMPLE_SECONDS] + [times[k] - times[k - 1] for k in seg[1:]]
        smooth = _smooth([targets[k] for k in seg], dts)
        seg_points = [(times[k], min(max(x, half), 1 - half)) for k, x in zip(seg, smooth)]
        if points:  # segura a posição anterior até o instante do corte
            points.append((seg_points[0][0] - 0.001, points[-1][1]))
        points += _simplify(seg_points, 0.002)
    return [[round(float(t), 3), round(float(x), 4)] for t, x in points]


CAM_SAMPLES = 40
CAM_ANALYSIS_WIDTH = 640


def _border(profile, lo: int, hi: int, from_end: bool) -> Optional[int]:
    """Linha contínua (borda da moldura da webcam) em profile[lo:hi] mais perto da borda do retângulo."""
    rows = range(hi - 1, lo - 1, -1) if from_end else range(lo, hi)
    return next((k for k in rows if profile[k] >= 0.5), None)


def _band_with(pos: float, profile, offset: int) -> tuple[int, int]:
    """Divide o bloco nas linhas que o atravessam por inteiro e devolve (início, tamanho) da faixa com pos."""
    n = len(profile)
    edge = max(4, n // 10)
    cuts = [k for k in range(edge, n - edge) if profile[k] >= 0.85]
    bounds = [0, *cuts, n]
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 4 and offset + a <= pos < offset + b:
            return offset + a, b - a
    return offset, n


def detect_cam_box(video: Path, start: float, end: float) -> list[float]:
    """Retângulo da webcam sobreposta ao jogo, [x, y, w, h] em fração do quadro; vazio se não achar.

    A webcam é a região onde as bordas ficam paradas (moldura, quarto do streamer) enquanto o jogo
    em volta muda: o percentil 25 da magnitude de borda ao longo do tempo isola essas regiões. Entre
    elas, fica a que contém o rosto e os lados são ajustados às linhas retas da moldura.
    """
    import cv2
    import numpy as np

    # O ffmpeg extrai as amostras já reduzidas e em cinza: bem mais rápido que um seek por quadro no OpenCV.
    info = probe(video)
    W, H = CAM_ANALYSIS_WIDTH, _even(CAM_ANALYSIS_WIDTH * info.height / info.width)
    duration = max(1.0, end - start)
    proc = subprocess.run(
        [
            config.ffmpeg_exe(), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(video),
            "-vf", f"fps={CAM_SAMPLES / duration:.5f},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ],
        capture_output=True,
    )
    grays = np.frombuffer(proc.stdout, np.uint8)[: len(proc.stdout) // (W * H) * W * H].reshape(-1, H, W)
    if len(grays) < 8:
        return []

    gx = np.stack([np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) for g in grays])
    gy = np.stack([np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)) for g in grays])
    still_x = np.percentile(gx, 25, axis=0)
    still_y = np.percentile(gy, 25, axis=0)
    dense = cv2.blur(((still_x + still_y) > 25).astype(np.float32), (15, 15)) > 0.12
    dense = cv2.morphologyEx(dense.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dense)

    def plausible(i: int) -> bool:
        x, y, w, h, area = stats[i]
        return 0.02 <= w * h / (W * H) <= 0.4 and 0.5 <= w / h <= 3.0 and area / (w * h) > 0.6

    candidates = [i for i in range(1, num) if plausible(i)]
    if not candidates:
        return []
    cascades = (
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"),
    )
    # Procura rosto dentro de cada candidato ampliado 2x: na imagem reduzida o rosto da webcam é pequeno.
    faces: dict[int, list[tuple[float, float]]] = {i: [] for i in candidates}
    for i in candidates:
        x, y, w, h, _ = stats[i]
        for g in grays:
            crop = cv2.resize(g[y : y + h, x : x + w], (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            faces[i] += [(x + (fx + fw / 2) / 2, y + (fy + fh / 2) / 2)
                         for fx, fy, fw, fh in _find_faces(cascades, cv2.equalizeHist(crop))]
    best = max(candidates, key=lambda i: (len(faces[i]), stats[i][4]))
    if len(faces[best]) < 2:
        return []  # nenhuma região parada com rosto: chat, logo ou placar, não uma webcam

    # Chat ou placar colado na webcam vira um bloco só; uma linha reta atravessando o bloco inteiro
    # separa os dois, e fica a parte onde o rosto aparece.
    x, y, w, h, _ = stats[best]
    face_x = statistics.median(f[0] for f in faces[best])
    face_y = statistics.median(f[1] for f in faces[best])
    y, h = _band_with(face_y, (still_y[y : y + h, x : x + w] > 25).mean(axis=1), y)
    x, w = _band_with(face_x, (still_x[y : y + h, x : x + w] > 25).mean(axis=0), x)

    # Ajusta cada lado à linha reta da moldura (o retângulo pode ter pego um texto vizinho).
    line_y = (still_y[:, x : x + w] > 25).mean(axis=1)
    line_x = (still_x[y : y + h, :] > 25).mean(axis=0)
    margin_y, margin_x = int(h * 0.35), int(w * 0.35)
    top = _border(line_y, max(0, y - 3), y + margin_y, from_end=False)
    bottom = _border(line_y, y + h - margin_y, min(H, y + h + 3), from_end=True)
    left = _border(line_x, max(0, x - 3), x + margin_x, from_end=False)
    right = _border(line_x, x + w - margin_x, min(W, x + w + 3), from_end=True)
    x0, y0 = (left + 1 if left is not None else x), (top + 1 if top is not None else y)
    x1, y1 = (right if right is not None else x + w), (bottom if bottom is not None else y + h)
    if x1 - x0 < w * 0.5 or y1 - y0 < h * 0.5:
        x0, y0, x1, y1 = x, y, x + w, y + h
    return [round(float(v), 4) for v in (x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H)]


def gameplay_split(doc: ClipDocument, info: VideoInfo) -> Optional[dict]:
    """Geometria do layout jogo em cima / webcam embaixo, ou None se a webcam não foi encontrada."""
    if doc.layout != "gameplay" or not doc.cam_box:
        return None
    W, H = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT
    bx, by, bw, bh = doc.cam_box
    cam_x, cam_y, cam_w, cam_h = bx * info.width, by * info.height, bw * info.width, bh * info.height
    bottom_h = _even(min(max(W * cam_h / cam_w, H * 0.3), H * 0.45))
    top_h = H - bottom_h

    # Webcam: recorta para a proporção do painel de baixo, sem esticar.
    target = W / bottom_h
    if cam_w / cam_h > target:
        cw, ch = cam_h * target, cam_h
    else:
        cw, ch = cam_w, cam_w / target
    cam = (_even(cw), _even(ch), int(cam_x + (cam_w - cw) / 2), int(cam_y + (cam_h - ch) / 2))

    # Jogo: recorte central na proporção do painel de cima, fugindo da webcam se ela estiver no caminho.
    target = W / top_h
    gw = min(info.width, info.height * target)
    gh = gw / target
    gx = (info.width - gw) / 2
    if cam_x + cam_w / 2 > info.width / 2:
        gx = max(0.0, min(gx, cam_x - gw))
    else:
        gx = min(info.width - gw, max(gx, cam_x + cam_w))
    game = (_even(gw), _even(gh), int(gx), int((info.height - gh) / 2))
    return {"top_h": top_h, "bottom_h": bottom_h, "game": game, "cam": cam}


def _track_at(track: list[list[float]], t: float) -> float:
    if t <= track[0][0]:
        return track[0][1]
    for (t0, x0), (t1, x1) in zip(track, track[1:]):
        if t <= t1:
            return x0 if t1 == t0 else x0 + (x1 - x0) * (t - t0) / (t1 - t0)
    return track[-1][1]


def _crop_x_expr(track: list[list[float]], ranges: list[tuple[float, float]], width: int, crop_w: int) -> str:
    """Posição do corte em função do tempo do clipe (já sem os trechos removidos), linear por partes."""
    pts: list[tuple[float, float]] = []
    offset = 0.0
    for a, b in ranges:
        inner = [(t, x) for t, x in track if a < t < b]
        for t, x in [(a, _track_at(track, a)), *inner, (b, _track_at(track, b))]:
            pts.append((offset + t - a, x * width - crop_w / 2))
        offset += b - a
    terms = []
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        if t1 - t0 < 1e-6:
            continue
        slope = (p1 - p0) / (t1 - t0)
        terms.append(f"gte(t,{t0:.3f})*lt(t,{t1:.3f})*({p0:.1f}+{slope:.2f}*(t-{t0:.3f}))")
    terms.append(f"gte(t,{pts[-1][0]:.3f})*{pts[-1][1]:.1f}")
    return f"clip({'+'.join(terms)},0,{width - crop_w})"


def keep_ranges(doc: ClipDocument) -> list[tuple[float, float]]:
    """Trechos do original que entram no clipe: [start, end] menos os cortes."""
    ranges = [(doc.start, doc.end)]
    for cut in sorted(doc.cuts, key=lambda c: c.start):
        next_ranges = []
        for a, b in ranges:
            if cut.end <= a or cut.start >= b:
                next_ranges.append((a, b))
                continue
            if cut.start > a:
                next_ranges.append((a, cut.start))
            if cut.end < b:
                next_ranges.append((cut.end, b))
        ranges = next_ranges
    return [(a, b) for a, b in ranges if b - a >= MIN_RANGE_SECONDS]


def _even(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def _layout_filter(doc: ClipDocument, info: VideoInfo, ranges: list[tuple[float, float]]) -> str:
    W, H = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT
    if doc.layout == "blur":
        return (
            f"split[bg][fg];"
            f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},gblur=sigma=40,eq=brightness=-0.08[bgb];"
            f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1"
        )
    if split := gameplay_split(doc, info):
        gw, gh, gx, gy = split["game"]
        cw, ch, cx, cy = split["cam"]
        return (
            f"split[game][cam];"
            f"[game]crop={gw}:{gh}:{gx}:{gy},scale={W}:{split['top_h']},setsar=1[gtop];"
            f"[cam]crop={cw}:{ch}:{cx}:{cy},scale={W}:{split['bottom_h']},setsar=1[cbot];"
            f"[gtop][cbot]vstack=2"
        )
    if info.width / info.height > W / H:
        crop_w = _even(info.height * W / H)
        if doc.layout in ("face", "gameplay") and doc.face_track:
            x = f"'{_crop_x_expr(doc.face_track, ranges, info.width, crop_w)}'"
        else:
            x = str((info.width - crop_w) // 2)
        return f"crop={crop_w}:{_even(info.height)}:{x}:0,scale={W}:{H},setsar=1"
    crop_h = _even(info.width * H / W)
    return f"crop={_even(info.width)}:{crop_h}:0:{(info.height - crop_h) // 2},scale={W}:{H},setsar=1"


def render_clip(doc: ClipDocument, source: Path, info: VideoInfo, out_dir: Path) -> Path:
    ranges = keep_ranges(doc)
    if not ranges:
        raise RuntimeError("Todos os trechos do clipe foram cortados.")

    ass_path = out_dir / f"{doc.id}.ass"
    split = gameplay_split(doc, info)
    seam = split["top_h"] if split else None
    ass_path.write_text(build_ass(doc, ranges, config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT, seam), encoding="utf-8")

    # Busca rápida até perto do início e depois corte preciso com trim.
    offset = max(0.0, ranges[0][0] - 2.0)
    span = ranges[-1][1] - offset + 0.5

    parts, labels = [], []
    for k, (a, b) in enumerate(ranges):
        a, b = a - offset, b - offset
        parts.append(f"[0:v]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS[v{k}]")
        labels.append(f"[v{k}]")
        if info.has_audio:
            parts.append(f"[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS,afade=t=in:d=0.02[a{k}]")
            labels.append(f"[a{k}]")
    audio_flag = 1 if info.has_audio else 0
    parts.append(f"{''.join(labels)}concat=n={len(ranges)}:v=1:a={audio_flag}[vc]" + ("[ac]" if info.has_audio else ""))
    parts.append(f"[vc]{_layout_filter(doc, info, ranges)}[lay]")

    # O ffmpeg roda dentro de out_dir para o caminho da legenda não precisar de escape (C:\ no Windows).
    subtitle = f"ass=filename={ass_path.name}"
    try:
        fonts_rel = os.path.relpath(config.FONTS_DIR, out_dir).replace("\\", "/")
        subtitle += f":fontsdir={fonts_rel}"
    except ValueError:  # pastas em drives diferentes no Windows
        pass
    parts.append(f"[lay]{subtitle}[vout]")

    inputs = ["-ss", f"{offset:.3f}", "-t", f"{span:.3f}", "-i", str(source.resolve())]
    audio_label = "[ac]" if info.has_audio else None
    track = music.get_track(doc.music.track_id) if doc.music else None
    if track is not None:
        out_seconds = sum(b - a for a, b in ranges)
        inputs += ["-ss", f"{doc.music.offset:.3f}", "-stream_loop", "-1", "-i", str(music.MUSIC_DIR / track["file"])]
        fade_out = max(0.0, out_seconds - 1.5)
        parts.append(
            f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"volume={music.gain_db(track, doc.music.volume)}dB,atrim=0:{out_seconds:.3f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:d=0.6,afade=t=out:st={fade_out:.3f}:d=1.5[mus]"
        )
        if info.has_audio:
            parts.append("[ac]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[voice][sc]")
            if doc.music.ducking:
                # Compressor sidechain: a voz controla o ganho da música (ducking).
                parts.append("[mus][sc]sidechaincompress=threshold=0.02:ratio=6:attack=20:release=450[duck]")
            else:
                parts.append("[sc]anullsink;[mus]anull[duck]")
            parts.append("[voice][duck]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[amix]")
        else:
            parts.append("[mus]anull[amix]")
        audio_label = "[amix]"

    version = doc.render.version + 1
    output = out_dir / f"{doc.id}_v{version}.mp4"
    # O grafo vai num arquivo: a trajetória da câmera pode passar do limite de linha de comando do Windows.
    graph_path = out_dir / f"{doc.id}.filter"
    graph_path.write_text(";\n".join(parts), encoding="utf-8")
    cmd = [
        config.ffmpeg_exe(), "-y", "-v", "error", *inputs,
        "-/filter_complex", graph_path.name,
        "-map", "[vout]",
    ]
    if audio_label:
        cmd += ["-map", audio_label, "-c:a", "aac", "-b:a", "160k"]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-r", "30", "-movflags", "+faststart", output.name,
    ]
    proc = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg falhou: {proc.stderr.strip()[-1500:]}")

    for old in out_dir.glob(f"{doc.id}_v*.mp4"):
        if old != output:
            try:
                old.unlink()
            except OSError:
                pass  # o navegador pode estar com o arquivo aberto
    doc.render.version = version
    return output
