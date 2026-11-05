const app = document.getElementById("app");
const editor = document.getElementById("editor");
let pollTimer = null;
let currentClips = {};
let editing = null; // { jobId, clipId }

const STATUS_LABEL = { queued: "na fila", running: "processando", done: "pronto", error: "erro" };

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

function tpl(id) {
  return document.getElementById(id).content.cloneNode(true);
}

function fmtTime(sec) {
  const s = Math.round(sec);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function fmtSize(bytes) {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;
}

function clipDuration(clip) {
  const cut = clip.cuts.reduce((acc, c) => {
    const a = Math.max(c.start, clip.start), b = Math.min(c.end, clip.end);
    return acc + Math.max(0, b - a);
  }, 0);
  return clip.end - clip.start - cut;
}

function videoUrl(clip) {
  return clip.render.file ? `/media/${clip.job_id}/${clip.render.file}` : "";
}

function renderStateText(clip) {
  if (clip.render.status === "pending") return "na fila para renderizar…";
  if (clip.render.status === "rendering") return "renderizando…";
  if (clip.render.status === "error") return `erro: ${clip.render.error}`;
  return "";
}

function setVideo(video, clip) {
  const url = videoUrl(clip);
  if (url && !video.src.endsWith(url)) video.src = url;
}

/* ---------- Home ---------- */

async function showHome() {
  app.replaceChildren(tpl("tpl-home"));
  const form = document.getElementById("new-job");
  const fileInput = form.querySelector("input[type=file]");
  const fileLabel = form.querySelector(".file-btn");
  fileInput.addEventListener("change", () => {
    const name = fileInput.files[0]?.name;
    fileLabel.classList.toggle("has-file", !!name);
    fileLabel.querySelector("span").textContent = name || "Enviar arquivo";
    if (name) form.url.value = "";
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const error = form.querySelector(".form-error");
    const button = form.querySelector("button[type=submit]");
    error.hidden = true;
    const data = new FormData(form);
    data.set("remove_silences", form.remove_silences.checked ? "true" : "false");
    data.set("add_music", form.add_music.checked ? "true" : "false");
    if (!fileInput.files.length) data.delete("file");
    button.disabled = true;
    button.textContent = fileInput.files.length ? "Enviando…" : "Criando…";
    try {
      const job = await api("/api/jobs", { method: "POST", body: data });
      location.hash = `#/job/${job.id}`;
    } catch (err) {
      error.textContent = err.message;
      error.hidden = false;
      button.disabled = false;
      button.textContent = "Gerar cortes";
    }
  });

  const musicForm = document.getElementById("music-import");
  musicForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const state = app.querySelector(".music-state");
    try {
      await api("/api/music/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: musicForm.url.value }),
      });
      musicForm.url.value = "";
      refreshMusic();
    } catch (err) {
      state.textContent = err.message;
      state.classList.add("error");
    }
  });
  refreshMusic();

  const list = document.getElementById("jobs");
  const jobs = await api("/api/jobs");
  if (!jobs.length) {
    list.innerHTML = '<p class="muted">Nenhum projeto ainda.</p>';
    return;
  }
  for (const job of jobs) {
    const row = document.createElement("a");
    row.className = "job-row";
    row.href = `#/job/${job.id}`;
    row.innerHTML = `<span class="name"></span><span class="when muted"></span><span class="size muted"></span><span class="pill ${job.status}">${STATUS_LABEL[job.status]}</span><button class="ghost del" title="Apagar projeto e vídeos">✕</button>`;
    const name = job.source_name || job.source_url || job.id;
    row.querySelector(".name").textContent = name;
    row.querySelector(".when").textContent = new Date(job.created_at * 1000).toLocaleString("pt-BR");
    row.querySelector(".size").textContent = fmtSize(job.size);
    const del = row.querySelector(".del");
    del.disabled = job.status === "running" || job.status === "queued";
    del.addEventListener("click", async (ev) => {
      ev.preventDefault();
      if (!confirm(`Apagar "${name}"?
O vídeo baixado e todos os cortes (${fmtSize(job.size)}) serão excluídos.`)) return;
      try {
        await api(`/api/jobs/${job.id}`, { method: "DELETE" });
        row.remove();
        if (!list.children.length) list.innerHTML = '<p class="muted">Nenhum projeto ainda.</p>';
      } catch (err) {
        alert(err.message);
      }
    });
    list.append(row);
  }
}

/* ---------- Biblioteca de músicas ---------- */

let musicTimer = null;

async function refreshMusic() {
  clearTimeout(musicTimer);
  const container = document.getElementById("tracks");
  if (!container) return;
  const { state, tracks } = await api("/api/music");
  const stateEl = app.querySelector(".music-state");
  const bar = app.querySelector(".music-bar");
  const running = state.status === "running";
  stateEl.classList.toggle("error", state.status === "error");
  stateEl.textContent = running
    ? `${state.message}…`
    : state.status === "error" ? `Erro: ${state.message}`
    : tracks.length ? `${tracks.length} músicas analisadas. ${state.message || ""}`
    : "Importe uma playlist: cada música é analisada (BPM, tom, energia) e classificada pelo Claude.";
  bar.hidden = !running;
  bar.querySelector("div").style.width = `${Math.round((state.progress || 0) * 100)}%`;
  app.querySelector("#music-import button").disabled = running;

  container.replaceChildren();
  for (const t of tracks) {
    const row = document.createElement("div");
    row.className = "track";
    row.innerHTML = `
      <div>
        <div class="t-title"></div>
        <div class="t-meta"></div>
        <div class="moods"></div>
      </div>
      <div class="mood-dot" title="Posição no modelo de Russell: → valência (alegre), ↑ energia"><i></i></div>
      <button class="ghost del" title="Remover da biblioteca">✕</button>`;
    row.querySelector(".t-title").textContent = t.title;
    row.querySelector(".t-meta").textContent =
      `${t.bpm} BPM · ${t.key} ${t.mode === "major" ? "maior" : "menor"} · ${t.genre || "?"} · ${t.vocals ? "com voz" : "instrumental"}`;
    for (const m of t.moods || []) {
      const span = document.createElement("span");
      span.textContent = m;
      row.querySelector(".moods").append(span);
    }
    const dot = row.querySelector(".mood-dot i");
    dot.style.left = `${t.valence * 100}%`;
    dot.style.bottom = `${t.arousal * 100}%`;
    row.querySelector(".del").addEventListener("click", async () => {
      await api(`/api/music/${t.id}`, { method: "DELETE" });
      refreshMusic();
    });
    container.append(row);
  }
  if (running) musicTimer = setTimeout(refreshMusic, 2000);
}

/* ---------- Job ---------- */

async function showJob(jobId) {
  app.replaceChildren(tpl("tpl-job"));
  currentClips = {};
  await refreshJob(jobId);
}

async function refreshJob(jobId) {
  let data;
  try {
    data = await api(`/api/jobs/${jobId}`);
  } catch (err) {
    app.innerHTML = `<p class="job-error">${err.message}</p>`;
    return;
  }
  const { job, clips } = data;
  if (!location.hash.includes(jobId)) return;

  app.querySelector(".job-title").textContent = job.source_name || job.source_url || "Vídeo enviado";
  app.querySelector(".job-summary").textContent = job.summary;
  app.querySelector(".stage").textContent =
    job.status === "done" ? `${clips.length} cortes prontos` : `${job.stage}…`;
  app.querySelector(".bar div").style.width = `${Math.round(job.progress * 100)}%`;
  const error = app.querySelector(".job-error");
  error.hidden = job.status !== "error";
  error.textContent = job.error;

  const container = app.querySelector(".clips");
  for (const clip of clips) {
    let card = container.querySelector(`[data-id="${clip.id}"]`);
    if (!card) {
      container.append(tpl("tpl-clip"));
      card = container.lastElementChild;
      card.dataset.id = clip.id;
      card.querySelector(".edit").addEventListener("click", () => openEditor(jobId, clip.id));
      card.querySelector(".copy").addEventListener("click", (e) => {
        const c = currentClips[clip.id];
        const text = `${c.title}\n\n${c.description}\n\n${c.hashtags.map((h) => "#" + h).join(" ")}`;
        navigator.clipboard.writeText(text);
        e.target.textContent = "Copiado!";
        setTimeout(() => (e.target.textContent = "Copiar"), 1500);
      });
    }
    fillClipCard(card, clip);
    currentClips[clip.id] = clip;
    if (editing?.clipId === clip.id) fillEditor(clip);
  }

  const busy = job.status === "running" || job.status === "queued" ||
    clips.some((c) => c.render.status === "pending" || c.render.status === "rendering");
  clearTimeout(pollTimer);
  if (busy) pollTimer = setTimeout(() => refreshJob(jobId), 2500);
}

function fillClipCard(card, clip) {
  setVideo(card.querySelector("video"), clip);
  const state = card.querySelector(".render-state");
  state.textContent = renderStateText(clip);
  state.classList.toggle("error", clip.render.status === "error");
  card.querySelector(".score").textContent = `${clip.virality.total}/100`;
  card.querySelector(".duration").textContent = fmtTime(clipDuration(clip));
  card.querySelector(".hook").textContent = clip.hook_title || clip.title;
  card.querySelector(".reason").textContent = clip.virality.reasoning;
  const musicLine = card.querySelector(".music-line");
  musicLine.hidden = !clip.music;
  if (clip.music) {
    musicLine.textContent = `🎵 ${clip.music.title}`;
    musicLine.title = clip.music.reason;
  }
  card.querySelector(".post-title").textContent = clip.title;
  card.querySelector(".post-desc").textContent = clip.description;
  card.querySelector(".post-tags").textContent = clip.hashtags.map((h) => "#" + h).join(" ");
  const download = card.querySelector(".download");
  download.href = videoUrl(clip) || "#";
  download.setAttribute("download", `${(clip.title || clip.id).slice(0, 60)}.mp4`);
}

/* ---------- Editor com Claude ---------- */

function openEditor(jobId, clipId) {
  editing = { jobId, clipId };
  editor.querySelector(".chat-log").replaceChildren();
  editor.querySelector(".editor-video video").removeAttribute("src");
  fillEditor(currentClips[clipId]);
  editor.showModal();
  editor.querySelector("textarea").focus();
}

function addMessage(role, text, pending = false) {
  const log = editor.querySelector(".chat-log");
  const div = document.createElement("div");
  div.className = `msg ${role}${pending ? " pending" : ""}`;
  div.textContent = text;
  log.append(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

function fillEditor(clip) {
  if (!clip) return;
  setVideo(editor.querySelector(".editor-video video"), clip);
  const state = editor.querySelector(".editor-video .render-state");
  state.textContent = renderStateText(clip);
  state.classList.toggle("error", clip.render.status === "error");
  const log = editor.querySelector(".chat-log");
  if (!log.children.length) {
    addMessage("assistant", "Oi! Diga o que quer mudar neste corte: legenda, cores, título, cortes, enquadramento…");
    for (const m of clip.chat) addMessage(m.role, m.content);
  }
}

async function sendChat(text) {
  if (!editing || !text.trim()) return;
  const form = editor.querySelector(".chat-form");
  const button = form.querySelector("button");
  addMessage("user", text);
  const pending = addMessage("assistant", "Claude está editando…", true);
  button.disabled = true;
  try {
    const { reply, clip } = await api(`/api/jobs/${editing.jobId}/clips/${editing.clipId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    pending.remove();
    addMessage("assistant", reply);
    currentClips[clip.id] = clip;
    fillEditor(clip);
    refreshJob(editing.jobId);
  } catch (err) {
    pending.textContent = `Erro: ${err.message}`;
    pending.classList.remove("pending");
  } finally {
    button.disabled = false;
  }
}

editor.querySelector(".chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const textarea = editor.querySelector("textarea");
  const text = textarea.value;
  textarea.value = "";
  sendChat(text);
});
editor.querySelector("textarea").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    editor.querySelector(".chat-form").requestSubmit();
  }
});
editor.querySelectorAll(".suggestions button").forEach((b) =>
  b.addEventListener("click", () => sendChat(b.textContent))
);
editor.querySelector(".close").addEventListener("click", () => editor.close());
editor.addEventListener("close", () => {
  editor.querySelector("video").pause();
  editing = null;
});

/* ---------- Rotas ---------- */

function route() {
  clearTimeout(pollTimer);
  clearTimeout(musicTimer);
  const match = location.hash.match(/^#\/job\/(\w+)/);
  if (match) showJob(match[1]);
  else showHome();
}

window.addEventListener("hashchange", route);
route();
