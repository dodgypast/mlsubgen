// mlsubgen web — no framework: a poll for the queue and status, an EventSource for a job's log, a folder browser.
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const api = async (url, opts) => { const r = await fetch(url, opts); if (!r.ok) throw new Error(await r.text()); return r.json(); };

async function refreshStatus() {
  try {
    const s = await api("/api/status");
    const gpu = s.gpu.ok ? `GPU ${Math.round(s.gpu.used_mb / 1024)}/${Math.round(s.gpu.total_mb / 1024)} GB · ${s.gpu.util}%` : "GPU ?";
    const run = s.running.length ? ` · <b>job ${s.running[0].id} running</b>` : (s.other_run ? " · <b>a run outside the queue holds the GPU</b>" : "");
    $("#status").innerHTML = `worker <b>${s.worker ? "up" : "down"}</b> · LLM <b>${s.llm ? "up" : "down"}</b> · ${gpu} · queued <b>${s.queued}</b>${run}`;
    const live = $("#nav-live");
    const j = s.running.length ? s.running[0] : s.latest;
    if (j) {
      const p = j.progress;
      live.href = `/jobs/${j.id}`;
      if (s.running.length) {
        live.innerHTML = `▶ live log${p ? ` · ${esc(p.label)} · ${esc(p.phase)}` : ""}`;
        live.title = p && p.current ? p.current : `job ${j.id}`;
      } else {
        const wait = j.status === "queued" && j.not_before > s.now ? ` · retry in ${Math.max(1, Math.ceil((j.not_before - s.now) / 60))} min` : "";
        live.innerHTML = `log · job ${j.id} ${esc(j.status)}${wait}`;
        live.title = j.note || `job ${j.id}`;
      }
      live.classList.toggle("idle", !s.running.length);
      live.classList.remove("hidden");
    } else live.classList.add("hidden");
  } catch (e) { $("#status").textContent = "status unavailable"; }
}

const btn = (act, id, label) => `<button class="secondary" data-act="${act}" data-id="${id}">${label || act}</button>`;

function jobActions(j) {
  // final → retry; paused → resume + cancel; queued / running → pause + cancel; cancelling / pausing → nothing yet
  if (j.final) return btn("retry", j.id);
  if (j.status === "paused") return btn("resume", j.id) + " " + btn("cancel", j.id);
  if (j.status === "cancelling" || j.status === "pausing") return "";
  return btn("pause", j.id) + " " + btn("cancel", j.id);
}

function jobRow(j) {
  const act = jobActions(j);
  const prog = j.progress ? `<div class="note">${esc(j.progress.label)} · ${esc(j.progress.phase)}</div>` : "";
  return `<tr><td><a href="/jobs/${j.id}">${j.id}</a></td><td><span class="pill ${j.status}">${j.status}</span>${prog}</td>
    <td>${j.attempts}</td><td class="nowrap">${esc((j.started || j.created).slice(0, 16))}</td>
    <td><code>mlsubgen ${esc(j.label)}</code>${j.note ? `<div class="note">↳ ${esc(j.note)}</div>` : ""}</td><td>${act}</td></tr>`;
}

async function refreshJobs() {
  const js = await api("/api/jobs");
  $("#jobs tbody").innerHTML = js.length ? js.map(jobRow).join("") : `<tr><td colspan="6" class="muted">no jobs yet</td></tr>`;
  if (js.length > 20) $("#queue-note").textContent = "last 20";
}

async function act(el) {
  const r = await api(`/api/jobs/${el.dataset.id}/${el.dataset.act}`, { method: "POST" });
  return r.result;
}

function setPicked(paths) {
  $("#picked").innerHTML = paths.map(p => `<input type="hidden" name="paths" value="${esc(p)}">`).join("");
  $("#picked-note").textContent = paths.length ? `${paths.length} file${paths.length > 1 ? "s" : ""} picked — the job runs on those, not the folder` : "";
}

async function browse(path) {
  const d = await api("/api/ls?path=" + encodeURIComponent(path || ""));
  $("#cwd").textContent = d.path || "media roots";
  $("#counts").textContent = d.path ? `${d.videos} videos · ${d.srt} .srt here` : "";
  const up = d.path ? `<li><a href="#" data-path="${esc(d.parent)}">⬑ up</a></li>` : "";
  $("#dirs").innerHTML = up + d.dirs.map(x => `<li><a href="#" data-path="${esc(x.path)}">${esc(x.name)}/</a></li>`).join("");
  const badge = f => (d.targets || []).map(t => `<span class="${f.have.includes(t) ? "have" : "lack"}">${t}</span>`).join(" ");
  $("#files").innerHTML = (d.files || []).map(f =>
    `<li><input type="checkbox" data-file="${esc(f.path)}" title="tick to add to a multi-file job"> ` +
    `<a href="#" data-file="${esc(f.path)}" title="queue just this file">${esc(f.name)}</a> ${badge(f)}</li>`).join("");
  if (d.path) $("#path").value = d.path;
  setPicked([]);                                    // a new folder: start the selection over
}

function mlsubgenIndex() {
  refreshStatus(); refreshJobs();
  // the "Subtitles in" dropdown: the summary names the ticked languages; click outside closes it
  const dd = $("#targets-dd");
  if (dd) {
    const summarise = () => {
      const names = [...dd.querySelectorAll("input[type=checkbox]:checked")].map(c => c.dataset.name);
      $("#targets-summary").textContent = names.length ? names.join(", ") : "none — nothing will be written";
      $("#targets-summary").classList.toggle("muted", !names.length);
    };
    dd.addEventListener("change", summarise);
    $("#targets-none").addEventListener("click", () => { dd.querySelectorAll("input[type=checkbox]").forEach(c => c.checked = false); summarise(); });
    document.addEventListener("click", e => { if (dd.open && !dd.contains(e.target)) dd.open = false; });
    summarise();
  }
  const purge = async (done) => {
    try { $("#queue-note").textContent = (await api("/api/jobs/purge" + (done ? "?done=true" : ""), { method: "POST" })).result; }
    catch (err) { alert("purge failed: " + err.message); }
    refreshJobs();
  };
  $("#purge").addEventListener("click", () => purge(false));
  $("#purge-done").addEventListener("click", () => { if (confirm("Remove every finished job from the list? Their logs stay in ~/mlsubgen/logs.")) purge(true); });
  setInterval(refreshStatus, 5000); setInterval(refreshJobs, 3000);
  $("#jobs").addEventListener("click", async e => {
    const b = e.target.closest("button[data-act]"); if (!b) return;
    b.disabled = true; try { await act(b); } catch (err) { alert(err.message); } refreshJobs();
  });
  $("#browse-toggle").addEventListener("click", () => {
    const box = $("#browser"); box.classList.toggle("hidden");
    if (!box.classList.contains("hidden")) browse($("#path").value.trim()).catch(() => browse(""));
  });
  $("#files").addEventListener("click", e => {
    const a = e.target.closest("a[data-file]");
    if (a) { e.preventDefault(); $("#path").value = a.dataset.file; $("#files").querySelectorAll("input:checked").forEach(c => c.checked = false); setPicked([]); return; }
    if (e.target.matches("input[data-file]"))
      setPicked([...$("#files").querySelectorAll("input:checked")].map(c => c.dataset.file));
  });
  $("#dirs").addEventListener("click", e => {
    const a = e.target.closest("a[data-path]"); if (!a) return; e.preventDefault(); browse(a.dataset.path).catch(err => alert(err.message));
  });
  $("#preview").addEventListener("click", async () => {
    const out = $("#preview-out"); out.classList.remove("hidden"); out.textContent = "running --dry-run…";
    try {
      const r = await api("/api/preview", { method: "POST", body: new FormData($("#newjob")) });
      out.textContent = "$ " + r.command + " --dry-run\n" + r.output;
    } catch (err) { out.textContent = err.message; }
  });
}

function mlsubgenJob(id) {
  refreshStatus(); setInterval(refreshStatus, 5000);
  const log = $("#log");
  let follow = true;
  log.addEventListener("scroll", () => { follow = log.scrollTop + log.clientHeight >= log.scrollHeight - 8; });
  const append = lines => { log.textContent += lines.join("\n") + "\n"; if (follow) log.scrollTop = log.scrollHeight; };
  const es = new EventSource(`/api/jobs/${id}/stream`);
  es.addEventListener("log", e => append(JSON.parse(e.data)));
  es.addEventListener("end", e => { $("#log-note").textContent = "finished: " + JSON.parse(e.data); es.close(); });
  es.onerror = () => { $("#log-note").textContent = "stream lost — reload to reconnect"; };
  const refresh = async () => {
    const j = await api(`/api/jobs/${id}`);
    const st = $("#job-status"); st.textContent = j.status; st.className = "pill " + j.status;
    $("#job-started").textContent = j.started || "—"; $("#job-attempts").textContent = j.attempts;
    $("#job-note").textContent = j.note || "—"; $("#job-log-path").textContent = j.log || "— not started";
    $("#job-progress").textContent = j.progress ? `${j.progress.label} · ${j.progress.phase}${j.progress.current ? " · " + j.progress.current : ""}` : (j.final ? "finished" : "—");
    $("#cancel").disabled = j.final || j.status === "cancelling"; $("#retry").disabled = !j.final;
    $("#pause").disabled = !(j.status === "queued" || j.status === "running");
    $("#resume").disabled = j.status !== "paused";
    const f = await api(`/api/jobs/${id}/files`);
    $("#files tbody").innerHTML = f.files.length
      ? f.files.map(x => `<tr><td>${esc(x.name)}</td><td class="st-${x.status}">${x.status}</td><td>${esc(x.detail)}</td></tr>`).join("")
      : `<tr><td colspan="3" class="muted">${j.log ? "nothing started yet" : "the job has not started"}</td></tr>`;
    $("#files-note").textContent = [f.round, f.existing ? `${f.existing} already had an .srt` : "", f.summary].filter(Boolean).join(" · ");
    return j;
  };
  refresh(); const t = setInterval(async () => { const j = await refresh(); if (j.final) clearInterval(t); }, 4000);
  for (const which of ["pause", "resume", "cancel", "retry"]) $("#" + which).addEventListener("click", async e => {
    e.target.disabled = true;
    try { $("#action-result").textContent = (await api(`/api/jobs/${id}/${which}`, { method: "POST" })).result; }
    catch (err) { $("#action-result").textContent = err.message; }
    refresh();
  });
}

function mlsubgenSkipped() {
  refreshStatus(); setInterval(refreshStatus, 5000);
  const rows = [...document.querySelectorAll("#skipped tbody tr")];
  $("#filter").addEventListener("input", e => {
    const q = e.target.value.trim().toLowerCase();
    for (const r of rows) r.classList.toggle("hidden", q && !r.dataset.text.includes(q));
  });
  $("#selall").addEventListener("change", e => {
    for (const r of rows) if (!r.classList.contains("hidden")) { const c = r.querySelector("input[name=paths]"); if (!c.disabled) c.checked = e.target.checked; }
  });
}
