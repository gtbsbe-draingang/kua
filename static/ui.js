/* Kuá result page: upload, job status polling, verdict, video with an event timeline, event cards.
   Plain browser JavaScript, no libraries; all text goes through textContent. */
(function () {
  "use strict";

  var STAGE_LABELS = {
    upload: "Загрузка файла",
    queued: "В очереди",
    decoding: "Чтение видео",
    grounding: "Поиск событий",
    verifying: "Проверка фрагментов",
    writing: "Формирование результата"
  };
  var BASE_STEPS = ["upload", "queued", "decoding", "verifying", "writing"];
  var VERDICTS = {
    violation: "Обнаружены признаки возможных нарушений",
    no_violation: "Нарушение не обнаружено"
  };
  var POLL_MS = 2000;
  var MAX_RETRIES = 3;
  var CAP_MS = 6 * 60 * 1000;
  var MAX_MB = 200;
  var MAX_BYTES = MAX_MB * 1024 * 1024;
  var KEY_STORE = "kua_api_key";
  var JOB_RE = /^[0-9a-f]{12}$/;
  var VIDEO_EXT = /\.(mov|mp4|m4v|qt)$/i;
  var WINDOW_FALLBACK_SEC = 1.5; // highlight span around a timecode when the result has no t_start/t_end

  /* ---------- Small helpers ---------- */

  function $(id) { return document.getElementById(id); }

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  var SVG_NS = "http://www.w3.org/2000/svg";
  var ICONS = {
    eye: ["M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z", "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"],
    eyeOff: ["M3 3l18 18", "M10.6 5.1A10.8 10.8 0 0 1 12 5c6.4 0 10 7 10 7a17.6 17.6 0 0 1-3.2 4.2", "M6.6 6.6C3.8 8.4 2 12 2 12s3.6 7 10 7c1.9 0 3.6-.6 5-1.4", "M9.9 9.9a3 3 0 0 0 4.2 4.2"],
    upload: ["M12 16V4", "M6 10l6-6 6 6", "M4 15v5h16v-5"],
    clock: ["M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z", "M12 7v5l3 2"],
    arrow: ["M4 12h15", "M13 6l6 6-6 6"],
    external: ["M14 4h6v6", "M20 4l-9 9", "M18 14v6H4V6h6"],
    close: ["M6 6l12 12", "M18 6L6 18"],
    alert: ["M12 8v5", "M12 16.5v.5", "M10.3 3.9L2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"]
  };

  function icon(name) {
    var s = document.createElementNS(SVG_NS, "svg");
    s.setAttribute("viewBox", "0 0 24 24");
    s.setAttribute("fill", "none");
    s.setAttribute("stroke", "currentColor");
    s.setAttribute("stroke-width", "2.2");
    s.setAttribute("stroke-linecap", "square");
    s.setAttribute("aria-hidden", "true");
    s.setAttribute("focusable", "false");
    s.setAttribute("class", "ico");
    ICONS[name].forEach(function (d) {
      var p = document.createElementNS(SVG_NS, "path");
      p.setAttribute("d", d);
      s.appendChild(p);
    });
    return s;
  }

  function setIcon(host, name) {
    host.textContent = "";
    host.appendChild(icon(name));
  }

  function num(x) {
    var n = Number(x);
    return isFinite(n) ? n : NaN;
  }

  // 75.3 -> "1:15.3"; 3725 -> "1:02:05.0"
  function fmtTime(sec, decimals) {
    var d = decimals == null ? 1 : decimals;
    var scale = Math.pow(10, d);
    var n = Math.round(Math.max(0, num(sec) || 0) * scale) / scale;
    var h = Math.floor(n / 3600);
    var m = Math.floor((n % 3600) / 60);
    var s = n - h * 3600 - m * 60;
    var ss = s.toFixed(d);
    if (s < 10) ss = "0" + ss;
    return (h ? h + ":" + (m < 10 ? "0" : "") + m : String(m)) + ":" + ss;
  }

  function fmtClock(sec) { return fmtTime(Math.floor(Math.max(0, sec)), 0); }

  function fmtSeconds(sec) { return (Math.round((num(sec) || 0) * 10) / 10).toFixed(1) + " с"; }

  function fmtSize(bytes) {
    var mb = bytes / (1024 * 1024);
    if (mb >= 1) return (mb >= 100 ? Math.round(mb) : mb.toFixed(1)).toString().replace(".", ",") + " МБ";
    return Math.max(1, Math.round(bytes / 1024)) + " КБ";
  }

  function plural(n, one, few, many) {
    var m10 = n % 10, m100 = n % 100;
    if (m10 === 1 && m100 !== 11) return one;
    if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
    return many;
  }

  function str(x) { return typeof x === "string" ? x.trim() : ""; }

  function readKey() { try { return localStorage.getItem(KEY_STORE) || ""; } catch (e) { return ""; } }
  function writeKey(v) { try { if (v) localStorage.setItem(KEY_STORE, v); else localStorage.removeItem(KEY_STORE); } catch (e) { /* storage blocked */ } }

  function jobFromHash() {
    var m = /(?:^#|&)job=([0-9a-f]{12})(?:&|$)/.exec(location.hash || "");
    return m ? m[1] : null;
  }

  function setHash(id) {
    var url = location.pathname + location.search + (id ? "#job=" + id : "");
    try { history.replaceState(null, "", url); } catch (e) { if (id) location.hash = "job=" + id; }
  }

  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var wideQuery = window.matchMedia ? window.matchMedia("(min-width: 1100px)") : null;
  function scrollIntoViewSoft(node, block) {
    if (!node) return;
    try { node.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: block || "nearest" }); }
    catch (e) { node.scrollIntoView(); }
  }

  /* ---------- Elements ---------- */

  var form = $("form"), keyEl = $("key"), keyToggle = $("key-toggle"), fileEl = $("file"), drop = $("drop");
  var submitBtn = $("submit"), noticeBox = $("notice"), noticeText = $("notice-text"), noticeAction = $("notice-action");
  var errBox = $("error"), errLabel = $("error-label"), errText = $("error-text"), errActions = $("error-actions"), errAction = $("error-action");
  var errDetails = $("error-details"), errRaw = $("error-raw");
  var statusCard = $("status"), stageNow = $("stage-now"), elapsedEl = $("elapsed"), stepperEl = $("stepper");
  var meter = $("upload-meter"), meterFill = $("upload-fill"), meterText = $("upload-text"), retryEl = $("retry");
  var jobLine = $("jobline"), jobIdEl = $("job-id"), copyBtn = $("copy");
  var resultEl = $("result"), againBtn = $("again");
  var verdictEl = $("verdict"), verdictTitle = $("verdict-title"), verdictMeta = $("verdict-meta");
  var workspace = $("workspace"), video = $("video"), player = $("player"), playNote = $("playnote");
  var tlMarks = $("tl-marks"), tlTrack = $("tl-track"), tlHead = $("tl-head"), tlScale = $("tl-scale"), tlLegend = $("tl-legend");
  var eventsEl = $("events");
  var lightbox = $("lightbox"), lbImg = $("lb-img"), lbCaption = $("lb-caption"), lbSeek = $("lb-seek"), lbClose = $("lb-close");

  setIcon($("drop-icon"), "upload");
  setIcon(keyToggle, "eye");
  setIcon(lbClose, "close");
  submitBtn.appendChild(icon("arrow"));
  errLabel.insertBefore(icon("alert"), errLabel.firstChild);

  /* ---------- State ---------- */

  var run = 0;               // bumped by every new upload or resume; stale callbacks compare against it
  var jobId = null;
  var selectedFile = null;
  var blob = { url: null, job: null };   // local copy of the file uploaded in this tab
  var xhr = null, fetchCtl = null, pollTimer = null, tickTimer = null;
  var steps = [], stepIndex = -1, liveLabel = "";
  var clock = { base: 0, at: 0, running: false };
  var events = [], cards = [], marks = [], bands = [];
  var activeIndex = -1, pendingSeek = null;
  var sources = [], sourceIndex = -1, videoUsable = false, lastKnownElapsed = null;
  var lbOpener = null, lbTime = null, lbEvent = -1;

  /* ---------- API key ---------- */

  keyEl.value = readKey();
  keyEl.addEventListener("input", function () { writeKey(keyEl.value.trim()); });
  keyToggle.addEventListener("click", function () {
    var show = keyEl.type === "password";
    keyEl.type = show ? "text" : "password";
    keyToggle.setAttribute("aria-pressed", show ? "true" : "false");
    keyToggle.setAttribute("aria-label", show ? "Скрыть ключ" : "Показать ключ");
    setIcon(keyToggle, show ? "eyeOff" : "eye");
  });

  /* ---------- File choice and drag-and-drop ---------- */

  function isVideoFile(f) {
    return (f.type && f.type.indexOf("video/") === 0) || VIDEO_EXT.test(f.name || "");
  }

  function fileProblem(f) {
    if (!f) return "Выберите видеофайл.";
    if (!isVideoFile(f)) return "Это не видеофайл. Нужна запись в формате MOV или MP4.";
    if (f.size > MAX_BYTES) return "Файл слишком большой: " + fmtSize(f.size) + ". Максимальный размер — " + MAX_MB + " МБ.";
    if (f.size === 0) return "Файл пустой. Выберите другую запись.";
    return "";
  }

  function chooseFile(f) {
    selectedFile = f || null;
    var problem = f ? fileProblem(f) : "";
    drop.classList.toggle("has-file", !!f);
    drop.classList.toggle("is-invalid", !!problem);
    $("file-name").textContent = f ? f.name : "";
    $("file-size").textContent = f ? fmtSize(f.size) : "";
    $("drop-pick").textContent = f ? "выберите другой файл" : "выберите файл";
    clearError();
    if (problem) showError(problem);
  }

  fileEl.addEventListener("change", function () { chooseFile(fileEl.files && fileEl.files[0]); });

  function hasFiles(e) {
    var t = e.dataTransfer && e.dataTransfer.types;
    if (!t) return false;
    for (var i = 0; i < t.length; i++) if (t[i] === "Files") return true;
    return false;
  }

  var dragDepth = 0;
  drop.addEventListener("dragenter", function (e) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth++;
    drop.classList.add("is-over");
  });
  drop.addEventListener("dragover", function (e) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  drop.addEventListener("dragleave", function () {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) drop.classList.remove("is-over");
  });
  drop.addEventListener("drop", function (e) {
    e.preventDefault();
    dragDepth = 0;
    drop.classList.remove("is-over");
    var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    try { fileEl.files = e.dataTransfer.files; } catch (err) { /* read-only in old browsers; selectedFile is enough */ }
    chooseFile(f);
  });
  // A file dropped next to the zone must not make the browser navigate away from the page.
  window.addEventListener("dragover", function (e) { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("drop", function (e) { if (hasFiles(e)) e.preventDefault(); });

  /* ---------- Messages ---------- */

  function showError(message, opts) {
    opts = opts || {};
    errText.textContent = message;
    if (opts.action) {
      errAction.textContent = opts.action.label;
      errAction.onclick = opts.action.run;
      errActions.hidden = false;
    } else {
      errAction.onclick = null;
      errActions.hidden = true;
    }
    errRaw.textContent = opts.raw || "";
    errDetails.hidden = !opts.raw;
    errDetails.open = false;
    errBox.hidden = false;
  }

  function clearError() { errBox.hidden = true; errText.textContent = ""; }

  function showNotice(text, actionLabel, action) {
    noticeText.textContent = text;
    noticeAction.textContent = actionLabel;
    noticeAction.onclick = action;
    noticeBox.hidden = false;
  }

  function clearNotice() { noticeBox.hidden = true; }

  /* ---------- HTTP ---------- */

  function HttpError(status, detail) { this.status = status; this.detail = detail || ""; }

  function detailOf(text) {
    try {
      var d = JSON.parse(text).detail;
      if (d == null) return "";
      if (typeof d === "string") return d;
      if (Array.isArray(d)) return d.map(function (x) { return x && x.msg ? x.msg : JSON.stringify(x); }).join("; ");
      return JSON.stringify(d);
    } catch (e) { return ""; }
  }

  function describeHttp(e, what) {
    if (e instanceof HttpError) {
      if (e.status === 401) return "Неверный API-ключ. Проверьте ключ и повторите.";
      if (e.status === 413) return "Файл слишком большой. Максимальный размер — " + MAX_MB + " МБ.";
      if (e.status === 404 && what === "poll") return "Задача не найдена на сервере. Загрузите видео ещё раз.";
      if (e.status >= 500) return "Ошибка сервера (HTTP " + e.status + "). Повторите попытку позже.";
      return "Запрос отклонён (HTTP " + e.status + ")" + (e.detail ? ": " + e.detail : ".");
    }
    return "Нет связи с сервером. Проверьте подключение и повторите.";
  }

  function isTransient(e) { return !(e instanceof HttpError) || e.status >= 500; }

  function failureText(raw) {
    var s = String(raw || "");
    if (/^VideoError\b/.test(s)) return "Не удалось прочитать видео. Проверьте, что файл не повреждён, и загрузите его ещё раз.";
    if (/^FileNotFoundError\b/.test(s)) return "Файл задачи не найден на сервере. Загрузите видео ещё раз.";
    if (/^result unavailable\b/.test(s)) return "Результат на сервере недоступен. Загрузите видео ещё раз.";
    return "Обработка завершилась ошибкой. Попробуйте загрузить видео ещё раз.";
  }

  /* ---------- Run lifecycle ---------- */

  function stopActivity() {
    clearTimeout(pollTimer);
    clearInterval(tickTimer);
    pollTimer = tickTimer = null;
    if (xhr) { xhr.onload = xhr.onerror = xhr.ontimeout = xhr.onabort = null; try { xhr.abort(); } catch (e) { /* ignore */ } xhr = null; }
    if (fetchCtl) { try { fetchCtl.abort(); } catch (e) { /* ignore */ } fetchCtl = null; }
    clock.running = false;
  }

  function newRun() {
    stopActivity();
    run++;
    clearError();
    clearNotice();
    hideResult();
    retryEl.hidden = true;
    meter.hidden = true;
    jobLine.hidden = true;
    elapsedEl.textContent = "";
    liveLabel = "";
    lastKnownElapsed = null;
    return run;
  }

  function setBusy(busy) {
    submitBtn.disabled = busy;
    submitBtn.firstChild.textContent = busy ? "Идёт проверка…" : "Проверить видео";
  }

  function finishRun() {
    stopActivity();
    setBusy(false);
  }

  /* ---------- Status card ---------- */

  function resetSteps(uploadDone) {
    steps = BASE_STEPS.slice();
    stepIndex = uploadDone ? 1 : 0;
    renderSteps();
  }

  function stageLabel(key) { return STAGE_LABELS[key] || key; }

  function setStage(key) {
    if (!key) key = "queued";
    var i = steps.indexOf(key);
    if (i === -1) {
      // A stage outside the usual path (or an unknown one): show it right after the current step.
      i = Math.max(stepIndex, 0) + 1;
      steps.splice(i, 0, key);
    }
    if (i < stepIndex) i = stepIndex; // never step backwards on a late or repeated report
    stepIndex = i;
    renderSteps();
    announce(stageLabel(steps[stepIndex]));
  }

  function announce(label) {
    if (label === liveLabel) return;
    liveLabel = label;
    stageNow.textContent = label;
  }

  function renderSteps() {
    stepperEl.textContent = "";
    steps.forEach(function (key, i) {
      var li = el("li", i < stepIndex ? "done" : i === stepIndex ? "current" : "todo");
      li.appendChild(el("span", "mk"));
      li.appendChild(el("span", "txt", stageLabel(key)));
      li.appendChild(el("span", "bar"));
      if (i === stepIndex) li.setAttribute("aria-current", "step");
      var state = i < stepIndex ? "выполнено" : i === stepIndex ? "сейчас" : "впереди";
      li.appendChild(el("span", "sr-only", ", " + state));
      stepperEl.appendChild(li);
    });
  }

  function showStatus() {
    statusCard.hidden = false;
  }

  // Keep the status card as a finished step above the result: every stage ticked, the final time, the job id.
  function completeStatus() {
    stepIndex = steps.length;
    renderSteps();
    announce("Проверка завершена");
    meter.hidden = true;
    retryEl.hidden = true;
    elapsedEl.textContent = lastKnownElapsed != null ? fmtClock(lastKnownElapsed) : "";
    statusCard.hidden = false;
  }

  function startClock(base) {
    clock.base = base || 0;
    clock.at = Date.now();
    clock.running = true;
    renderClock();
    clearInterval(tickTimer);
    tickTimer = setInterval(renderClock, 1000);
  }

  function syncClock(serverElapsed) {
    var s = num(serverElapsed);
    if (!isFinite(s) || s < 0) return;
    lastKnownElapsed = s;
    clock.base = s;
    clock.at = Date.now();
    renderClock();
  }

  function renderClock() {
    if (!clock.running) return;
    var sec = clock.base + (Date.now() - clock.at) / 1000;
    lastKnownElapsed = sec;
    elapsedEl.textContent = fmtClock(sec);
  }

  function showJob(id) {
    jobIdEl.textContent = id;
    copyBtn.textContent = "Копировать";
    jobLine.hidden = false;
  }

  copyBtn.addEventListener("click", function () {
    var text = jobIdEl.textContent;
    function done() {
      copyBtn.textContent = "Скопировано";
      setTimeout(function () { copyBtn.textContent = "Копировать"; }, 2000);
    }
    function selectIt() {
      var r = document.createRange();
      r.selectNodeContents(jobIdEl);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(r);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, selectIt);
    } else {
      selectIt();
    }
  });

  /* ---------- Upload ---------- */

  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    clearError();
    var key = keyEl.value.trim();
    if (!key) { showError("Введите API-ключ."); keyEl.focus(); return; }
    var f = selectedFile || (fileEl.files && fileEl.files[0]);
    var problem = fileProblem(f);
    if (problem) { showError(problem); fileEl.focus(); return; }
    upload(f, key);
  });

  function upload(f, key) {
    var my = newRun();
    jobId = null;
    setHash(null);
    if (blob.url) URL.revokeObjectURL(blob.url);
    blob = { url: URL.createObjectURL(f), job: null };
    setBusy(true);
    resetSteps(false);
    announce(stageLabel("upload"));
    showStatus();
    meter.hidden = false;
    setMeter(0, f.size, 0);
    scrollIntoViewSoft(statusCard, "nearest");

    var fd = new FormData();
    fd.append("file", f, f.name || "video.mov");
    xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/v1/analyze");
    xhr.setRequestHeader("X-API-Key", key);
    xhr.responseType = "text";
    xhr.upload.onprogress = function (e) {
      if (my !== run) return;
      if (e.lengthComputable) setMeter(e.loaded / e.total, e.total, e.loaded);
    };
    xhr.onload = function () {
      if (my !== run) return;
      var status = xhr.status, text = xhr.responseText;
      xhr = null;
      if (status >= 200 && status < 300) {
        var body = null;
        try { body = JSON.parse(text); } catch (e) { /* handled below */ }
        if (!body || typeof body.job_id !== "string" || !JOB_RE.test(body.job_id)) {
          uploadFailed("Сервер вернул неожиданный ответ. Повторите попытку.");
          return;
        }
        setMeter(1, f.size, f.size);
        meter.hidden = true;
        accepted(body.job_id, key, my);
      } else {
        uploadFailed(describeHttp(new HttpError(status, detailOf(text)), "post"));
      }
    };
    xhr.onerror = function () { if (my === run) { xhr = null; uploadFailed(describeHttp(null, "post")); } };
    xhr.ontimeout = xhr.onerror;
    xhr.send(fd);
  }

  function setMeter(frac, total, loaded) {
    var pct = Math.max(0, Math.min(100, Math.round(frac * 100)));
    meterFill.style.width = pct + "%";
    meterText.textContent = pct + "% · " + fmtSize(loaded) + " из " + fmtSize(total);
    elapsedEl.textContent = pct + "%";
  }

  function uploadFailed(message) {
    finishRun();
    statusCard.hidden = true;
    showError(message, { action: { label: "Повторить", run: function () { form.requestSubmit ? form.requestSubmit() : submitBtn.click(); } } });
    scrollIntoViewSoft(errBox, "nearest");
  }

  function accepted(id, key, my) {
    jobId = id;
    blob.job = id;
    setHash(id);
    showJob(id);
    setStage("queued");
    startClock(0);
    poll(id, key, my, Date.now(), 0);
  }

  /* ---------- Resume from #job=<id> ---------- */

  function resume(id) {
    var key = keyEl.value.trim();
    if (!key) {
      showNotice("Чтобы открыть задачу " + id + ", введите API-ключ.", "Открыть задачу", function () {
        if (!keyEl.value.trim()) { keyEl.focus(); return; }
        resume(id);
      });
      return;
    }
    var my = newRun();
    jobId = id;
    setBusy(true);
    resetSteps(true);
    showJob(id);
    announce("Открываю задачу…");
    statusCard.hidden = true; // shown on the first "processing" answer, so a finished job opens straight on its result
    poll(id, key, my, Date.now(), 0, true);
  }

  window.addEventListener("hashchange", function () {
    var id = jobFromHash();
    if (id && id !== jobId) resume(id);
  });

  /* ---------- Polling ---------- */

  function poll(id, key, my, t0, fails, first) {
    if (my !== run) return;
    if (Date.now() - t0 > CAP_MS) {
      finishRun();
      showError("Обработка идёт дольше 6 минут, страница перестала ждать. Номер задачи: " + id + ".", {
        action: { label: "Проверить ещё раз", run: function () { resume(id); } }
      });
      scrollIntoViewSoft(errBox, "nearest");
      return;
    }
    fetchCtl = typeof AbortController === "function" ? new AbortController() : null;
    fetch("/api/v1/jobs/" + encodeURIComponent(id), {
      headers: { "X-API-Key": key },
      cache: "no-store",
      signal: fetchCtl ? fetchCtl.signal : undefined
    }).then(function (r) {
      if (r.ok) return r.json();
      return r.text().then(function (t) { throw new HttpError(r.status, detailOf(t)); },
        function () { throw new HttpError(r.status, ""); });
    }).then(function (j) {
      if (my !== run) return;
      fetchCtl = null;
      retryEl.hidden = true;
      if (!j || typeof j !== "object") throw new Error("bad body");
      if (j.status === "done") {
        finishRun();
        completeStatus();
        renderResult(j);
      } else if (j.status === "failed") {
        finishRun();
        statusCard.hidden = true;
        showError(failureText(j.error), { raw: j.error ? String(j.error) : "" });
        scrollIntoViewSoft(errBox, "nearest");
      } else {
        if (first) { showStatus(); startClock(0); }
        if (j.elapsed_sec != null) syncClock(j.elapsed_sec);
        setStage(typeof j.stage === "string" && j.stage ? j.stage : "queued");
        pollTimer = setTimeout(function () { poll(id, key, my, t0, 0); }, POLL_MS);
      }
    }).catch(function (e) {
      if (my !== run || (e && e.name === "AbortError")) return;
      fetchCtl = null;
      if (isTransient(e) && fails < MAX_RETRIES) {
        if (first) showStatus();
        retryEl.textContent = "Сервер не ответил, повторяю запрос (" + (fails + 1) + " из " + MAX_RETRIES + ")…";
        retryEl.hidden = false;
        pollTimer = setTimeout(function () { poll(id, key, my, t0, fails + 1, first); }, POLL_MS);
        return;
      }
      finishRun();
      statusCard.hidden = true;
      var msg = describeHttp(e, "poll");
      if (e instanceof HttpError && e.status === 404) setHash(null);
      var opts = {};
      if (e instanceof HttpError && e.status === 401) keyEl.focus();
      else if (isTransient(e)) opts.action = { label: "Проверить ещё раз", run: function () { resume(id); } };
      showError(msg, opts);
      scrollIntoViewSoft(errBox, "nearest");
    });
  }

  /* ---------- Result ---------- */

  function hideResult() {
    resultEl.hidden = true;
    closeLightbox();
    try { video.pause(); } catch (e) { /* ignore */ }
    video.removeAttribute("src");
    try { video.load(); } catch (e) { /* ignore */ }
    sources = [];
    sourceIndex = -1;
    videoUsable = false;
    events = []; cards = []; marks = []; bands = [];
    activeIndex = -1;
    pendingSeek = null;
    eventsEl.textContent = "";
    tlMarks.textContent = "";
    tlScale.textContent = "";
    Array.prototype.slice.call(tlTrack.querySelectorAll(".tl-band")).forEach(function (b) { b.remove(); });
  }

  function normalizeEvents(list) {
    var out = [];
    (Array.isArray(list) ? list : []).forEach(function (v, i) {
      if (!v || typeof v !== "object") return;
      var tc = num(v.timecode_sec);
      var ts = num(v.t_start), te = num(v.t_end);
      var hasWindow = isFinite(ts) && isFinite(te) && te >= ts;
      out.push({
        raw: v,
        order: i,
        timecode: isFinite(tc) ? Math.max(0, tc) : 0,
        start: hasWindow ? Math.max(0, ts) : null,
        end: hasWindow ? te : null,
        frames: (Array.isArray(v.evidence_frames) ? v.evidence_frames : []).filter(function (f) {
          return f && typeof f.image_base64 === "string" && f.image_base64;
        })
      });
    });
    out.sort(function (a, b) { return a.timecode - b.timecode || a.order - b.order; });
    return out;
  }

  function renderResult(job) {
    hideResult();
    events = normalizeEvents(job.violations);
    var v = job.verdict;
    verdictEl.className = "verdict" + (v === "violation" ? " is-violation" : v === "no_violation" ? " is-clean" : "");
    verdictTitle.textContent = VERDICTS[v] || "Результат не распознан. Обратитесь к администратору сервиса.";
    var n = events.length;
    verdictMeta.textContent = v === "no_violation" && !n
      ? "Эпизодов для проверки нет"
      : n + " " + plural(n, "эпизод", "эпизода", "эпизодов") + " для проверки";

    workspace.classList.toggle("no-events", !n);
    if (n) events.forEach(function (e, i) { eventsEl.appendChild(renderCard(e, i)); });
    else eventsEl.appendChild(renderEmpty());

    setupVideo(typeof job.video_url === "string" ? job.video_url : "");
    renderTimeline();

    resultEl.hidden = false;
    try { verdictEl.focus({ preventScroll: true }); } catch (e) { verdictEl.focus(); }
    scrollIntoViewSoft(resultEl, "start");
  }

  function renderEmpty() {
    var li = el("li");
    var card = el("div", "card empty-card");
    card.appendChild(el("p", "title", "Отметок нет"));
    card.appendChild(el("p", null, "Kuá не нашёл фрагментов, похожих на нарушения из каталога правил. Это не гарантия: при сомнениях просмотрите запись сами."));
    li.appendChild(card);
    return li;
  }

  function section(title) {
    var s = el("section", "ev-sec");
    s.appendChild(el("h4", "label", title));
    return s;
  }

  function linkHost(url) {
    try {
      var u = new URL(url);
      if (u.protocol !== "https:" && u.protocol !== "http:") return "";
      return u.hostname.replace(/^www\./, "");
    } catch (e) { return ""; }
  }

  function renderCard(e, i) {
    var v = e.raw;
    var li = el("li", "event");
    li.id = "event-" + (i + 1);
    var card = el("article", "card event-card");
    card.setAttribute("aria-labelledby", "event-title-" + (i + 1));

    var head = el("div", "ev-head");
    head.appendChild(el("span", "ev-no", String(i + 1).padStart(2, "0")));
    var tc = el("button", "tc-btn");
    tc.type = "button";
    tc.appendChild(icon("clock"));
    tc.appendChild(document.createTextNode(fmtTime(e.timecode)));
    tc.setAttribute("aria-label", "Перейти к " + fmtTime(e.timecode) + " на видео");
    tc.addEventListener("click", function () { jumpTo(e.timecode, i, "card"); });
    head.appendChild(tc);
    if (v.type_source === "judge") {
      var tj = el("span", "tag outline", "оценка модели");
      tj.title = "Тип и обоснование сформулировала языковая модель по найденным нормам";
      head.appendChild(tj);
    } else if (v.type_source === "fallback") {
      var tf = el("span", "tag ghost", "по каталогу");
      tf.title = "Тип и обоснование взяты из каталога правил";
      head.appendChild(tf);
    }
    card.appendChild(head);

    var titleBox = el("div");
    var h = el("h3", "ev-type", str(v.type) || "Возможное нарушение");
    h.id = "event-title-" + (i + 1);
    titleBox.appendChild(h);
    if (e.start != null) titleBox.appendChild(el("p", "ev-window caption", "Фрагмент " + fmtTime(e.start) + " – " + fmtTime(e.end)));
    card.appendChild(titleBox);

    var observation = str(v.observation), justification = str(v.justification);
    if (observation) {
      var s1 = section("Что видно");
      s1.appendChild(el("p", "ev-text", observation));
      card.appendChild(s1);
    }
    if (justification) {
      var s2 = section("Почему это может быть нарушением");
      s2.appendChild(el("p", "ev-text", justification));
      card.appendChild(s2);
    }
    if (!observation && !justification && str(v.description)) {
      // results stored before the split fields existed carry only the merged text
      var s0 = section("Описание");
      s0.appendChild(el("p", "ev-text", str(v.description)));
      card.appendChild(s0);
    }

    var r = v.rule && typeof v.rule === "object" ? v.rule : {};
    var doc = str(r.document), clause = str(r.clause), quote = str(r.quote), url = str(r.url);
    var host = url ? linkHost(url) : "";
    if (doc || clause || quote || host) {
      var s3 = section("Правило и источник");
      if (doc) s3.appendChild(el("p", "rule-doc", doc));
      if (clause) s3.appendChild(el("p", "rule-clause", clause));
      if (quote) s3.appendChild(el("blockquote", null, "«" + quote.replace(/^[«"]+|[»"]+$/g, "") + "»"));
      if (host) {
        var a = el("a", "src-link");
        a.href = url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.appendChild(document.createTextNode("Открыть на " + host));
        a.appendChild(icon("external"));
        a.setAttribute("aria-label", "Открыть на " + host + " (в новой вкладке)");
        s3.appendChild(a);
      }
      card.appendChild(s3);
    }

    if (e.frames.length) {
      var s4 = section(e.frames.length > 1 ? "Кадры-доказательства" : "Кадр-доказательство");
      var ul = el("ul", "frames");
      e.frames.slice().sort(function (a, b) { return num(a.timestamp_sec) - num(b.timestamp_sec); }).forEach(function (f) {
        ul.appendChild(renderFrame(f, e, i));
      });
      s4.appendChild(ul);
      card.appendChild(s4);
    }

    li.appendChild(card);
    cards[i] = card;
    return li;
  }

  function renderFrame(f, e, i) {
    var t = num(f.timestamp_sec);
    if (!isFinite(t)) t = e.timecode;
    var isKey = Math.abs(t - e.timecode) < 0.05;
    var li = el("li");
    var fig = el("figure");
    var open = el("button", "frame-open" + (isKey ? " is-key" : ""));
    open.type = "button";
    open.setAttribute("aria-label", "Открыть кадр " + fmtSeconds(t) + " крупно");
    var img = document.createElement("img");
    img.src = "data:image/jpeg;base64," + f.image_base64;
    img.alt = "Кадр-доказательство, " + fmtSeconds(t);
    img.decoding = "async";
    open.appendChild(img);
    open.addEventListener("click", function () { openLightbox(img.src, t, i, open); });
    fig.appendChild(open);
    var cap = el("figcaption");
    var tb = el("button", "frame-tc", fmtTime(t));
    tb.type = "button";
    tb.setAttribute("aria-label", "Перейти к кадру " + fmtTime(t) + " на видео");
    tb.addEventListener("click", function () { jumpTo(t, i, "card"); });
    cap.appendChild(tb);
    if (isKey) cap.appendChild(el("span", "key-mark", "таймкод"));
    fig.appendChild(cap);
    li.appendChild(fig);
    return li;
  }

  /* ---------- Video ---------- */

  function setupVideo(url) {
    sources = [];
    if (url) sources.push(url);
    if (blob.url && blob.job && blob.job === jobId) sources.push(blob.url);
    sourceIndex = -1;
    nextSource();
  }

  function nextSource() {
    sourceIndex++;
    videoUsable = false;
    if (sourceIndex >= sources.length) {
      player.hidden = true;
      playNote.hidden = false;
      video.removeAttribute("src");
      return;
    }
    player.hidden = false;
    playNote.hidden = true;
    video.src = sources[sourceIndex];
  }

  video.addEventListener("error", function () {
    if (sourceIndex >= 0 && sourceIndex < sources.length && video.getAttribute("src")) nextSource();
  });

  video.addEventListener("loadedmetadata", function () {
    if (!video.videoWidth && !video.videoHeight) { nextSource(); return; } // audio only or an undecodable video track
    videoUsable = true;
    renderTimeline();
    if (pendingSeek != null) { var t = pendingSeek; pendingSeek = null; setTime(t); }
  });
  video.addEventListener("durationchange", function () { if (videoUsable) renderTimeline(); });
  video.addEventListener("timeupdate", onTime);
  video.addEventListener("seeked", onTime);

  function setTime(t) {
    try { video.currentTime = t; } catch (e) { pendingSeek = t; }
  }

  function seek(t) {
    t = Math.max(0, num(t) || 0);
    var d = duration();
    if (d > 0 && t > d) t = d;
    if (!videoUsable || video.readyState < 1) { pendingSeek = t; return; }
    setTime(t);
    placeHead(t);
  }

  function jumpTo(t, i, from) {
    seek(t);
    setActive(i);
    if (from === "card") {
      // keep the video in view so the jump is visible
      scrollIntoViewSoft(player.hidden ? playNote : player, "nearest");
    } else if (from === "timeline" && wideQuery && wideQuery.matches && cards[i]) {
      scrollIntoViewSoft(cards[i].parentNode, "nearest");
    }
  }

  function duration() {
    var d = num(video.duration);
    if (videoUsable && isFinite(d) && d > 0) return d;
    var m = 0;
    events.forEach(function (e) {
      m = Math.max(m, e.timecode, e.end != null ? e.end : 0);
      e.frames.forEach(function (f) { var t = num(f.timestamp_sec); if (isFinite(t)) m = Math.max(m, t); });
    });
    return m;
  }

  function eventUnder(t) {
    var best = -1, bestDist = Infinity;
    events.forEach(function (e, i) {
      var s = e.start != null ? e.start : e.timecode - WINDOW_FALLBACK_SEC;
      var en = e.end != null ? e.end : e.timecode + WINDOW_FALLBACK_SEC;
      if (t >= s - 0.05 && t <= en + 0.05) {
        var dist = Math.abs(t - e.timecode);
        if (dist < bestDist) { best = i; bestDist = dist; }
      }
    });
    return best;
  }

  function onTime() {
    var t = num(video.currentTime) || 0;
    placeHead(t);
    var i = eventUnder(t);
    if (i !== -1) setActive(i);
    else if (!video.paused) setActive(-1);
  }

  function setActive(i) {
    if (i === activeIndex) return;
    [cards, marks, bands].forEach(function (list) {
      if (list[activeIndex]) list[activeIndex].classList.remove("is-current");
      if (list[i]) list[i].classList.add("is-current");
    });
    if (cards[activeIndex]) cards[activeIndex].removeAttribute("aria-current");
    if (cards[i]) cards[i].setAttribute("aria-current", "true");
    activeIndex = i;
  }

  /* ---------- Timeline ---------- */

  var NICE_STEPS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];

  function renderTimeline() {
    var d = duration();
    tlMarks.textContent = "";
    tlScale.textContent = "";
    marks = [];
    bands.forEach(function (b) { b.remove(); });
    bands = [];
    tlLegend.hidden = !events.length;
    tlHead.hidden = !(d > 0 && videoUsable);
    if (!(d > 0)) return;

    events.forEach(function (e, i) {
      if (e.start != null) {
        var b = el("div", "tl-band");
        b.style.left = pct(e.start, d);
        b.style.width = "calc(" + pct(Math.min(e.end, d) - e.start, d) + ")";
        b.title = "Эпизод " + (i + 1) + ": " + fmtTime(e.start) + " – " + fmtTime(e.end);
        b.addEventListener("click", function (ev) { ev.stopPropagation(); jumpTo(e.timecode, i, "timeline"); });
        tlTrack.insertBefore(b, tlHead);
        bands[i] = b;
      }
      var m = el("button", "tl-mark", String(i + 1));
      m.type = "button";
      m.style.left = pct(e.timecode, d);
      m.setAttribute("aria-label", "Эпизод " + (i + 1) + ", " + fmtTime(e.timecode) + ": перейти");
      m.addEventListener("click", function () { jumpTo(e.timecode, i, "timeline"); });
      tlMarks.appendChild(m);
      marks[i] = m;
    });
    if (activeIndex !== -1) {
      var keep = activeIndex;
      activeIndex = -1;
      setActive(keep);
    }
    layoutMarks();
    renderScale(d);
    placeHead(num(video.currentTime) || 0);
  }

  function pct(t, d) { return (Math.max(0, Math.min(1, t / d)) * 100).toFixed(3) + "%"; }

  // Markers closer than their own width go to a higher lane, so every number stays readable and clickable.
  function layoutMarks() {
    var w = tlTrack.clientWidth || 1, d = duration(), gap = 30, lanes = [], maxLane = 0;
    events.forEach(function (e, i) {
      if (!marks[i]) return;
      var x = Math.max(0, Math.min(1, e.timecode / d)) * w, lane = 0;
      while (lane < lanes.length && x - lanes[lane] < gap) lane++;
      if (lane > 2) lane = 2;
      lanes[lane] = x;
      maxLane = Math.max(maxLane, lane);
      marks[i].style.bottom = (4 + lane * 30) + "px";
      marks[i].style.setProperty("--stem", (6 + lane * 30) + "px");
    });
    tlMarks.style.height = (30 + maxLane * 30) + "px";
  }

  function renderScale(d) {
    var w = tlTrack.clientWidth || 300;
    var maxTicks = Math.max(2, Math.floor(w / 64));
    var step = NICE_STEPS[NICE_STEPS.length - 1];
    for (var k = 0; k < NICE_STEPS.length; k++) {
      if (d / NICE_STEPS[k] <= maxTicks) { step = NICE_STEPS[k]; break; }
    }
    var decimals = step < 1 ? 1 : 0;
    for (var t = 0; t <= d + 1e-6; t += step) {
      var frac = t / d;
      var tick = el("span", "tl-tick", fmtTime(t, decimals));
      tick.style.left = (frac * 100).toFixed(3) + "%";
      if (t === 0) tick.classList.add("first");
      else if (frac > 0.94) tick.classList.add("last");
      tlScale.appendChild(tick);
    }
  }

  function placeHead(t) {
    var d = duration();
    if (d > 0) tlHead.style.left = pct(t, d);
  }

  tlTrack.addEventListener("click", function (ev) {
    var d = duration();
    if (!(d > 0)) return;
    var rect = tlTrack.getBoundingClientRect();
    var t = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width)) * d;
    seek(t);
    var i = eventUnder(t);
    if (i !== -1) setActive(i);
  });

  if (typeof ResizeObserver === "function") {
    new ResizeObserver(function () { if (!resultEl.hidden && duration() > 0) { layoutMarks(); tlScale.textContent = ""; renderScale(duration()); } }).observe(tlTrack);
  }

  /* ---------- Lightbox ---------- */

  function openLightbox(src, t, i, opener) {
    lbOpener = opener;
    lbTime = t;
    lbEvent = i;
    lbImg.src = src;
    lbImg.alt = "Кадр-доказательство, " + fmtSeconds(t);
    var type = events[i] ? str(events[i].raw.type) : "";
    lbCaption.textContent = "Эпизод " + (i + 1) + " · " + fmtTime(t) + (type ? " · " + type : "");
    if (typeof lightbox.showModal === "function") lightbox.showModal();
    else lightbox.setAttribute("open", "");
    lbClose.focus();
  }

  function closeLightbox() {
    if (!lightbox.open) return;
    if (typeof lightbox.close === "function") lightbox.close();
    else lightbox.removeAttribute("open");
  }

  lightbox.addEventListener("close", function () {
    lbImg.removeAttribute("src");
    if (lbOpener && document.contains(lbOpener)) lbOpener.focus();
    lbOpener = null;
  });
  lbClose.addEventListener("click", closeLightbox);
  lbSeek.addEventListener("click", function () {
    var t = lbTime, i = lbEvent;
    lbOpener = null;
    closeLightbox();
    jumpTo(t, i, "card");
  });
  // a click on the dimmed backdrop (outside the dialog box) closes it
  lightbox.addEventListener("click", function (ev) {
    if (ev.target !== lightbox) return;
    var r = lightbox.getBoundingClientRect();
    if (ev.clientX < r.left || ev.clientX > r.right || ev.clientY < r.top || ev.clientY > r.bottom) closeLightbox();
  });
  lightbox.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape" && typeof lightbox.showModal !== "function") closeLightbox();
  });

  /* ---------- New check ---------- */

  againBtn.addEventListener("click", function () {
    newRun();
    jobId = null;
    setHash(null);
    statusCard.hidden = true;
    setBusy(false);
    scrollIntoViewSoft($("upload"), "start");
    try { fileEl.focus({ preventScroll: true }); } catch (e) { fileEl.focus(); }
  });

  /* ---------- Start ---------- */

  // The verdict title is Cyrillic in the display face (its Cyrillic file is used nowhere else on the first
  // screen). Fetch it now so the verdict does not wait for the font when it appears.
  try { if (document.fonts && document.fonts.load) document.fonts.load("400 40px Anton", "Нарушение").catch(function () {}); } catch (e) { /* ignore */ }

  var startId = jobFromHash();
  if (startId) resume(startId);
})();
