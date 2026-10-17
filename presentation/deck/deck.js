// Deck runtime: fit-to-window scaling, keyboard navigation, overview grid,
// speaker notes, rehearsal timer, architecture connectors and an edit mode
// that saves back into the HTML file. No dependencies; works from file://.
(() => {
  const W = 1920;
  const H = 1080;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const deck = document.querySelector(".deck");
  let slides = [...deck.querySelectorAll(":scope > .slide")];
  const printing = new URLSearchParams(location.search).has("print");
  let current = 0;

  const ui = document.createElement("div");
  ui.className = "deck-ui";
  ui.innerHTML =
    '<div class="deck-timer" hidden></div>' +
    '<div class="deck-notes" hidden></div>' +
    '<div class="deck-editbar" hidden>' +
    '<span class="deck-editbar-title">Edit mode</span>' +
    '<span class="deck-editbar-hint">Click text to type · drop or paste an image on a box · N notes</span>' +
    '<span class="deck-dirty" hidden>● Unsaved</span>' +
    '<button type="button" data-act="save">Save ⌘S</button>' +
    '<button type="button" data-act="exit">Done E</button>' +
    "</div>" +
    '<div class="deck-toast" hidden></div>' +
    '<input type="file" accept="image/*,video/*" hidden>';
  document.body.append(ui);
  const timerEl = ui.querySelector(".deck-timer");
  const notesEl = ui.querySelector(".deck-notes");
  const editbarEl = ui.querySelector(".deck-editbar");
  const dirtyEl = ui.querySelector(".deck-dirty");
  const toastEl = ui.querySelector(".deck-toast");
  const fileInput = ui.querySelector('input[type="file"]');

  function fit() {
    const scale = Math.min(innerWidth / W, innerHeight / H);
    deck.style.setProperty("--scale", scale);
  }

  function go(index, { updateHash = true } = {}) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    slides.forEach((slide, i) => {
      const active = i === current;
      slide.classList.toggle("is-active", active);
      slide.setAttribute("aria-hidden", String(!active));
      for (const video of slide.querySelectorAll("video[data-autoplay]")) {
        if (active) video.play().catch(() => {});
        else video.pause();
      }
    });
    if (updateHash) history.replaceState(null, "", `#${current + 1}`);
    renderNotes();
    if (deck.classList.contains("is-overview")) slides[current].scrollIntoView({ block: "nearest" });
  }

  function fromHash() {
    const n = parseInt(location.hash.slice(1), 10);
    return Number.isFinite(n) ? n - 1 : 0;
  }

  let toastTimer = null;
  function toast(message, kind = "info") {
    toastEl.textContent = message;
    toastEl.dataset.kind = kind;
    toastEl.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toastEl.hidden = true; }, kind === "error" ? 6000 : 2600);
  }

  // ----- Overview -----

  function toggleOverview(force) {
    const on = deck.classList.toggle("is-overview", force);
    if (on) slides[current].scrollIntoView({ block: "center" });
  }

  deck.addEventListener("click", (event) => {
    if (!deck.classList.contains("is-overview")) return;
    const slide = event.target.closest(".slide");
    if (!slide) return;
    go(slides.indexOf(slide));
    toggleOverview(false);
  });

  // ----- Speaker notes -----

  function renderNotes() {
    if (notesEl.hidden) return;
    const slide = slides[current];
    const notes = slide.querySelector(".notes");
    const title = slide.dataset.title || "";
    const hint = editor.on
      ? " · ⌥↵ duplicate block · ⌥⇧⌫ remove block · ⇧D duplicate slide · ⇧←/→ move slide · ⇧⌫ delete slide"
      : " · ←/→ move · G overview · T timer · R reset · F full screen · E edit";
    notesEl.innerHTML =
      `<div class="deck-notes-head">${current + 1} / ${slides.length}${title ? " · " + title : ""}${hint}</div>` +
      `<div class="deck-notes-body">${notes ? notes.innerHTML : editor.on ? "<p></p>" : "<p>No notes on this slide.</p>"}</div>`;
    if (editor.on) notesEl.querySelector(".deck-notes-body").setAttribute("contenteditable", "true");
  }

  function toggleNotes() {
    notesEl.hidden = !notesEl.hidden;
    renderNotes();
  }

  // Edits in the notes panel are written back into the slide's <aside class="notes">.
  notesEl.addEventListener("input", () => {
    const slide = slides[current];
    let notes = slide.querySelector(".notes");
    if (!notes) {
      notes = document.createElement("aside");
      notes.className = "notes";
      slide.append(notes);
    }
    notes.innerHTML = notesEl.querySelector(".deck-notes-body").innerHTML;
    editor.markDirty();
  });

  // ----- Rehearsal timer (target from <body data-pitch-seconds>) -----

  const target = parseInt(document.body.dataset.pitchSeconds || "120", 10);
  let startedAt = null;
  let elapsedBefore = 0;
  let tick = null;

  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  function elapsed() {
    return elapsedBefore + (startedAt ? (performance.now() - startedAt) / 1000 : 0);
  }

  function renderTimer() {
    const s = elapsed();
    timerEl.textContent = `${fmt(s)} / ${fmt(target)}`;
    timerEl.classList.toggle("is-warn", s >= target * 0.8 && s < target);
    timerEl.classList.toggle("is-over", s >= target);
    timerEl.classList.toggle("is-paused", !startedAt);
  }

  function toggleTimer() {
    timerEl.hidden = false;
    if (startedAt) {
      elapsedBefore = elapsed();
      startedAt = null;
      clearInterval(tick);
    } else {
      startedAt = performance.now();
      tick = setInterval(renderTimer, 250);
    }
    renderTimer();
  }

  function resetTimer() {
    clearInterval(tick);
    startedAt = null;
    elapsedBefore = 0;
    timerEl.hidden = true;
  }

  // ----- Architecture diagrams -----
  // <div class="arch" data-edges="ui>api: HTTPS; api>vlm: :30000; api~>disk">, nodes carry data-node="id".
  // ">" is a solid arrow, "~>" a dashed one; the text after ":" is the edge label.

  function boxIn(el, container) {
    let x = 0;
    let y = 0;
    for (let e = el; e && e !== container; e = e.offsetParent) {
      x += e.offsetLeft;
      y += e.offsetTop;
    }
    return { x, y, w: el.offsetWidth, h: el.offsetHeight };
  }

  function arrowMarker(defs, id, color) {
    const marker = document.createElementNS(SVG_NS, "marker");
    for (const [k, v] of Object.entries({ id, viewBox: "0 0 12 12", refX: "11", refY: "6", markerWidth: "12", markerHeight: "12", markerUnits: "userSpaceOnUse", orient: "auto" })) marker.setAttribute(k, v);
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", "M0 0 12 6 0 12z");
    path.setAttribute("fill", color);
    marker.append(path);
    defs.append(marker);
  }

  let archCount = 0;
  function drawArch(arch) {
    arch.querySelector(":scope > .arch-wires")?.remove();
    const nodes = Object.fromEntries([...arch.querySelectorAll("[data-node]")].map((n) => [n.dataset.node, n]));
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.classList.add("arch-wires");
    svg.setAttribute("aria-hidden", "true");
    const defs = document.createElementNS(SVG_NS, "defs");
    const uid = `arch${++archCount}`;
    arrowMarker(defs, `${uid}-solid`, "#9b7bff");
    arrowMarker(defs, `${uid}-dashed`, "#8a82a0");
    svg.append(defs);
    arch.prepend(svg);

    for (const spec of (arch.dataset.edges || "").split(";")) {
      const m = spec.match(/^\s*([\w-]+)\s*(~?>)\s*([\w-]+)\s*(?::\s*(.*?))?\s*$/);
      if (!m) continue;
      const [, fromId, kind, toId, label] = m;
      if (!nodes[fromId] || !nodes[toId]) {
        console.warn(`arch: unknown node in "${spec.trim()}"`);
        continue;
      }
      const a = boxIn(nodes[fromId], arch);
      const b = boxIn(nodes[toId], arch);
      const gapX = Math.max(b.x - (a.x + a.w), a.x - (b.x + b.w));
      const gapY = Math.max(b.y - (a.y + a.h), a.y - (b.y + b.h));
      let d;
      let mid;
      if (gapX >= gapY) {
        const right = b.x >= a.x + a.w;
        const sx = right ? a.x + a.w : a.x;
        const ex = right ? b.x : b.x + b.w;
        const top = Math.max(a.y, b.y);
        const bottom = Math.min(a.y + a.h, b.y + b.h);
        if (bottom - top > 24) {
          const y = (top + bottom) / 2;
          d = `M${sx} ${y}H${ex}`;
          mid = [(sx + ex) / 2, y];
        } else {
          const sy = a.y + a.h / 2;
          const ey = b.y + b.h / 2;
          const mx = (sx + ex) / 2;
          d = `M${sx} ${sy}H${mx}V${ey}H${ex}`;
          mid = [mx, (sy + ey) / 2];
        }
      } else {
        const down = b.y >= a.y + a.h;
        const sy = down ? a.y + a.h : a.y;
        const ey = down ? b.y : b.y + b.h;
        const left = Math.max(a.x, b.x);
        const right = Math.min(a.x + a.w, b.x + b.w);
        if (right - left > 24) {
          const x = (left + right) / 2;
          d = `M${x} ${sy}V${ey}`;
          mid = [x, (sy + ey) / 2];
        } else {
          const sx = a.x + a.w / 2;
          const ex = b.x + b.w / 2;
          const my = (sy + ey) / 2;
          d = `M${sx} ${sy}V${my}H${ex}V${ey}`;
          mid = [(sx + ex) / 2, my];
        }
      }
      const dashed = kind === "~>";
      const path = document.createElementNS(SVG_NS, "path");
      path.setAttribute("d", d);
      if (dashed) path.classList.add("dashed");
      path.setAttribute("marker-end", `url(#${uid}-${dashed ? "dashed" : "solid"})`);
      svg.append(path);

      if (label) {
        const text = document.createElementNS(SVG_NS, "text");
        text.textContent = label.toUpperCase();
        text.setAttribute("x", mid[0]);
        text.setAttribute("y", mid[1]);
        svg.append(text);
        const bb = text.getBBox();
        const bg = document.createElementNS(SVG_NS, "rect");
        for (const [k, v] of Object.entries({ x: bb.x - 10, y: bb.y - 6, width: bb.width + 20, height: bb.height + 12 })) bg.setAttribute(k, v);
        svg.insertBefore(bg, text);
      }
    }
  }

  function drawAllArch(root = deck) {
    root.querySelectorAll(".arch[data-edges]").forEach(drawArch);
  }

  // ----- Edit mode -----

  const editor = (() => {
    const fileName = decodeURIComponent(location.pathname.split("/").pop() || "index.html");
    const handleKey = `dir:${location.href.split(/[?#]/)[0].replace(/[^/]*$/, "")}`;
    const MEDIA = ".shot, .avatar";
    const BLOCK = "li, tr, .card, .bar, .node, figure";
    let on = false;
    let dirty = false;
    let dir = null;
    let backedUp = false;
    let selectedMedia = null;
    const baseline = serializeDoc(document);

    // Serialize a document the way it should live on disk: no runtime classes,
    // no editor attributes, no injected UI.
    function serializeDoc(doc) {
      const root = doc.documentElement.cloneNode(true);
      root.querySelectorAll(".deck-ui, .arch-wires").forEach((n) => n.remove());
      root.querySelectorAll("[contenteditable]").forEach((n) => n.removeAttribute("contenteditable"));
      const strip = (el, ...classes) => {
        if (!el) return;
        el.classList.remove(...classes);
        if (!el.classList.length) el.removeAttribute("class");
      };
      root.querySelectorAll(".slide").forEach((s) => {
        strip(s, "is-active");
        s.removeAttribute("aria-hidden");
      });
      root.querySelectorAll(".is-selected").forEach((n) => strip(n, "is-selected"));
      const deckEl = root.querySelector(".deck");
      strip(deckEl, "is-overview");
      if (deckEl) {
        deckEl.style.removeProperty("--scale");
        if (!deckEl.getAttribute("style")) deckEl.removeAttribute("style");
      }
      strip(root.querySelector("body"), "is-editing");
      return `<!doctype html>\n${root.outerHTML}\n`;
    }

    const normalize = (text) => serializeDoc(new DOMParser().parseFromString(text, "text/html"));

    function markDirty() {
      dirty = true;
      dirtyEl.hidden = false;
    }

    // Text hosts: the outermost elements that hold text directly.
    function textHosts(root) {
      const hosts = [];
      const walk = (el) => {
        for (const child of el.children) {
          if (child.matches(".notes, svg, script, style, img, video, .pin, .closing-art, .title-art") || child.matches(MEDIA)) continue;
          const hasText = [...child.childNodes].some((n) => n.nodeType === Node.TEXT_NODE && n.textContent.trim());
          if (hasText) hosts.push(child);
          else walk(child);
        }
      };
      walk(root);
      return hosts;
    }

    function setEditing(next) {
      on = next;
      document.body.classList.toggle("is-editing", on);
      editbarEl.hidden = !on;
      for (const slide of slides) {
        for (const host of textHosts(slide)) {
          if (on) host.setAttribute("contenteditable", "true");
          else host.removeAttribute("contenteditable");
        }
      }
      if (!on) {
        document.activeElement?.blur?.();
        selectMedia(null);
      }
      document.execCommand("defaultParagraphSeparator", false, "p");
      renderNotes();
      toast(on ? "Edit mode: click any text. ⌘S saves into the file." : dirty ? "Edit mode off. You have unsaved changes (⌘S)." : "Edit mode off.");
    }

    function focusEnd(el) {
      el.focus();
      const range = document.createRange();
      range.selectNodeContents(el);
      range.collapse(false);
      const sel = getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }

    function selectAll(el) {
      el.focus();
      const range = document.createRange();
      range.selectNodeContents(el);
      const sel = getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }

    function duplicateBlock(host) {
      const block = host.closest(BLOCK);
      if (!block || !block.closest(".slide")) return;
      const copy = block.cloneNode(true);
      copy.classList.remove("is-selected");
      block.after(copy);
      const first = copy.matches('[contenteditable="true"]') ? copy : copy.querySelector('[contenteditable="true"]');
      if (first) selectAll(first);
      redraw(copy);
      markDirty();
    }

    function removeBlock(host) {
      const block = host.closest(BLOCK);
      if (!block || !block.parentElement) return;
      if (block.parentElement.children.length < 2) return toast("This is the last one; edit it instead.", "error");
      if (!confirm("Remove this block?")) return;
      const prev = block.previousElementSibling || block.nextElementSibling;
      block.remove();
      const focusable = prev?.matches('[contenteditable="true"]') ? prev : prev?.querySelector('[contenteditable="true"]');
      if (focusable) focusEnd(focusable);
      redraw(prev);
      markDirty();
    }

    // Keys while the caret is inside editable text.
    function hostKey(event, host) {
      if (event.key === "Escape") {
        event.preventDefault();
        host.blur();
        return;
      }
      if (host.classList.contains("deck-notes-body")) return;
      if (event.key === "Enter" && event.altKey) {
        event.preventDefault();
        duplicateBlock(host);
        return;
      }
      if (event.key === "Backspace" && event.altKey && event.shiftKey) {
        event.preventDefault();
        removeBlock(host);
        return;
      }
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        if (host.tagName === "LI") {
          // New list item after this one, same classes and child markup.
          const item = host.cloneNode(true);
          for (const n of [...item.childNodes]) if (n.nodeType === Node.TEXT_NODE) n.remove();
          item.prepend(document.createTextNode("New item"));
          host.after(item);
          selectAll(item);
          redraw(item);
          markDirty();
        } else {
          document.execCommand("insertLineBreak");
        }
        return;
      }
      if (event.key === "Backspace" && host.tagName === "LI" && !host.textContent.trim() && host.parentElement.children.length > 1) {
        event.preventDefault();
        const prev = host.previousElementSibling || host.nextElementSibling;
        host.remove();
        if (prev?.matches('[contenteditable="true"]')) focusEnd(prev);
        markDirty();
      }
    }

    // ----- Slides: duplicate, move, delete -----

    function refreshSlides() {
      slides = [...deck.querySelectorAll(":scope > .slide")];
    }

    function duplicateSlide() {
      const copy = slides[current].cloneNode(true);
      slides[current].after(copy);
      refreshSlides();
      drawAllArch(copy);
      go(current + 1);
      markDirty();
      toast("Slide duplicated.");
    }

    function moveSlide(delta) {
      const to = current + delta;
      if (to < 0 || to >= slides.length) return;
      const slide = slides[current];
      if (delta < 0) slides[to].before(slide);
      else slides[to].after(slide);
      refreshSlides();
      go(to);
      markDirty();
    }

    function deleteSlide() {
      if (slides.length < 2) return;
      if (!confirm(`Delete slide ${current + 1}${slides[current].dataset.title ? ` (${slides[current].dataset.title})` : ""}?`)) return;
      slides[current].remove();
      refreshSlides();
      go(Math.min(current, slides.length - 1));
      markDirty();
      toast("Slide deleted. The previous file is backed up on the first save.");
    }

    // Keys in edit mode when the caret is not in text. Returns true if handled.
    function slideKey(event) {
      if (event.shiftKey && (event.key === "D" || event.key === "d")) return duplicateSlide(), true;
      if (event.shiftKey && event.key === "ArrowLeft") return moveSlide(-1), true;
      if (event.shiftKey && event.key === "ArrowRight") return moveSlide(1), true;
      if (event.shiftKey && (event.key === "Backspace" || event.key === "Delete")) return deleteSlide(), true;
      if (selectedMedia) {
        if (event.key === "Enter") return fileInput.click(), true;
        if (event.key === "Backspace" || event.key === "Delete") return clearMedia(selectedMedia), true;
        if (event.key === "Escape") return selectMedia(null), true;
      }
      return false;
    }

    // ----- Images and videos -----

    function selectMedia(el) {
      selectedMedia?.classList.remove("is-selected");
      selectedMedia = el;
      el?.classList.add("is-selected");
    }

    function clearMedia(target) {
      target.querySelectorAll(":scope > img, :scope > video").forEach((n) => n.remove());
      markDirty();
      toast("Image removed.");
    }

    const slug = (name) =>
      name.toLowerCase().normalize("NFKD").replace(/[^\w.-]+/g, "-").replace(/-+/g, "-").replace(/^-|-(?=\.)/g, "") || "image";

    async function exists(folder, name) {
      try {
        await folder.getFileHandle(name);
        return true;
      } catch {
        return false;
      }
    }

    async function storeAsset(file) {
      const folder = await getDir();
      if (!folder) {
        toast("No folder access, so the image is embedded in the HTML.", "error");
        return new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.readAsDataURL(file);
        });
      }
      const assets = await folder.getDirectoryHandle("assets", { create: true });
      const ext = (file.name.match(/\.\w+$/) || [`.${(file.type.split("/")[1] || "png").replace("jpeg", "jpg")}`])[0];
      const base = file.name && file.name !== "image.png" ? slug(file.name.replace(/\.\w+$/, "")) : `slide-${current + 1}`;
      let name = `${base}${ext}`;
      for (let i = 2; await exists(assets, name); i++) name = `${base}-${i}${ext}`;
      const handle = await assets.getFileHandle(name, { create: true });
      const out = await handle.createWritable();
      await out.write(file);
      await out.close();
      return `assets/${name}`;
    }

    async function placeMedia(target, file) {
      if (!file || !/^(image|video)\//.test(file.type)) return toast("That is not an image or a video.", "error");
      const src = await storeAsset(file);
      target.querySelectorAll(":scope > img, :scope > video").forEach((n) => n.remove());
      if (target.classList.contains("avatar")) {
        for (const n of [...target.childNodes]) if (n.nodeType === Node.TEXT_NODE) n.remove();
      }
      let el;
      if (file.type.startsWith("video/")) {
        el = document.createElement("video");
        for (const attr of ["data-autoplay", "muted", "loop", "playsinline"]) el.setAttribute(attr, "");
        el.muted = true;
        el.src = src;
        el.play().catch(() => {});
      } else {
        el = document.createElement("img");
        el.src = src;
        el.alt = "";
      }
      target.append(el);
      markDirty();
      toast(src.startsWith("data:") ? "Image embedded." : `Saved ${src}.`);
    }

    deck.addEventListener("click", (event) => {
      if (!on || deck.classList.contains("is-overview")) return;
      selectMedia(event.target.closest(MEDIA));
    });

    deck.addEventListener("dblclick", (event) => {
      if (!on) return;
      const media = event.target.closest(MEDIA);
      if (!media) return;
      selectMedia(media);
      fileInput.click();
    });

    fileInput.addEventListener("change", async () => {
      const file = fileInput.files[0];
      fileInput.value = "";
      if (file && selectedMedia) await placeMedia(selectedMedia, file);
    });

    deck.addEventListener("dragover", (event) => {
      if (!on) return;
      const media = event.target.closest(MEDIA);
      if (!media) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
      selectMedia(media);
    });

    deck.addEventListener("drop", async (event) => {
      if (!on) return;
      const media = event.target.closest(MEDIA);
      if (!media) return;
      event.preventDefault();
      await placeMedia(media, event.dataTransfer.files[0]);
    });

    document.addEventListener("paste", async (event) => {
      if (!on) return;
      const host = event.target.closest?.('[contenteditable="true"]');
      const image = [...(event.clipboardData?.files || [])].find((f) => /^(image|video)\//.test(f.type));
      if (image && selectedMedia && !host) {
        event.preventDefault();
        await placeMedia(selectedMedia, image);
        return;
      }
      if (host) {
        // Paste as plain text so foreign styles never land in the deck.
        event.preventDefault();
        document.execCommand("insertText", false, event.clipboardData.getData("text/plain"));
      }
    });

    // Drag the numbered pins on an annotated screenshot.
    deck.addEventListener("pointerdown", (event) => {
      if (!on) return;
      const pin = event.target.closest(".pin");
      const area = pin?.closest(".annot");
      if (!area) return;
      event.preventDefault();
      const box = area.getBoundingClientRect();
      const clamp = (v) => Math.max(0, Math.min(100, v)).toFixed(1);
      const move = (e) => {
        pin.style.setProperty("--x", clamp(((e.clientX - box.left) / box.width) * 100));
        pin.style.setProperty("--y", clamp(((e.clientY - box.top) / box.height) * 100));
      };
      pin.setPointerCapture(event.pointerId);
      pin.addEventListener("pointermove", move);
      pin.addEventListener("pointerup", () => {
        pin.removeEventListener("pointermove", move);
        markDirty();
      }, { once: true });
    });

    let redrawTimer = null;
    function redraw(el) {
      const arch = el?.closest?.(".arch[data-edges]");
      if (!arch) return;
      clearTimeout(redrawTimer);
      redrawTimer = setTimeout(() => drawArch(arch), 120);
    }

    deck.addEventListener("input", (event) => {
      if (!on) return;
      markDirty();
      redraw(event.target);
    });

    // ----- Saving -----

    const idb = (mode, fn) =>
      new Promise((resolve, reject) => {
        const open = indexedDB.open("deck-editor", 1);
        open.onupgradeneeded = () => open.result.createObjectStore("handles");
        open.onerror = () => reject(open.error);
        open.onsuccess = () => {
          const tx = open.result.transaction("handles", mode);
          const req = fn(tx.objectStore("handles"));
          tx.oncomplete = () => resolve(req.result);
          tx.onerror = () => reject(tx.error);
        };
      });

    async function permitted(handle) {
      const opts = { mode: "readwrite" };
      return (await handle.queryPermission(opts)) === "granted" || (await handle.requestPermission(opts)) === "granted";
    }

    // The folder must hold this very deck: same file name and same <title>.
    async function isDeckFolder(folder) {
      try {
        const text = await (await (await folder.getFileHandle(fileName)).getFile()).text();
        return new DOMParser().parseFromString(text, "text/html").title === document.title;
      } catch {
        return false;
      }
    }

    async function getDir() {
      if (dir && (await permitted(dir))) return dir;
      if (!("showDirectoryPicker" in window)) return null;
      try {
        const stored = await idb("readonly", (s) => s.get(handleKey));
        if (stored && (await permitted(stored)) && (await isDeckFolder(stored))) return (dir = stored);
      } catch {
        // No IndexedDB here: ask for the folder every session.
      }
      toast(`Choose the folder that contains ${fileName}.`);
      try {
        const picked = await showDirectoryPicker({ id: "deck", mode: "readwrite" });
        if (!(await isDeckFolder(picked))) {
          toast(`That folder does not contain this ${fileName}.`, "error");
          return null;
        }
        dir = picked;
        await idb("readwrite", (s) => s.put(picked, handleKey)).catch(() => {});
        return dir;
      } catch (error) {
        if (error.name !== "AbortError") toast(error.message, "error");
        return null;
      }
    }

    let lastSaved = baseline;
    const stamp = () => new Date().toISOString().slice(0, 19).replace(/[-:]/g, "").replace("T", "-");

    async function writeText(handle, text) {
      const out = await handle.createWritable();
      await out.write(text);
      await out.close();
    }

    async function save() {
      document.activeElement?.blur?.();
      const html = serializeDoc(document);
      const folder = await getDir();
      if (!folder) {
        if ("showDirectoryPicker" in window) return toast("Not saved.", "error");
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([html], { type: "text/html" }));
        a.download = fileName;
        a.click();
        URL.revokeObjectURL(a.href);
        return toast(`This browser can't write files: downloaded ${fileName}, replace the one in the folder.`, "error");
      }
      const handle = await folder.getFileHandle(fileName);
      const onDisk = await (await handle.getFile()).text();
      if (normalize(onDisk) !== lastSaved &&
          !confirm(`${fileName} changed on disk after you opened it (for example, Claude edited it).\n\nOK overwrites those changes with what you see here. Cancel keeps the file on disk; reload the page to see it.`)) {
        return toast("Not saved.", "error");
      }
      if (!backedUp) {
        const backups = await folder.getDirectoryHandle(".backups", { create: true });
        await writeText(await backups.getFileHandle(`${fileName.replace(/\.html?$/, "")}-${stamp()}.html`, { create: true }), onDisk);
        backedUp = true;
      }
      await writeText(handle, html);
      lastSaved = normalize(html);
      dirty = false;
      dirtyEl.hidden = true;
      toast(`Saved ${fileName}.`);
    }

    editbarEl.addEventListener("click", (event) => {
      const act = event.target.closest("button")?.dataset.act;
      event.target.closest("button")?.blur();
      if (act === "save") save();
      if (act === "exit") setEditing(false);
    });

    addEventListener("beforeunload", (event) => {
      if (!dirty) return;
      event.preventDefault();
      event.returnValue = "";
    });

    return {
      get on() { return on; },
      toggle: () => setEditing(!on),
      hostKey,
      slideKey,
      markDirty,
      save,
      serialize: () => serializeDoc(document),
    };
  })();

  // ----- Keyboard -----

  const keys = {
    ArrowRight: () => go(current + 1),
    ArrowDown: () => go(current + 1),
    PageDown: () => go(current + 1),
    " ": () => go(current + 1),
    Enter: () => (deck.classList.contains("is-overview") ? toggleOverview(false) : go(current + 1)),
    ArrowLeft: () => go(current - 1),
    ArrowUp: () => go(current - 1),
    PageUp: () => go(current - 1),
    Backspace: () => go(current - 1),
    Home: () => go(0),
    End: () => go(slides.length - 1),
    Escape: () => toggleOverview(false),
    g: () => toggleOverview(),
    o: () => toggleOverview(),
    n: toggleNotes,
    t: toggleTimer,
    r: resetTimer,
    e: () => editor.toggle(),
    f: () => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen()),
  };

  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      editor.save();
      return;
    }
    if (event.target.closest("input, textarea, button")) return;
    const host = event.target.closest('[contenteditable="true"]');
    if (host) {
      editor.hostKey(event, host);
      return;
    }
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (editor.on && editor.slideKey(event)) {
      event.preventDefault();
      return;
    }
    const action = keys[event.key.length === 1 ? event.key.toLowerCase() : event.key];
    if (!action) return;
    event.preventDefault();
    if (event.key === " " && event.shiftKey) go(current - 1);
    else action();
  });

  // Touch: swipe left / right.
  let touchX = null;
  deck.addEventListener("touchstart", (e) => { touchX = e.touches[0].clientX; }, { passive: true });
  deck.addEventListener("touchend", (e) => {
    if (touchX === null) return;
    const dx = e.changedTouches[0].clientX - touchX;
    if (Math.abs(dx) > 50) go(current + (dx < 0 ? 1 : -1));
    touchX = null;
  });

  addEventListener("resize", fit);
  addEventListener("hashchange", () => go(fromHash(), { updateHash: false }));

  fit();
  drawAllArch();
  document.fonts.ready.then(() => drawAllArch());
  if (!printing) go(fromHash(), { updateHash: false });

  // For tools/render-slides.mjs and tests.
  window.deck = { go, get count() { return slides.length; }, serialize: editor.serialize };
})();
