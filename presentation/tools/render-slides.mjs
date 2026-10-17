// Render every slide to PNG and check the deck.
//
//   node tools/render-slides.mjs [deck.html] [out-dir]      defaults: index.html, build/render
//
// Writes slide-01.png ... and overview.png, then reports per slide:
//   - layout problems: content outside the slide, text wider or taller than its box,
//     content running into the footer
//   - placeholders still on the slide ([...], XX, X.X)
//   - speaker-note timing ("About N s") against <body data-pitch-seconds>
// Exits with 1 if there are layout problems.
import { mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { launch } from "./cdp.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const deckFile = resolve(process.argv[2] || join(here, "..", "index.html"));
const outDir = resolve(process.argv[3] || join(here, "..", "build", "render"));
mkdirSync(outDir, { recursive: true });

// Runs in the page: inspect slide i.
const inspect = (i) => {
  const slide = document.querySelectorAll(".deck > .slide")[i];
  const S = slide.getBoundingClientRect();
  const skip = ".closing-art, .title-art, .arch-wires, .notes, .pin";
  const hasFooter = slide.matches(".std, .statement");
  const describe = (el) => {
    const cls = el.getAttribute("class");
    const text = (el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 50);
    return `<${el.tagName.toLowerCase()}${cls ? ` class="${cls}"` : ""}> "${text}"`;
  };
  const problems = [];
  const flagged = new Set();
  const flag = (el, why) => {
    for (let p = el.parentElement; p && p !== slide; p = p.parentElement) if (flagged.has(p)) return;
    flagged.add(el);
    problems.push(`${why}: ${describe(el)}`);
  };
  for (const el of slide.querySelectorAll("*")) {
    if (el.closest(skip)) continue;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    const style = getComputedStyle(el);
    if (r.left < S.left - 1 || r.top < S.top - 1 || r.right > S.right + 1 || r.bottom > S.bottom + 1) flag(el, "outside the slide");
    else if (hasFooter && r.bottom > S.top + 975 && !el.closest(".photo")) flag(el, "runs into the footer");
    else if (style.display !== "inline" && !el.matches(".shot, .avatar, .photo") && el.clientWidth && el.scrollWidth > el.clientWidth + 2) flag(el, "text wider than its box");
    else {
      // Taller than its box: an in-flow child's layout box ends below this element's box.
      const below = [...el.children].find((c) => {
        const cs = getComputedStyle(c);
        return !c.closest(skip) && cs.display !== "inline" && cs.position !== "absolute" && c.getBoundingClientRect().bottom > r.bottom + 4;
      });
      if (below) flag(below, "does not fit in its box");
    }
  }
  const visible = [...slide.querySelectorAll("*")]
    .filter((el) => !el.closest(".notes") && [...el.childNodes].some((n) => n.nodeType === 3))
    .map((el) => [...el.childNodes].filter((n) => n.nodeType === 3).map((n) => n.textContent).join(" "))
    .join(" ");
  const placeholders = [...new Set(visible.match(/\[[^\]\n\s][^\]\n]{0,59}\]|[+−-]?\bX{2,}%?|\bX\.X\b/g) || [])];
  const shotsEmpty = [...slide.querySelectorAll(".shot")].filter((s) => !s.querySelector("img, video")).length;
  const notes = slide.querySelector(".notes")?.textContent || "";
  const seconds = Number((notes.match(/About\s+(\d+)\s*s/i) || [])[1] || 0);
  return { title: slide.dataset.title || "", problems: problems.slice(0, 8), placeholders, shotsEmpty, seconds };
};

const browser = await launch({ width: 1920, height: 1080, scale: 1 });
let failed = false;
try {
  await browser.goto(pathToFileURL(deckFile).href, { wait: 300 });
  const count = await browser.evaluate("deck.count");
  const target = await browser.evaluate("Number(document.body.dataset.pitchSeconds || 0)");
  let total = 0;
  for (let i = 0; i < count; i++) {
    await browser.evaluate(`deck.go(${i})`);
    await browser.sleep(150);
    const nn = String(i + 1).padStart(2, "0");
    await browser.screenshot(join(outDir, `slide-${nn}.png`));
    const r = await browser.evaluate(`(${inspect})(${i})`);
    total += r.seconds;
    const bits = [];
    if (r.problems.length) bits.push(`${r.problems.length} layout problem(s)`);
    if (r.placeholders.length) bits.push(`placeholders: ${r.placeholders.join(", ")}`);
    if (r.shotsEmpty) bits.push(`${r.shotsEmpty} empty screenshot box(es)`);
    console.log(`slide ${nn} ${r.title.padEnd(22)} ${r.seconds ? `${r.seconds}s`.padStart(4) : "  - "}  ${bits.join(" · ") || "ok"}`);
    for (const p of r.problems) console.log(`    ${p}`);
    failed ||= r.problems.length > 0;
  }
  await browser.evaluate(`(() => {
    const d = document.querySelector(".deck");
    d.classList.add("is-overview");
    Object.assign(d.style, { position: "static", overflow: "visible" });
    document.body.style.overflow = "visible";
    document.documentElement.style.height = document.body.style.height = "auto";
  })()`);
  await browser.sleep(300);
  await browser.screenshot(join(outDir, "overview.png"), { full: true });
  const fmt = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  console.log(`\nnotes timing: ${fmt(total)}${target ? ` of ${fmt(target)}` : ""}${target && total > target ? "  (over!)" : ""}`);
  console.log(`images: ${outDir}/slide-*.png, overview.png`);
} finally {
  await browser.close();
}
process.exit(failed ? 1 : 0);
