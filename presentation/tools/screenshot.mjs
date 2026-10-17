// Screenshot a web page (e.g. the product UI) for the deck.
//
//   node tools/screenshot.mjs <url> <out.png> [--size 1600x1000] [--scale 2] [--wait 1500]
//                             [--selector "css"] [--full] [--eval "js"]...
//
// --selector  crop to one element        --full   whole page, not just the viewport
// --eval      JS to run after load, before the shot (click a tab, open a result...);
//             may repeat, promises are awaited
// --scale 2   retina-sharp image; the deck shows screenshots large
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { launch } from "./cdp.mjs";

const args = process.argv.slice(2);
const positional = [];
const opts = { size: "1600x1000", scale: "2", wait: "1500", evals: [] };
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === "--full") opts.full = true;
  else if (a === "--eval") opts.evals.push(args[++i]);
  else if (a.startsWith("--")) opts[a.slice(2)] = args[++i];
  else positional.push(a);
}
const [url, out] = positional;
if (!url || !out) {
  console.error("usage: node tools/screenshot.mjs <url> <out.png> [--size WxH] [--scale N] [--wait ms] [--selector css] [--full] [--eval js]");
  process.exit(2);
}

const [width, height] = opts.size.split("x").map(Number);
const browser = await launch({ width, height, scale: Number(opts.scale) });
try {
  await browser.goto(url, { wait: Number(opts.wait) });
  for (const js of opts.evals) {
    await browser.evaluate(js);
    await browser.sleep(Number(opts.wait));
  }
  let clip;
  if (opts.selector) {
    clip = await browser.evaluate(`(() => {
      const el = document.querySelector(${JSON.stringify(opts.selector)});
      if (!el) throw new Error("selector not found: " + ${JSON.stringify(opts.selector)});
      el.scrollIntoView({ block: "center" });
      const r = el.getBoundingClientRect();
      return { x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height };
    })()`);
  }
  mkdirSync(dirname(out), { recursive: true });
  await browser.screenshot(out, { clip, full: opts.full });
  console.log(`wrote ${out}`);
} finally {
  await browser.close();
}
