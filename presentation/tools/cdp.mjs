// Minimal Chrome DevTools Protocol driver for headless Chrome.
// Node 22+ (global fetch and WebSocket); no npm packages.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const found = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].find((path) => existsSync(path));
  if (!found) throw new Error("Chrome not found; set CHROME=/path/to/chrome");
  return found;
}

export async function launch({ width = 1920, height = 1080, scale = 1 } = {}) {
  const profile = mkdtempSync(join(tmpdir(), "deck-cdp-"));
  const chrome = spawn(findChrome(), [
    "--headless=new", "--disable-gpu", "--hide-scrollbars", "--remote-debugging-port=0",
    `--user-data-dir=${profile}`, `--window-size=${width},${height}`, "about:blank",
  ], { stdio: "ignore" });

  let port;
  for (let i = 0; i < 100 && !port; i++) {
    try {
      port = readFileSync(join(profile, "DevToolsActivePort"), "utf8").split("\n")[0];
    } catch {
      await sleep(100);
    }
  }
  if (!port) throw new Error("Chrome did not start");

  const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });

  let nextId = 0;
  const pending = new Map();
  const listeners = new Map();
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) reject(new Error(msg.error.message));
      else resolve(msg.result);
    } else if (msg.method) {
      for (const fn of listeners.get(msg.method) || []) fn(msg.params);
    }
  };

  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = ++nextId;
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    });

  const once = (method) =>
    new Promise((resolve) => {
      if (!listeners.has(method)) listeners.set(method, new Set());
      const fn = (params) => {
        listeners.get(method).delete(fn);
        resolve(params);
      };
      listeners.get(method).add(fn);
    });

  async function evaluate(expression) {
    const res = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (res.exceptionDetails) throw new Error(res.exceptionDetails.exception?.description || res.exceptionDetails.text);
    return res.result.value;
  }

  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: scale, mobile: false });

  return {
    send,
    evaluate,
    sleep,
    async goto(url, { wait = 0 } = {}) {
      const loaded = once("Page.loadEventFired");
      await send("Page.navigate", { url });
      await loaded;
      await evaluate("document.fonts.ready.then(() => true)");
      if (wait) await sleep(wait);
    },
    // clip is in CSS pixels, relative to the document.
    async screenshot(path, { clip, full = false } = {}) {
      let region = clip;
      if (full && !region) {
        const { cssContentSize: size } = await send("Page.getLayoutMetrics");
        region = { x: 0, y: 0, width: size.width, height: size.height };
      }
      const params = { format: "png", captureBeyondViewport: Boolean(region) };
      if (region) params.clip = { ...region, scale: 1 };
      const { data } = await send("Page.captureScreenshot", params);
      writeFileSync(path, Buffer.from(data, "base64"));
    },
    async close() {
      ws.close();
      chrome.kill();
      await sleep(300);
      rmSync(profile, { recursive: true, force: true });
    },
  };
}
