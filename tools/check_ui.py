"""Browser check of the result page (static/) against a running API. Google Chrome, synthetic video only.

    # 1. a stub API on a free port (no models, no GPU; JOBS_DIR is a throwaway folder):
    PIPELINE=stub JOBS_DIR="$(mktemp -d)" API_KEY=ui-check RULES_STRICT=0 \\
        uv run uvicorn api:app --host 127.0.0.1 --port 8010
    # 2. this script, with Playwright from a throwaway env (it drives the installed Chrome, downloads no browser):
    uv run --no-project --with playwright python tools/check_ui.py --base http://127.0.0.1:8010 --key ui-check \\
        [--clip CLIP] [--shots DIR] [--timeout 120]

--clip   a short H.264 video to upload. Without it the script makes a 6 s synthetic one with
         ffmpeg -f lavfi -i testsrc=duration=6:size=640x360:rate=25 -c:v libx264 -pix_fmt yuv420p
         Never pass a dataset video: screenshots and frames from this run may be shown to people and tools.
--shots  folder for full-page screenshots at 1440 and 390 px (optional).

Needs ffmpeg (clip and one JPEG frame for the synthetic events) and Google Chrome. Not collected by pytest.

The first scenario runs the server's real answer end to end (upload, status, result, seek, reload with #job=) and
compares the page with the API's own JSON, so it holds for the stub and for a real pipeline (one real job; raise
--timeout). Every other state (several events, a dense cluster of 16 markers, clean, failed, slow stages, 413,
503s, unsafe source links, a result without the split fields, an unplayable video, the 6-minute cap) is produced in
the browser by answering the page's requests with page.route, reusing the first job's id and video URL, so no
further job is started. Exit code 0 when every check passes.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright

V_TEXT = {"violation": "Обнаружены признаки возможных нарушений", "no_violation": "Нарушение не обнаружено"}
SOURCE_HOSTS = {"adilet.zan.kz", "www.adilet.zan.kz", "nationalbank.kz", "www.nationalbank.kz"}
NOTE = "[проверка UI]"
results: list[tuple[str, bool, str]] = []


def ok(name: str, cond, detail="") -> None:
    results.append((name, bool(cond), str(detail)))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail != "" else ""), flush=True)


def expected_link_host(url: str) -> str | None:
    """The page's rule: https, no credentials, one of the two publishers' hosts."""
    try:
        u = urlsplit(url)
    except ValueError:
        return None
    if u.scheme != "https" or u.username or u.password or u.hostname not in SOURCE_HOSTS:
        return None
    return u.hostname.removeprefix("www.")


def make_clip(folder: Path) -> Path:
    clip = folder / "synthetic.mov"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=duration=6:size=640x360:rate=25", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
                   check=True)
    return clip


def frame_b64(clip: Path, t: float) -> str:
    jpg = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(clip), "-frames:v", "1",
                          "-f", "image2", "-c:v", "mjpeg", "pipe:1"], check=True, capture_output=True).stdout
    return base64.b64encode(jpg).decode("ascii")


def event(tc, ts, te, src, frames, n, url="https://adilet.zan.kz/rus/docs/V0000000000", split=True):
    ev = {
        "type": f"{NOTE} Тип события {n}",
        "timecode_sec": round(tc, 1),
        "description": f"{NOTE} Объединённое описание события {n}. Основание: тестовый документ, п. {n}.",
        "evidence_frames": [{"timestamp_sec": round(t, 1), "image_base64": b} for t, b in frames],
        "type_id": "other",
        "rule": {"document": f"{NOTE} Тестовый документ {n}", "clause": f"п. {n}",
                 "quote": f"{NOTE} Тестовая цитата {n}.", "url": url},
        "confidence": 0.42,
    }
    if split:
        ev.update({"observation": f"{NOTE} Что видно в событии {n}.",
                   "justification": f"{NOTE} Почему событие {n} может быть нарушением.",
                   "type_source": src, "t_start": round(ts, 1), "t_end": round(te, 1)})
    return ev


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8010")
    ap.add_argument("--key", required=True)
    ap.add_argument("--clip", type=Path)
    ap.add_argument("--shots", type=Path)
    ap.add_argument("--timeout", type=float, default=120, help="seconds to wait for the real job")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    tmp = Path(tempfile.mkdtemp(prefix="ui-check-"))
    clip = a.clip or make_clip(tmp)
    if a.shots:
        a.shots.mkdir(parents=True, exist_ok=True)
    frame = frame_b64(clip, 1.0)
    wait_ms = int(a.timeout * 1000)

    def shot(page, name, full=True):
        if a.shots:
            page.wait_for_timeout(900)  # smooth scrolls settle
            page.screenshot(path=str(a.shots / f"{name}.png"), full_page=full)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)

        def new_page(width=1440, height=900, key=a.key, clock=False, go=True):
            ctx = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=1,
                                      extra_http_headers={"ngrok-skip-browser-warning": "1"})
            if key is not None:
                ctx.add_init_script(f"try{{localStorage.setItem('kua_api_key',{json.dumps(key)})}}catch(e){{}}")
            page = ctx.new_page()
            errs: list[str] = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
            if clock:
                page.clock.install()
            if go:
                page.goto(base + "/")
            return ctx, page, errs

        def upload(page):
            page.set_input_files("#file", str(clip))
            page.click("#submit")

        def no_hscroll(page):
            return page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

        def video_ready(page):
            page.wait_for_function("document.getElementById('video').readyState >= 1", timeout=20000)

        # ---------- 1. the server's own answer, end to end ----------
        ctx, page, errs = new_page()
        ok("styles applied", page.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(14, 11, 21)")
        ok("key restored and masked", page.input_value("#key") == a.key and page.get_attribute("#key", "type") == "password")
        page.click("#key-toggle")
        ok("key show/hide", page.get_attribute("#key", "type") == "text" and page.get_attribute("#key-toggle", "aria-pressed") == "true")
        page.click("#key-toggle")
        upload(page)
        expect(page.locator("#status")).to_be_visible(timeout=10000)
        expect(page.locator("#result")).to_be_visible(timeout=wait_ms)
        job_id = page.evaluate("location.hash.slice(5)")
        ok("hash holds the job id", len(job_id) == 12, job_id)
        api = page.evaluate("""id => fetch('/api/v1/jobs/' + id, {headers: {'X-API-Key': localStorage.getItem('kua_api_key')}})
                                     .then(r => r.json())""", job_id)
        video_url = api.get("video_url", "")
        viols = sorted(api.get("violations", []), key=lambda v: v["timecode_sec"])
        ok("verdict text exact (DOM)", page.text_content("#verdict-title") == V_TEXT.get(api["verdict"]), api["verdict"])
        ok("verdict rendered as worded", page.inner_text("#verdict-title").strip() == V_TEXT.get(api["verdict"]))
        ok("status kept as a finished step", page.text_content("#stage-now") == "Проверка завершена"
           and page.locator("#stepper > li.current").count() == 0)
        ok("one card per violation", page.locator(".event-card").count() == len(viols)
           and (bool(viols) or page.locator(".empty-card").count() == 1), len(viols))
        for k, v in enumerate(viols):
            card = page.locator(".event-card").nth(k)
            ok(f"card {k + 1}: frames", card.locator(".frames img").count() == len(v["evidence_frames"]))
            alts = card.locator(".frames img").evaluate_all("a => a.map(i => i.alt)")
            ok(f"card {k + 1}: frame alt text", all(x.startswith("Кадр-доказательство, ") and x.endswith(" с") for x in alts), alts)
            want = expected_link_host(v.get("rule", {}).get("url", ""))
            links = card.locator(".src-link")
            ok(f"card {k + 1}: source link only for an official https URL", (links.count() == 1) == (want is not None)
               and (want is None or links.inner_text().strip() == f"Открыть на {want}"), v.get("rule", {}).get("url"))
            if "type_source" in v:
                tag = {"judge": "оценка модели", "fallback": "по каталогу"}.get(v["type_source"])
                ok(f"card {k + 1}: source tag", card.locator(".tag").text_content() == tag)
        ok("confidence not shown", page.evaluate("!/уверенност|confidence|вероятност/i.test(document.getElementById('events').textContent)"))
        video_ready(page)
        ok("video plays the server copy", "/video?token=" in page.evaluate("document.getElementById('video').currentSrc")
           and page.evaluate("document.getElementById('video').videoWidth") > 0)
        clip_dur = page.evaluate("document.getElementById('video').duration")
        if viols:
            tc = viols[0]["timecode_sec"]
            page.locator(".tl-mark").first.click()
            page.wait_for_timeout(400)
            ct = page.evaluate("document.getElementById('video').currentTime")
            ok("marker click seeks to the timecode", abs(ct - tc) < 0.15, f"{ct} vs {tc}")
            ok("marker click highlights its card", "is-current" in page.locator(".event-card").first.get_attribute("class"))
            if page.locator(".tl-band").count():
                page.evaluate("document.getElementById('video').currentTime = 0")
                page.wait_for_timeout(300)
                page.locator(".tl-band").first.click()
                page.wait_for_timeout(400)
                ok("band click seeks to the timecode", abs(page.evaluate("document.getElementById('video').currentTime") - tc) < 0.15)
        ok("no horizontal scroll at 1440", no_hscroll(page))
        shot(page, "real-result-1440")
        job_url = page.url
        page.reload()
        expect(page.locator("#result")).to_be_visible(timeout=20000)
        ok("reload with #job= restores the verdict", page.text_content("#verdict-title") == V_TEXT.get(api["verdict"]))
        ok("reload restores the cards", page.locator(".event-card").count() == len(viols))
        video_ready(page)
        ok("reload: server video plays", page.evaluate("document.getElementById('video').videoWidth") > 0)
        ok("no page errors (1440)", not errs, "; ".join(errs))
        ctx.close()

        ctx, page, errs = new_page(390, 844, go=False)
        page.goto(job_url)
        expect(page.locator("#result")).to_be_visible(timeout=20000)
        video_ready(page)
        page.wait_for_timeout(300)
        ok("no horizontal scroll at 390", no_hscroll(page), page.evaluate("document.documentElement.scrollWidth"))
        if viols:
            page.locator(".tc-btn").first.click()
            page.wait_for_timeout(400)
            ok("card timecode seeks (390)", abs(page.evaluate("document.getElementById('video').currentTime") - viols[0]["timecode_sec"]) < 0.15)
        shot(page, "real-result-390")
        ok("no page errors (390)", not errs, "; ".join(errs))
        ctx.close()

        # ---------- helpers for the answered (synthetic) states; they reuse the real job's id and video ----------
        def fake_job(page, bodies, analyze_status=200, analyze_body=None):
            """POST /analyze answers with the real job id; GET /jobs/<id> answers from `bodies` (a list used in
            order, the last one repeated). A body of None passes the request to the server."""
            calls = {"n": 0}
            page.route("**/api/v1/analyze", lambda r: r.fulfill(
                status=analyze_status, content_type="application/json",
                body=json.dumps(analyze_body or {"job_id": job_id, "status": "processing"})))

            def jobs(route):
                if not route.request.url.split("?")[0].endswith("/" + job_id):
                    return route.continue_()
                b = bodies[min(calls["n"], len(bodies) - 1)]
                calls["n"] += 1
                if b is None:
                    return route.continue_()
                if isinstance(b, int):
                    return route.fulfill(status=b, content_type="application/json", body='{"detail":"busy"}')
                return route.fulfill(status=200, content_type="application/json", body=json.dumps(b))
            page.route("**/api/v1/jobs/*", jobs)
            return calls

        def done(events, verdict="violation"):
            return {"job_id": job_id, "status": "done", "verdict": verdict, "violations": events, "video_url": video_url}

        d = clip_dur if clip_dur and clip_dur > 0 else 6.0
        three = [event(d * 0.13, d * 0.03, d * 0.30, "judge", [(d * 0.13, frame), (d * 0.03, frame)], 1),
                 event(d * 0.43, d * 0.35, d * 0.55, "fallback", [(d * 0.43, frame)], 2,
                       url="https://www.nationalbank.kz/ru/npa/0"),
                 event(d * 0.82, d * 0.72, d * 0.97, "judge", [(d * 0.82, frame), (d * 0.72, frame), (d * 0.97, frame)], 3)]

        # ---------- 2. several events ----------
        for width in (1440, 390):
            ctx, page, errs = new_page(width, 900 if width > 400 else 844)
            fake_job(page, [done(list(reversed(three)))])
            upload(page)
            expect(page.locator("#result")).to_be_visible(timeout=20000)
            video_ready(page)
            page.wait_for_timeout(300)
            ok(f"three events sorted by timecode ({width})", page.locator(".ev-type").all_text_contents()
               == [f"{NOTE} Тип события {n}" for n in (1, 2, 3)])
            ok(f"judge/fallback tags ({width})", page.locator(".event-card .tag").all_text_contents()
               == ["оценка модели", "по каталогу", "оценка модели"])
            ok(f"event count ({width})", page.text_content("#verdict-meta") == "3 эпизода для проверки")
            ok(f"nationalbank link ({width})", page.locator(".event-card").nth(1).locator(".src-link").inner_text().strip()
               == "Открыть на nationalbank.kz")
            ok(f"no horizontal scroll ({width}, three events)", no_hscroll(page))
            page.evaluate(f"const v = document.getElementById('video'); v.muted = true; v.currentTime = {d * 0.75}; v.play()")
            page.wait_for_timeout(700)
            ok(f"card under the playhead highlighted while playing ({width})", page.locator(".event-card").evaluate_all(
                "a => a.map(x => x.classList.contains('is-current'))") == [False, False, True])
            page.evaluate("document.getElementById('video').pause()")
            if width == 1440:
                page.focus(".tl-mark >> nth=0")
                page.keyboard.press("Enter")
                page.wait_for_timeout(400)
                ok("keyboard Enter on a marker seeks", abs(page.evaluate("document.getElementById('video').currentTime") - round(d * 0.13, 1)) < 0.15)
                page.locator(".event-card").nth(0).locator(".frame-open").nth(0).click()
                expect(page.locator("#lightbox")).to_be_visible()
                ok("lightbox shows the full frame", page.evaluate("document.getElementById('lb-img').naturalWidth") > 0)
                shot(page, "lightbox-1440", full=False)
                page.keyboard.press("Escape")
                page.wait_for_timeout(200)
                ok("Esc closes the lightbox, focus returns", page.locator("#lightbox").is_hidden()
                   and page.evaluate("document.activeElement.classList.contains('frame-open')"))
                page.locator(".event-card").nth(2).locator(".frame-tc").nth(0).click()
                page.wait_for_timeout(400)
                ok("frame time click seeks", abs(page.evaluate("document.getElementById('video').currentTime") - round(d * 0.72, 1)) < 0.15)
            shot(page, f"multi-result-{width}")
            ok(f"no page errors ({width}, three events)", not errs, "; ".join(errs))
            ctx.close()

        # ---------- 3. 16 events, clustered: every marker stays visible and clickable ----------
        tcs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.50, 3.00, 4.20, 5.60, 5.70, 5.80, 5.90, 5.95]
        dense = [event(d * t / 6, max(0, d * t / 6 - 0.2), min(d, d * t / 6 + 0.2), "fallback", [(d * t / 6, frame)], n + 1)
                 for n, t in enumerate(tcs)]
        for width in (1440, 390):
            ctx, page, errs = new_page(width, 900 if width > 400 else 844)
            fake_job(page, [done(dense)])
            upload(page)
            expect(page.locator("#result")).to_be_visible(timeout=20000)
            video_ready(page)
            page.wait_for_timeout(300)
            hits = page.evaluate("""() => [...document.querySelectorAll('.tl-mark')].map(m => {
                m.scrollIntoView({block: 'center'});
                const r = m.getBoundingClientRect();
                return document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2) === m; })""")
            ok(f"16 clustered markers all clickable ({width})", len(hits) == 16 and all(hits), hits)
            boxes = page.locator(".tl-mark").evaluate_all("a => a.map(m => m.getBoundingClientRect()).map(r => [r.left, r.top, r.right, r.bottom])")
            overlap = any(a1 < b2 - 0.5 and b1 < a2 - 0.5 and a3 < b4 - 0.5 and b3 < a4 - 0.5
                          for i, (a1, a3, a2, a4) in enumerate(boxes) for (b1, b3, b2, b4) in boxes[i + 1:])
            ok(f"no two markers overlap ({width})", not overlap)
            strip = page.evaluate("document.getElementById('tl-marks').getBoundingClientRect().height")
            ok(f"marker strip at most 4 rows ({width})", strip <= 120, strip)
            page.locator(".tl-mark").nth(12).click()
            page.wait_for_timeout(400)
            ok(f"a clustered marker seeks to its own timecode ({width})",
               abs(page.evaluate("document.getElementById('video').currentTime") - round(d * tcs[12] / 6, 1)) < 0.15)
            ok(f"no horizontal scroll ({width}, 16 events)", no_hscroll(page))
            page.evaluate("document.getElementById('timeline').scrollIntoView({block: 'center'})")
            shot(page, f"dense-timeline-{width}", full=False)
            ctx.close()

        # ---------- 4. source links: only https on the two publishers' hosts ----------
        urls = ["https://adilet.zan.kz/rus/docs/V1", "https://www.adilet.zan.kz/rus/docs/V2", "https://nationalbank.kz/ru/3",
                "http://adilet.zan.kz/rus/docs/V4", "https://evil.example.com/adilet.zan.kz", "https://adilet.zan.kz.evil.example/5",
                "javascript:alert(1)", "https://user@adilet.zan.kz/6", "not a url", ""]
        linked = [event(d * (k + 1) / 12, d * (k + 1) / 12, d * (k + 1) / 12 + 0.1, "fallback", [(d * (k + 1) / 12, frame)], k + 1, url=u)
                  for k, u in enumerate(urls)]
        ctx, page, errs = new_page()
        fake_job(page, [done(linked)])
        upload(page)
        expect(page.locator("#result")).to_be_visible(timeout=20000)
        for k, u in enumerate(urls):
            card = page.locator(".event-card").nth(k)
            want = expected_link_host(u)
            got = card.locator(".src-link")
            ok(f"link for {u!r}: {'shown' if want else 'none'}", (got.count() == 1) == (want is not None)
               and (want is None or (got.inner_text().strip() == f"Открыть на {want}" and got.get_attribute("href") == u
                                     and got.get_attribute("target") == "_blank" and "noopener" in got.get_attribute("rel"))))
            ok(f"citation text kept for {u!r}", card.locator(".rule-doc").count() == 1 and card.locator(".rule-clause").count() == 1
               and card.locator("blockquote").count() == 1)
        ctx.close()

        # ---------- 5. a result stored before the split fields existed ----------
        ctx, page, errs = new_page()
        fake_job(page, [done([event(d * 0.3, 0, 0, "", [(d * 0.3, frame)], 1, split=False)])])
        upload(page)
        expect(page.locator("#result")).to_be_visible(timeout=20000)
        labels = page.locator(".event-card .ev-sec > .label").all_text_contents()
        ok("old result: description shown, split sections hidden", labels[0] == "Описание" and "Что видно" not in labels, labels)
        ok("old result: no source tag", page.locator(".event-card .tag").count() == 0)
        ctx.close()

        # ---------- 6. clean, failed, slow stages, live region ----------
        ctx, page, errs = new_page()
        fake_job(page, [done([], "no_violation")])
        upload(page)
        expect(page.locator("#result")).to_be_visible(timeout=20000)
        ok("clean verdict exact + mint", page.text_content("#verdict-title") == V_TEXT["no_violation"]
           and "is-clean" in page.get_attribute("#verdict", "class"))
        ok("clean: count line", page.text_content("#verdict-meta") == "Эпизодов для проверки нет")
        shot(page, "clean-1440", full=False)
        ctx.close()

        ctx, page, errs = new_page()
        fake_job(page, [{"job_id": job_id, "status": "failed", "error": "VideoError: cannot open video: synthetic", "video_url": video_url}])
        upload(page)
        expect(page.locator("#error")).to_be_visible(timeout=20000)
        ok("failed job: Russian message + raw details", "Не удалось прочитать видео" in page.text_content("#error-text")
           and page.locator("#error-details").is_visible())
        ok("failed job: status card hidden", page.locator("#status").is_hidden())
        ctx.close()

        for width in (1440, 390):
            ctx, page, errs = new_page(width, 900 if width > 400 else 844)
            proc = [{"job_id": job_id, "status": "processing", "stage": s, "video_url": video_url, "elapsed_sec": e}
                    for s, e in (("queued", 1.0), ("decoding", 3.1), ("grounding", 5.2), ("grounding", 7.2),
                                 ("verifying", 9.3), ("verifying", 11.3), ("writing", 13.4), ("mystery_stage", 15.4))]
            fake_job(page, proc + [done(three)])
            page.evaluate("""() => { window.__live = []; new MutationObserver(() => window.__live.push(
                document.getElementById('stage-now').textContent)).observe(document.getElementById('stage-now'),
                {childList: true, characterData: true, subtree: true}); }""")
            upload(page)
            page.wait_for_function("document.getElementById('stage-now').textContent === 'Проверка фрагментов'", timeout=30000)
            page.wait_for_timeout(1200)
            ok(f"status: job id + clock ({width})", len(page.text_content("#job-id")) == 12 and ":" in page.text_content("#elapsed"))
            ok(f"no horizontal scroll ({width}, status)", no_hscroll(page))
            shot(page, f"status-{width}", full=False)
            expect(page.locator("#result")).to_be_visible(timeout=30000)
            steps = page.locator("#stepper > li .txt").all_text_contents()
            ok(f"stepper: extra and unknown stages shown as they arrive ({width})",
               steps == ["Загрузка файла", "В очереди", "Чтение видео", "Поиск событий", "Проверка фрагментов",
                         "Формирование результата", "mystery_stage"], steps)
            live = page.evaluate("window.__live")
            ok(f"live region changes only on stage changes ({width})", len(live) == len(set(live)), live)
            ctx.close()

        # ---------- 7. errors on the way ----------
        ctx, page, errs = new_page(key="wrong-key")
        upload(page)
        expect(page.locator("#error")).to_be_visible(timeout=20000)
        ok("401 from the server -> wrong key", "Неверный API-ключ" in page.text_content("#error-text"))
        ctx.close()

        ctx, page, errs = new_page()
        page.click("#submit")
        ok("no file -> message", "Выберите видеофайл" in page.text_content("#error-text"))
        page.fill("#key", "")
        page.set_input_files("#file", str(clip))
        page.click("#submit")
        ok("no key -> message", "Введите API-ключ" in page.text_content("#error-text"))
        page.fill("#key", a.key)
        page.route("**/api/v1/analyze", lambda r: r.abort())
        page.click("#submit")
        expect(page.locator("#error")).to_be_visible(timeout=10000)
        ok("network failure -> message + retry", "Нет связи с сервером" in page.text_content("#error-text")
           and page.locator("#error-action").is_visible())
        page.unroute("**/api/v1/analyze")
        calls = fake_job(page, [503, 503, done(three)])
        page.click("#error-action")
        expect(page.locator("#retry")).to_be_visible(timeout=10000)
        ok("transient 503 -> retry note", "повторяю запрос" in page.text_content("#retry"))
        expect(page.locator("#result")).to_be_visible(timeout=20000)
        ok("result after two 503s", page.text_content("#verdict-title") == V_TEXT["violation"], calls["n"])
        page.unroute("**/api/v1/analyze")
        page.unroute("**/api/v1/jobs/*")
        page.route("**/api/v1/analyze", lambda r: r.fulfill(status=413, content_type="application/json", body='{"detail":"file too large"}'))
        page.click("#again")
        upload(page)
        expect(page.locator("#error")).to_be_visible(timeout=10000)
        ok("413 -> too large with the limit", "слишком большой" in page.text_content("#error-text")
           and "200 МБ" in page.text_content("#error-text"))
        page.evaluate("""() => {
          const f = new File([new Uint8Array(10)], 'big.mov', {type: 'video/quicktime'});
          Object.defineProperty(f, 'size', {value: 250 * 1024 * 1024});
          const ev = new DragEvent('drop', {bubbles: true, cancelable: true});
          Object.defineProperty(ev, 'dataTransfer', {value: {files: [f], types: ['Files']}});
          document.getElementById('drop').dispatchEvent(ev); }""")
        ok("dropped file over the limit -> message before upload", "Максимальный размер — 200 МБ" in page.text_content("#error-text")
           and "big.mov" in page.text_content("#drop"))
        ctx.close()

        ctx, page, errs = new_page(key=None, go=False)
        page.goto(job_url)
        expect(page.locator("#notice")).to_be_visible()
        page.fill("#key", a.key)
        page.click("#notice-action")
        expect(page.locator("#result")).to_be_visible(timeout=20000)
        ok("link without a saved key -> notice, then the result", page.text_content("#verdict-title") == V_TEXT.get(api["verdict"]))
        ctx.close()

        # ---------- 8. unplayable video: local copy after an upload, else a note and the frames ----------
        ctx, page, errs = new_page()
        page.route("**/video?token=*", lambda r: r.fulfill(status=404, body="no"))
        fake_job(page, [done(three)])
        upload(page)
        expect(page.locator("#result")).to_be_visible(timeout=20000)
        video_ready(page)
        ok("server video fails -> the uploaded file plays", page.evaluate("document.getElementById('video').currentSrc").startswith("blob:"))
        page.reload()
        expect(page.locator("#result")).to_be_visible(timeout=20000)
        page.wait_for_function("!document.getElementById('playnote').hidden", timeout=20000)
        ok("no playable video -> note, frames kept", page.locator(".frames img").count() >= 1 and page.locator("#player").is_hidden())
        ctx.close()

        # ---------- 9. the 6-minute cap (fake clock) ----------
        ctx, page, errs = new_page(clock=True)
        fake_job(page, [{"job_id": job_id, "status": "processing", "stage": "verifying", "video_url": video_url, "elapsed_sec": 12.0}])
        upload(page)
        page.clock.run_for(5000)
        expect(page.locator("#stage-now")).to_have_text("Проверка фрагментов")
        page.clock.run_for(6 * 60 * 1000 + 5000)
        expect(page.locator("#error")).to_be_visible()
        ok("6-minute cap -> message with the job id, retry, submit enabled",
           "дольше 6 минут" in page.text_content("#error-text") and job_id in page.text_content("#error-text")
           and page.locator("#error-action").is_visible() and page.is_enabled("#submit"))
        ctx.close()

        # ---------- 10. keyboard, fonts, widths ----------
        ctx, page, errs = new_page()
        page.wait_for_function("document.fonts.status === 'loaded'")
        page.wait_for_timeout(300)
        faces = page.evaluate("[...document.fonts].map(f => f.status)")
        ok("all six font faces load", len(faces) == 6 and all(s == "loaded" for s in faces), faces)
        order = []
        for _ in range(4):
            page.keyboard.press("Tab")
            order.append(page.evaluate("document.activeElement.id"))
        ok("tab order key, toggle, file, submit", order == ["key", "key-toggle", "file", "submit"], order)
        page.keyboard.press("Shift+Tab")
        ok("drop zone shows focus", page.evaluate("getComputedStyle(document.getElementById('drop')).outlineStyle") == "solid")
        ctx.close()
        for width in (320, 768, 1024, 1920):
            ctx, page, errs = new_page(width, go=False)
            fake_job(page, [done(dense)])
            page.goto(base + "/")
            upload(page)
            expect(page.locator("#result")).to_be_visible(timeout=20000)
            page.wait_for_timeout(400)
            ok(f"no horizontal scroll at {width} (16 events)", no_hscroll(page), page.evaluate("document.documentElement.scrollWidth"))
            ctx.close()

        browser.close()

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    for name, _, detail in failed:
        print(f"FAILED: {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
