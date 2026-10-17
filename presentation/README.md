# Datathon ’26 pitch deck

HTML slide deck in the design of *How to Win the Datathon.pdf*: same fonts (Anton, Rubik, Pixelify Sans), palette, grid and components. Works offline: the fonts are stored in `deck/fonts/`.

| File | What it is |
|---|---|
| `index.html` | The pitch. 9 slides in the order the workshop recommends. |
| `layouts.html` | 19 spare layouts copied from the PDF. Copy a `<section>` into `index.html` when you need one. |
| `deck/deck.css` | Design tokens and components. |
| `deck/deck.js` | Scaling, navigation, overview, notes, timer. |
| `assets/` | Put screenshots, photos and videos here. |
| `export-pdf.sh` | Writes `pitch.pdf`, one 16:9 page per slide. |

## Present

Open `index.html` in Chrome. Press `F` for full screen.

| Key | Action |
|---|---|
| `→` `Space` `PgDn` / `←` `PgUp` | Next / previous (clickers work) |
| `Home` / `End` | First / last slide |
| `G` | Overview of all slides (click one to jump to it) |
| `N` | Speaker notes |
| `T` / `R` | Start or pause the rehearsal timer / reset it (target: `data-pitch-seconds` on `<body>`, set to 300 s: the final is 5 min talk + 2 min Q&A) |

`index.html#5` opens slide 5.

## Fill it in

- Everything in `[brackets]` is a placeholder. `+XX%` and `−XX%` are the big numbers.
- Team name: the footer is set once in the `<style>` block in `<head>` (`--footer-text`). The other mentions are in the title and closing slides.
- An accent-coloured word in a title: wrap it in `<em>`.
- Brief checklist (slide 2): while you build, mark open items `<li class="todo">`. They turn orange.
- Bars: `style="--v: 72"` is the bar length in percent.
- Screenshot: `<div class="shot"><img src="assets/demo.png" alt=""></div>`. For a backup video: `<video src="assets/demo.mp4" data-autoplay muted loop playsinline></video>` inside `.shot`. It plays when its slide is shown.
- Team photos: `<div class="avatar"><img src="assets/team-1.jpg" alt=""></div>`.
- Speaker notes: `<aside class="notes">` inside a slide.

Slide types: `title-slide`, `std` (eyebrow + title + body), `hero` (section opener), `statement`, `grad` (violet gradient), `closing`.
Components: `card` (`hi`, `ghost`, `bevel`, `warn`, `light`, `flat`, `tight`), `checklist`, `chips`, `rows`, `steps`, `flow`, `vflow`, `bars`, `metric`, `quotes`, `callout`, `tag` (`warn`, `outline`), `shot`, `grid` (`c3`, `c4`, `c5`), `ico` icons.

## Export to PDF

```sh
./export-pdf.sh            # index.html -> pitch.pdf
./export-pdf.sh layouts.html layouts.pdf
```

Or, in Chrome: Print → Save as PDF, Margins: None, Background graphics: on.
