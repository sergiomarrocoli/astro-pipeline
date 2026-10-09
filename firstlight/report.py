"""The one-page report for a target: what went in, how good it was, what the pipeline did, and the result.

It reads what the pipeline already wrote (manifest, frame_quality.json, settings.json, hot_pixels.lst,
background.json) plus the finished layers, renders the draft image, and writes <target>/report.html with every
image embedded, so the page is a single file you can email or archive.
"""
from __future__ import annotations

import base64
import html
import io
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from . import quality, render
from .stack import slug


def _load(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _jpeg_b64(img: np.ndarray, quality_: int = 90, max_width: Optional[int] = None) -> str:
    from PIL import Image
    im = Image.fromarray(img)
    if max_width and im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality_)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h, m = divmod(seconds // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


# ---- data gathering (pure, tested) ------------------------------------------------------

def summarise(target: str, manifest: dict, fq: Optional[dict]) -> dict:
    """Per filter: what exists, what was stacked, what was rejected, and the typical quality."""
    all_frames = manifest.get("targets", {}).get(target, {})
    stacked = defaultdict(list)
    for f in (fq or {}).get("frames", []):
        if f["target"] == target:
            stacked[f["filter"]].append(f)
    rows, notes = [], []
    for flt in sorted(all_frames):
        have = all_frames[flt]
        used = [f for f in stacked.get(flt, []) if not f["rejected"]]
        rejected = [f for f in stacked.get(flt, []) if f["rejected"]]
        if fq is None:                                         # rejection did not run: everything counts
            exps = sorted({f["exptime"] for f in have})
            used, rejected = [{"exposure": f["exptime"]} for f in have], []
        exposure = sorted({f.get("exposure", f.get("exptime")) for f in used})
        med = lambda k: (float(np.median([f[k] for f in used if f.get(k) is not None])) if any(f.get(k) is not None for f in used) else None)
        rows.append({"filter": flt, "available": len(have), "used": len(used), "rejected": len(rejected),
                     "exposure": exposure, "integration": float(sum(f.get("exposure") or f.get("exptime") or 0 for f in used)),
                     "fwhm": med("fwhm"), "roundness": med("roundness"), "noise": med("noise"),
                     "rejected_frames": rejected})
        ignored = [f for f in have if (f["exptime"] not in exposure)] if used else []
        if ignored:
            notes.append(f"{flt}: {len(ignored)} frame(s) at {sorted({int(f['exptime']) for f in ignored})} s were not used "
                         f"(the pipeline stacks the exposure with the most integration; --all-exposures includes them).")
        if len(used) == 1:
            notes.append(f"{flt}: only 1 frame, used as it is. This is not a real stack, so that channel is noisy.")
    return {"rows": rows, "notes": notes}


def zone_map(a: np.ndarray, top_down: bool = False, grid: int = 3) -> dict:
    """Median star FWHM and roundness in a grid x grid arrangement of the image, row 0 = top of the picture."""
    st = quality.measure_stars(np.asarray(a, dtype=np.float32), n=400)
    h, w = a.shape
    cells = {}
    for y, x, fw, ra, _ in (st if len(st) else []):
        if not (np.isfinite(fw) and np.isfinite(ra)):
            continue
        row = min(grid - 1, int(grid * y / h))
        if not top_down:
            row = grid - 1 - row                                   # Siril stacks are bottom-up: flip to display orientation
        cells.setdefault((row, min(grid - 1, int(grid * x / w))), []).append((fw, ra))
    fwhm = [[None] * grid for _ in range(grid)]
    rnd = [[None] * grid for _ in range(grid)]
    count = [[0] * grid for _ in range(grid)]
    for (r, c), v in cells.items():
        count[r][c] = len(v)
        if len(v) >= 5:
            fwhm[r][c] = float(np.median([p[0] for p in v]))
            rnd[r][c] = float(np.median([p[1] for p in v]))
    return {"fwhm": fwhm, "roundness": rnd, "count": count}


def svg_series(series: dict[str, list[tuple[float, float]]], width: int = 420, height: int = 130, ylabel: str = "") -> str:
    """A small inline SVG line chart. series: name -> [(x, y)], x in any number (e.g. minutes since the first frame)."""
    pts = [p for v in series.values() for p in v if p[1] is not None and np.isfinite(p[1])]
    if not pts:
        return ""
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    pad = (y1 - y0) * 0.15 or abs(y1) * 0.1 or 1
    y0, y1 = y0 - pad, y1 + pad
    left, right, top, bottom = 38, 8, 8, 18
    sx = lambda x: left + (x - x0) / ((x1 - x0) or 1) * (width - left - right)
    sy = lambda y: top + (1 - (y - y0) / ((y1 - y0) or 1)) * (height - top - bottom)
    colours = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)"]
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(ylabel)}" class="chart">']
    for frac in (0, 0.5, 1):
        yv = y0 + frac * (y1 - y0)
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}" class="grid"/>'
                   f'<text x="{left - 4}" y="{sy(yv) + 3:.1f}" class="axis" text-anchor="end">{yv:.4g}</text>')
    for i, (name, v) in enumerate(series.items()):
        v = [p for p in v if p[1] is not None and np.isfinite(p[1])]
        col = colours[i % len(colours)]
        if len(v) > 1:
            out.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.6" points="' +
                       " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in v) + '"/>')
        out += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.2" fill="{col}"><title>{html.escape(name)}: {y:.4g}</title></circle>' for x, y in v]
    out.append(f'<text x="{width - right}" y="{height - 4}" class="axis" text-anchor="end">minutes</text></svg>')
    return "".join(out)


def _legend(names: list[str]) -> str:
    return "".join(f'<span class="key"><i style="background:var(--c{i % 4 + 1})"></i>{html.escape(n)}</span>' for i, n in enumerate(names))


def _zone_table(zm: dict, key: str, fmt: str, good_low: bool) -> str:
    vals = [v for row in zm[key] for v in row if v is not None]
    if not vals:
        return "<p class='dim'>Not enough stars to map.</p>"
    lo, hi = min(vals), max(vals)
    rows = []
    for row in zm[key]:
        cells = []
        for v in row:
            if v is None:
                cells.append("<td class='dim'>–</td>")
                continue
            t = 0 if hi == lo else (v - lo) / (hi - lo)
            t = t if good_low else 1 - t                          # 0 = best, 1 = worst
            cells.append(f"<td style='background:rgba(214,96,77,{0.08 + 0.55 * t:.2f})'>{v:{fmt}}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table class='zone'>" + "".join(rows) + "</table>"


# ---- the page ---------------------------------------------------------------------------

CSS = """
:root { --bg:#fafaf8; --ink:#1c1d21; --dim:#6a6e78; --line:#dedfe3; --panel:#fff; --accent:#1f5fd1; --warn:#b4540a;
        --c1:#1f6fd0; --c2:#c4552d; --c3:#2a9d6a; --c4:#8a55c9; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#101114; --ink:#e7e8eb; --dim:#9a9ea8; --line:#2a2d35; --panel:#17191e;
        --accent:#7fb0ff; --warn:#f0a35e; --c1:#6aa9ff; --c2:#ff8a5c; --c3:#5fd38a; --c4:#c99aff; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 -apple-system, system-ui, "Segoe UI", sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 56px; }
h1 { margin:0 0 2px; font-size:26px; } h2 { margin:34px 0 10px; font-size:18px; border-top:1px solid var(--line); padding-top:18px; }
.sub { color:var(--dim); margin:0 0 18px; } .dim { color:var(--dim); }
.hero { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:10px 0 6px; }
.hero figure { margin:0; } .hero img { width:100%; height:auto; display:block; border-radius:8px; border:1px solid var(--line); background:#000; }
.hero figcaption { color:var(--dim); font-size:13px; margin-top:4px; }
@media (max-width:760px) { .hero { grid-template-columns:1fr; } }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); } th:first-child, td:first-child { text-align:left; }
th { color:var(--dim); font-weight:600; font-size:13px; }
.scroll { overflow-x:auto; }
.notes li { margin:3px 0; } .warn { color:var(--warn); }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:16px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
.card h3 { margin:0 0 6px; font-size:14px; }
.chart { width:100%; height:auto; } .chart .grid { stroke:var(--line); stroke-width:1; } .chart .axis { fill:var(--dim); font-size:10px; }
.key { display:inline-flex; align-items:center; gap:5px; margin-right:12px; font-size:13px; color:var(--dim); }
.key i { width:10px; height:10px; border-radius:2px; display:inline-block; }
.zone { width:auto; } .zone td { text-align:center; min-width:64px; padding:10px 8px; border:2px solid var(--panel); }
ul.plain { padding-left:18px; } code { background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:0 4px; font-size:13px; }
a { color:var(--accent); }
"""


def build_html(target: str, data: dict) -> str:
    esc = html.escape
    s = data["summary"]
    rows = "".join(
        f"<tr><td>{esc(r['filter'])}</td><td>{', '.join(str(int(e)) for e in r['exposure']) or '–'} s</td>"
        f"<td>{r['used']} of {r['available']}</td><td>{r['rejected'] or ''}</td><td>{_fmt_time(r['integration'])}</td>"
        f"<td>{('%.2f px' % r['fwhm']) if r['fwhm'] else '–'}</td><td>{('%.2f' % r['roundness']) if r['roundness'] else '–'}</td></tr>"
        for r in s["rows"])
    total = sum(r["integration"] for r in s["rows"])
    notes = "".join(f"<li class='warn'>{esc(n)}</li>" for n in s["notes"])
    rejected = "".join(f"<li>{esc(r['filter'])} at {esc(f['date_obs'][11:19])}: {esc('; '.join(f['reasons']))}</li>"
                       for r in s["rows"] for f in r["rejected_frames"])
    charts = ""
    if data.get("series"):
        for title, key in (("Star size (FWHM, px)", "fwhm"), ("Stars detected", "stars"), ("Sky noise", "noise")):
            ser = data["series"].get(key)
            if ser:
                charts += f"<div class='card'><h3>{title}</h3>{svg_series(ser, ylabel=title)}</div>"
    legend = _legend(list(data["series"]["fwhm"])) if data.get("series", {}).get("fwhm") else ""
    zones = ""
    for flt, zm in data.get("zones", {}).items():
        zones += (f"<div class='card'><h3>{esc(flt)}: star size (FWHM, px) &nbsp;·&nbsp; roundness</h3>"
                  f"<div style='display:flex; gap:14px; flex-wrap:wrap'>{_zone_table(zm, 'fwhm', '.2f', True)}{_zone_table(zm, 'roundness', '.2f', False)}</div></div>")
    proc = "".join(f"<li>{x}</li>" for x in data.get("processing", []))
    hero_before = (f"<figure><img alt='A single raw sub' src='data:image/jpeg;base64,{data['before']}'>"
                   f"<figcaption>One raw {esc(data.get('before_label', 'sub'))}, stretched to be visible.</figcaption></figure>") if data.get("before") else ""
    cap = esc(data["draft_caption"])
    head = " · ".join(x for x in (data.get("when"), data.get("setup")) if x)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(target)} report</title><style>{CSS}</style></head><body><main>
<h1>{esc(target)}</h1><p class="sub">{esc(head)}</p>
<div class="hero">{hero_before}<figure><img alt="The draft image" src="data:image/jpeg;base64,{data['draft']}"><figcaption>{cap}</figcaption></figure></div>
<p>Total integration {_fmt_time(total)}. Open <a href="preview/widget.html">the interactive widget</a> to change palette, stretch and colour, rotate and crop, and zoom to full resolution.</p>

<h2>The data</h2><div class="scroll"><table><tr><th>Filter</th><th>Exposure</th><th>Frames used</th><th>Rejected</th><th>Integration</th><th>Star FWHM</th><th>Roundness</th></tr>{rows}</table></div>
{('<ul class="notes">' + notes + '</ul>') if notes else ''}

<h2>Frame quality</h2>
<p class="dim">Each frame is measured and compared with its neighbours in time. Only clear outliers are dropped.</p>
{('<ul>' + rejected + '</ul>') if rejected else '<p>No frames were rejected.</p>'}
{('<p>' + legend + '</p>') if legend else ''}<div class="grid2">{charts}</div>

<h2>Across the field</h2>
<p class="dim">Median star size and roundness in each ninth of the picture (top row is the top of the image). A strong, one-directional gradient means the sensor is tilted relative to the optical axis; high at the edges all round means the spacing or field curvature.</p>
<div class="grid2">{zones}</div>

<h2>What the pipeline did</h2><ul class="plain">{proc}</ul>
</main></body></html>"""


def write(out: Path, target: str, names: list[str], cfg: Optional[dict], layers_dir: Path,
          raw_jpg: Optional[Path] = None) -> Optional[Path]:
    """Gather everything and write <out>/<target>/report.html (and draft.jpg beside it)."""
    from astropy.io import fits

    tdir = out / slug(target)
    manifest = _load(out / "manifest.json", {})
    fq = _load(out / "frame_quality.json")
    summary = summarise(target, manifest, fq)
    settings = (cfg or _load(tdir / "settings.json", {}).get("auto", {})) or {}

    # the draft image, from the same layers the widget uses
    e = 4
    layers = {n: (fits.getdata(layers_dir / f"starless_{slug(n)}.fit")[e:-e, e:-e],
                  fits.getdata(layers_dir / f"stars_{slug(n)}.fit")[e:-e, e:-e]) for n in names}
    img, info = render.render(layers, noise=settings.get("noise"), width=1600)
    from PIL import Image
    Image.fromarray(img).save(tdir / "draft.jpg", quality=92)

    # per-frame trends, in minutes since the first frame of the target
    series: dict = {"fwhm": {}, "stars": {}, "noise": {}}
    t0 = None
    frames = [f for f in (fq or {}).get("frames", []) if f["target"] == target]
    for f in sorted(frames, key=lambda f: f["date_obs"]):
        t = datetime.fromisoformat(f["date_obs"][:19])
        t0 = t0 or t
        minutes = (t - t0).total_seconds() / 60
        for key in series:
            if f.get(key) is not None:
                series[key].setdefault(f["filter"], []).append((minutes, f[key]))

    # star size across the field, from the finished stacks
    zones = {}
    for n in names:
        p = tdir / "linear" / f"linear_{slug(n)}.fit"
        if p.exists():
            top_down = str(fits.getheader(p).get("ROWORDER", "BOTTOM-UP")).upper() == "TOP-DOWN"
            zones[n] = zone_map(fits.getdata(p).astype(np.float32), top_down)

    # what happened
    proc = []
    first_hdr = None
    first = next((f for filt in manifest.get("targets", {}).get(target, {}).values() for f in filt), None)
    if first:
        try:
            first_hdr = fits.getheader(first["path"])
        except OSError:
            first_hdr = None
    hot = out / "hot_pixels.lst"
    if hot.exists():
        proc.append(f"<b>Hot pixels:</b> {sum(1 for _ in hot.open()):,} stable hot pixels found from the lights themselves (no darks) and corrected in every sub before registration.")
    bgj = _load(tdir / "background.json", {})
    if bgj:
        bits = [f"{n}: {v['amplitude_over_noise']:.1f}σ" for n, v in sorted(bgj.items()) if v.get("applied")]
        if bits:
            proc.append("<b>Sky gradient:</b> removed per filter (largest variation across the frame, in noise sigma: " + ", ".join(bits) + ").")
    if settings.get("fwhm"):
        proc.append(f"<b>Stars:</b> separated from the nebula; measured star FWHM {settings['fwhm']} px on {settings.get('fwhm_measured_on', 'H')}.")
        proc.append(f"<b>Starless layer:</b> denoised ({round(100 * settings.get('denoise_amount', 1))}% blend). <b>Stars layer:</b> deconvolved "
                    f"({settings.get('rl_iterations')} iterations), noise-gated, negative lobes clipped.")
    proc.append(f"<b>Draft image:</b> palette {''.join(r_[0] if r_ else '?' for r_ in info['assignment'])} (R, G, B), "
                f"green removal {info['scnr']:.2f} (auto), stars boosted with an asinh stretch. Everything here is adjustable in the widget.")
    proc.append("<b>Files:</b> the linear, aligned stacks are in <code>linear/</code>; the starless and stars layers in <code>final+rl+dn/</code>; "
                "scripts and logs for every Siril step sit beside their outputs.")

    before = before_label = None
    if raw_jpg and Path(raw_jpg).exists():
        from PIL import Image as _I
        before = _jpeg_b64(np.asarray(_I.open(raw_jpg).convert("RGB")), 88, 1000)
        before_label = f"{Path(raw_jpg).stem.split('_')[-1]} sub"
    when = None
    if first_hdr is not None and first_hdr.get("DATE-OBS"):
        when = str(first_hdr["DATE-OBS"])[:10]
    setup = None
    if first_hdr is not None:
        px, fl = first_hdr.get("XPIXSZ"), first_hdr.get("FOCALLEN")
        bits = [first_hdr.get("INSTRUME"), first_hdr.get("TELESCOP") and f"{first_hdr['TELESCOP']}" + (f" {fl:g} mm" if fl else "")]
        if px and fl:
            bits.append(f"{206.265 * px / fl:.1f}\u2033/px")
        setup = " · ".join(str(b) for b in bits if b)
    data = {"summary": summary, "draft": _jpeg_b64(img, 90), "series": series, "zones": zones, "processing": proc, "before": before,
            "before_label": before_label, "when": when, "setup": setup,
            "draft_caption": "The draft the pipeline produces on its own: its settings, no tweaking."}
    dest = tdir / "report.html"
    dest.write_text(build_html(target, data))
    return dest
