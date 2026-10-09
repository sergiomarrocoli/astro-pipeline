"""The widget's colour and stretch primitives, run in a real browser.

Skipped when Playwright or Chrome is not available (pip install playwright; uses the system Chrome).
"""
import pytest

from firstlight.widget import DSP

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def page():
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Exception as exc:  # no Chrome installed
            pytest.skip(f"Chrome not available: {exc}")
        pg = browser.new_page()
        pg.set_content("<html></html>")
        pg.add_script_tag(content=DSP.read_text())
        yield pg
        browser.close()


def test_mtf_solves_for_the_target(page):
    r = page.evaluate("() => { const x = 0.03, t = 0.2, m = solveM(x, t); return { mapped: mtf(m, x), lo: mtf(m, 0), hi: mtf(m, 1) }; }")
    assert abs(r["mapped"] - 0.2) < 1e-6 and r["lo"] == 0 and r["hi"] == 1


def test_stretch_is_monotonic_and_clipped(page):
    r = page.evaluate("""() => {
      const c = { lo: -0.01, span: 1.0, m: solveM(0.01, 0.22) }, S = { contrast: 0.3 };
      let mono = true, prev = -1;
      for (let p = -0.05; p <= 1.05; p += 0.01) { const v = stretchP(c, p, S); if (v < prev - 1e-9) mono = false; prev = v; }
      return { mono, lo: stretchP(c, -1, S), hi: stretchP(c, 5, S) };
    }""")
    assert r["mono"] and r["lo"] == 0 and abs(r["hi"] - 1) < 1e-9


def test_star_stretch_lifts_faint_stars_and_keeps_bright_ones_at_one(page):
    r = page.evaluate("() => ({ faint: starStretch(300, 0.01), bright: starStretch(300, 1), zero: starStretch(300, 0), gentle: starStretch(5, 0.01) })")
    assert r["zero"] == 0 and abs(r["bright"] - 1) < 1e-9
    assert r["faint"] > 10 * r["gentle"] and r["faint"] > 0.25   # a strong k makes faint stars clearly visible


def test_scnr_caps_green_at_the_mean_of_red_and_blue(page):
    r = page.evaluate("() => ({ full: scnr(0.2, 0.9, 0.4, 1), half: scnr(0.2, 0.9, 0.4, 0.5), none: scnr(0.2, 0.9, 0.4, 0), low: scnr(0.5, 0.1, 0.5, 1) })")
    assert abs(r["full"] - 0.3) < 1e-9          # (0.2 + 0.4) / 2
    assert abs(r["half"] - 0.6) < 1e-9          # halfway between 0.9 and 0.3
    assert r["none"] == 0.9 and r["low"] == 0.1  # no change when green is already below the cap


def test_screen_blend(page):
    r = page.evaluate("() => ({ a: screen(0, 0.4), b: screen(1, 0.4), c: screen(0.5, 0.5) })")
    assert r["a"] == 0.4 and r["b"] == 1 and abs(r["c"] - 0.75) < 1e-9


def test_auto_scnr_only_when_green_dominates(page):
    r = page.evaluate("() => ({ neutral: autoScnr(0.3, 0.3, 0.3), slight: autoScnr(0.3, 0.31, 0.3), green: autoScnr(0.15, 0.4, 0.15), extreme: autoScnr(0.1, 0.9, 0.1) })")
    assert r["neutral"] == 0 and r["slight"] == 0
    assert 0 < r["green"] <= 1 and r["extreme"] == 1


def test_gblur_keeps_constants_and_mass(page):
    r = page.evaluate("""() => {
      const w = 40, h = 40, c = gblur(new Float32Array(w * h).fill(3), w, h, 2);
      const imp = new Float32Array(w * h); imp[20 * w + 20] = 1;
      const b = gblur(imp, w, h, 2); let sum = 0, mx = 0; for (const v of b) { sum += v; mx = Math.max(mx, v); }
      return { cmin: Math.min(...c), cmax: Math.max(...c), mass: sum, centrePeak: b[20 * w + 20] === mx };
    }""")
    assert abs(r["cmin"] - 3) < 1e-4 and abs(r["cmax"] - 3) < 1e-4 and abs(r["mass"] - 1) < 1e-3 and r["centrePeak"]


def test_chroma_smooth_removes_colour_noise_but_keeps_luminance(page):
    r = page.evaluate("""() => {
      const w = 50, h = 50, n = w * h; let s = 7; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
      const R = new Float32Array(n), G = new Float32Array(n), B = new Float32Array(n);
      for (let i = 0; i < n; i++) { R[i] = 0.4 + 0.2 * rnd(); G[i] = 0.4 + 0.2 * rnd(); B[i] = 0.4 + 0.2 * rnd(); }
      const Y = i => 0.299 * R[i] + 0.587 * G[i] + 0.114 * B[i], y0 = Array.from(R, (_, i) => Y(i));
      const sd = a => { let m = 0, q = 0; for (const v of a) { m += v; q += v * v; } m /= a.length; return Math.sqrt(q / a.length - m * m); };
      const before = sd(Array.from(R, (v, i) => v - Y(i)));
      chromaSmooth(R, G, B, w, h, 3);
      let dY = 0; for (let i = 0; i < n; i++) dY = Math.max(dY, Math.abs(Y(i) - y0[i]));
      return { dY, before, after: sd(Array.from(R, (v, i) => v - Y(i))) };
    }""")
    assert r["dY"] < 1e-5 and r["after"] < 0.4 * r["before"]


def test_with_chroma_pairs_luminance_with_given_colour(page):
    r = page.evaluate("() => { const o = [0, 0, 0]; withChroma(0.5, 0.1, -0.1, o); return { o, y: 0.299 * o[0] + 0.587 * o[1] + 0.114 * o[2], cb: o[2] - 0.5, cr: o[0] - 0.5 }; }")
    assert abs(r["y"] - 0.5) < 1e-6 and abs(r["cb"] - 0.1) < 1e-6 and abs(r["cr"] + 0.1) < 1e-6


# ---- frame geometry ----

def test_bbox_of_a_rotated_rectangle(page):
    r = page.evaluate("() => ({ none: bboxOf(400, 300, 0), q: bboxOf(400, 300, 90), r: bboxOf(400, 300, 10), neg: bboxOf(400, 300, -10) })")
    assert (r["none"]["Wb"], r["none"]["Hb"]) == (400, 300)
    assert abs(r["q"]["Wb"] - 300) < 1e-9 and abs(r["q"]["Hb"] - 400) < 1e-9
    assert abs(r["r"]["Wb"] - 446.02) < 0.05 and abs(r["r"]["Hb"] - 364.9) < 0.05   # Siril rounds this up to 446 x 365
    assert r["r"] == r["neg"]


def test_fit_inside_stays_inside_the_rotated_image_with_the_requested_aspect(page):
    r = page.evaluate("""() => {
      const out = [];
      for (const rot of [0, 3, 10, 30, 45, 90, -17]) for (const aspect of [4 / 3, 1, 16 / 9]) {
        const crop = fitInside(400, 300, rot, aspect), g = frameGeom(400, 300, rot, crop);
        const t = rot * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
        let inside = true;
        for (const [px, py] of [[g.x0, g.y0], [g.x0 + g.Rw, g.y0], [g.x0, g.y0 + g.Rh], [g.x0 + g.Rw, g.y0 + g.Rh]]) {
          const rx = px - g.Wb / 2, ry = py - g.Hb / 2, u = rx * c + ry * s, v = -rx * s + ry * c;
          if (Math.abs(u) > 200 + 1e-6 || Math.abs(v) > 150 + 1e-6) inside = false;
        }
        out.push({ rot, aspect, inside, ratio: g.Rw / g.Rh });
      }
      return out;
    }""")
    for o in r:
        assert o["inside"], o
        assert abs(o["ratio"] - o["aspect"]) < 1e-6, o


def test_frame_matrix_matches_siril_for_a_clockwise_rotation(page):
    r = page.evaluate("""() => {
      const w = 400, h = 300, rot = 10, g = frameGeom(w, h, rot, [0, 0, 1, 1]);
      const m = frameMatrix(w, h, rot, g, 1), ap = (x, y) => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];
      const m2 = frameMatrix(w, h, rot, g, 2), ap2 = (x, y) => [m2[0] * x + m2[2] * y + m2[4], m2[1] * x + m2[3] * y + m2[5]];
      return { marker: ap(40, 30), centre: ap(200, 150), centre2: ap2(200, 150), Wb: g.Wb, Hb: g.Hb };
    }""")
    # Siril rotated the same synthetic image by +10 degrees: a marker at (40, 30) landed at (86, 36) in a 446x365 frame
    assert abs(r["marker"][0] - 86.2) < 0.6 and abs(r["marker"][1] - 36.5) < 0.6
    assert abs(r["centre"][0] - r["Wb"] / 2) < 1e-6 and abs(r["centre"][1] - r["Hb"] / 2) < 1e-6   # centre stays at the box centre
    assert abs(r["centre2"][0] - r["Wb"]) < 1e-6 and abs(r["centre2"][1] - r["Hb"]) < 1e-6        # and scales with k


def test_view_to_image_inverts_the_frame_matrix(page):
    r = page.evaluate("""() => {
      let worst = 0;
      for (const rot of [0, 7, -23, 90, 133]) {
        const w = 400, h = 300, crop = [0.1, 0.05, 0.9, 0.8], g = frameGeom(w, h, rot, crop), k = 1.7;
        const m = frameMatrix(w, h, rot, g, k);
        for (const [u, v] of [[0, 0], [123, 45], [399, 299], [200, 150]]) {
          const vx = m[0] * u + m[2] * v + m[4], vy = m[1] * u + m[3] * v + m[5];
          const [u2, v2] = viewToImage(w, h, rot, g, k, vx, vy);
          worst = Math.max(worst, Math.abs(u2 - u), Math.abs(v2 - v));
        }
      }
      return worst;
    }""")
    assert r < 1e-9
