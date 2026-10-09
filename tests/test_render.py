import numpy as np
import pytest

from firstlight import render


def layers(size=240, seed=0):
    """starless: a bright blob in the middle of noisy sky, different strength per filter; stars: a few spots"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    out = {}
    for i, n in enumerate("HOS"):
        sl = rng.normal(100, 2, (size, size)).astype(np.float32)
        sl += (40 - 10 * i) * np.exp(-((xx - 150) ** 2 + (yy - 70) ** 2) / (2 * 30 ** 2))   # nebula right of centre, upper (array rows 0..)
        st = np.zeros((size, size), np.float32)
        st[200:203, 40:43] = 300 + 20 * i                                                       # a bright star, lower left in the array
        out[n] = (sl, st)
    return out


def test_render_shape_dtype_and_choices():
    img, info = render.render(layers(), width=240)
    assert img.shape == (240, 240, 3) and img.dtype == np.uint8
    assert info["assignment"] == ["S", "H", "O"] and info["width"] == 240 and info["star_k"] > 0


def test_the_nebula_is_brighter_than_the_sky_and_the_star_is_white_and_bright():
    img, _ = render.render(layers(), width=240)
    flip = lambda y: 240 - 1 - y                    # the picture is flipped to display orientation
    nebula = img[flip(70), 150].astype(int)
    sky = img[flip(200), 200].astype(int)
    star = img[flip(201), 41].astype(int)
    assert nebula.sum() > sky.sum() + 60            # the blob stands out
    assert star.min() > 150                         # a bright star is bright in every band (close to white)


def test_orientation_matches_the_widget_rows_are_flipped():
    # the nebula sits at array row 70, so in the (bottom-up) layers it must appear near display row 240 - 1 - 70
    img, _ = render.render(layers(), width=240)
    nebula_row_display = np.argmax(img[:, 150].astype(int).sum(axis=1))
    assert abs(nebula_row_display - (240 - 1 - 70)) < 25


def test_deterministic():
    a = render.render(layers(), width=240)[0]
    b = render.render(layers(), width=240)[0]
    assert np.array_equal(a, b)


def test_output_width_follows_the_request():
    big = layers(size=480)
    img, info = render.render(big, width=240)
    assert img.shape[1] == 240 and info["width"] == 240


def test_two_filters_still_render_an_hs_bicolour():
    two = {n: v for n, v in layers().items() if n in "HS"}
    img, info = render.render(two, width=240)
    assert info["assignment"] == ["H", "S", "S"] and img.shape == (240, 240, 3)


def test_stretch_primitives_have_their_limits():
    assert render.mtf(0.2, np.array([0.0, 1.0, -1.0, 2.0])).tolist() == [0.0, 1.0, 0.0, 1.0]
    m = render.solve_m(0.03, 0.22)
    assert render.mtf(m, np.array([0.03]))[0] == pytest.approx(0.22, abs=1e-6)
    assert render.auto_scnr(0.3, 0.3, 0.3) == 0 and 0 < render.auto_scnr(0.15, 0.4, 0.15) <= 1
    assert render.screen(np.array([0.5]), np.array([0.5]))[0] == pytest.approx(0.75)


# ---- parity with the JavaScript: the port must not drift from the widget ----

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def js():
    from firstlight.widget import DSP
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Exception as exc:
            pytest.skip(f"Chrome not available: {exc}")
        pg = browser.new_page()
        pg.set_content("<html></html>")
        pg.add_script_tag(content=DSP.read_text())
        yield pg
        browser.close()


def test_python_stretch_matches_the_widgets_javascript(js):
    xs = np.linspace(-0.05, 1.05, 40).tolist()
    c = {"lo": -0.004, "span": 0.9, "m": render.solve_m(0.02, 0.22)}
    expect = js.evaluate("([xs, c]) => xs.map(p => stretchP(c, p, { contrast: 0.15 }))", [xs, c])
    got = render.stretch(c, np.array(xs, np.float32), 0.15)
    assert np.allclose(got, expect, atol=1e-5)


def test_python_star_stretch_scnr_and_screen_match_the_javascript(js):
    qs = [0.0, 0.003, 0.05, 0.4, 1.0]
    assert np.allclose(render.star_stretch(120.0, np.array(qs)), js.evaluate("qs => qs.map(q => starStretch(120, q))", qs), atol=1e-6)
    triples = [(0.2, 0.9, 0.4), (0.5, 0.1, 0.5), (0.1, 0.3, 0.1)]
    js_g = js.evaluate("ts => ts.map(([r, g, b]) => scnr(r, g, b, 0.7))", triples)
    py_g = [float(render.scnr(np.array(r), np.array(g), np.array(b), 0.7)) for r, g, b in triples]
    assert np.allclose(py_g, js_g, atol=1e-9)
    assert np.allclose(float(render.screen(np.array(0.3), np.array(0.6))), js.evaluate("() => screen(0.3, 0.6)"))
    for rgb in ((0.3, 0.3, 0.3), (0.15, 0.4, 0.15), (0.1, 0.9, 0.1)):
        assert render.auto_scnr(*rgb) == pytest.approx(js.evaluate("([r, g, b]) => autoScnr(r, g, b)", list(rgb)))


def test_python_chroma_smooth_matches_the_javascript_away_from_the_edges(js):
    rng = np.random.default_rng(4)
    r, g, b = (rng.uniform(0.2, 0.6, (40, 40)).astype(np.float32) for _ in range(3))
    expect = js.evaluate("""([r, g, b]) => { const R = Float32Array.from(r.flat()), G = Float32Array.from(g.flat()), B = Float32Array.from(b.flat());
        chromaSmooth(R, G, B, 40, 40, 2.0); return [Array.from(R), Array.from(G), Array.from(B)]; }""", [r.tolist(), g.tolist(), b.tolist()])
    got = render.chroma_smooth(r, g, b, 2.0)
    sl = (slice(8, 32), slice(8, 32))           # edge handling differs (clamp vs reflect); the interior must agree
    for gch, ech in zip(got, expect):
        assert np.allclose(gch[sl], np.array(ech).reshape(40, 40)[sl], atol=2e-3)
