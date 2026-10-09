"""The widget page itself (frame tool), driven in a real browser with a small synthetic image.

Skipped when Playwright or Chrome is not available.
"""
import numpy as np
import pytest

from firstlight.widget import build_payload, render_html

sync_api = pytest.importorskip("playwright.sync_api")


def synthetic_widget(w=160, h=120):
    rng = np.random.default_rng(0)
    layers = {n: (rng.normal(50, 2, (h, w)) + np.linspace(0, 20, w), np.zeros((h, w))) for n in "HOS"}
    return render_html(build_payload(layers))


@pytest.fixture(scope="module")
def page():
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Exception as exc:
            pytest.skip(f"Chrome not available: {exc}")
        ctx = browser.new_context(viewport={"width": 1400, "height": 900}, accept_downloads=True)
        pg = ctx.new_page()
        pg.errors = []
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        pg.set_content(synthetic_widget())
        pg.wait_for_selector("#c")
        yield pg
        browser.close()


def state(pg):
    return pg.evaluate("() => ({ rot: F.rot, crop: F.crop, editing, ar: parseFloat(getComputedStyle(view).getPropertyValue('--ar')), box: cropBox })")


def test_page_loads_without_errors_and_starts_unframed(page):
    s = state(page)
    assert page.errors == [] and s["rot"] == 0 and s["crop"] == [0, 0, 1, 1] and abs(s["ar"] - 160 / 120) < 1e-9


def test_rotating_crops_to_a_clean_frame_with_the_images_own_shape(page):
    page.click("#frameReset")
    page.evaluate("() => { const e = document.getElementById('rot'); e.value = 8; e.dispatchEvent(new Event('input')); }")
    s = state(page)
    assert s["rot"] == 8
    expected = page.evaluate("() => fitInside(W, H, 8, W / H)")
    assert all(abs(a - b) < 1e-9 for a, b in zip(s["crop"], expected))
    assert abs(s["ar"] - 160 / 120) < 1e-6                       # the view takes the crop's shape


def test_quarter_turns_accumulate_and_wrap(page):
    page.click("#frameReset")
    for _ in range(4):
        page.click("#rotr")
    assert state(page)["rot"] == 0
    page.click("#rotr")
    s = state(page)
    assert s["rot"] == 90 and abs(s["ar"] - 120 / 160) < 1e-6      # a portrait view after a quarter turn
    page.click("#rotl"); page.click("#rotl")
    assert state(page)["rot"] == -90


def test_dragging_a_corner_keeps_the_shape_and_done_applies_it(page):
    page.click("#frameReset")
    page.click("#cropEdit")
    assert state(page)["editing"]
    h = page.locator("#crop i.nw").bounding_box()
    page.mouse.move(h["x"] + 7, h["y"] + 7); page.mouse.down(); page.mouse.move(h["x"] + 80, h["y"] + 60, steps=6); page.mouse.up()
    box = state(page)["box"]
    ratio = page.evaluate("() => { const g = frameGeom(W, H, F.rot, cropBox); return g.Rw / g.Rh; }")
    assert abs(ratio - 160 / 120) < 1e-6 and box[0] > 0 and box[2] == 1     # shrank from the NW corner, shape kept
    page.click("#cropEdit")                                                  # Done
    s = state(page)
    assert not s["editing"] and s["crop"] == box and abs(s["ar"] - 160 / 120) < 1e-6


def test_free_crop_changes_the_shape(page):
    page.click("#frameReset")
    page.select_option("#aspect", "free")
    page.click("#cropEdit")
    h = page.locator("#crop i.e").bounding_box()
    page.mouse.move(h["x"] + 7, h["y"] + 7); page.mouse.down(); page.mouse.move(h["x"] - 120, h["y"] + 7, steps=6); page.mouse.up()
    page.click("#cropEdit")
    assert state(page)["ar"] < 160 / 120 - 0.1                     # narrower than the original
    page.select_option("#aspect", "orig")


def test_escape_cancels_an_edit(page):
    page.click("#frameReset")
    page.click("#cropEdit")
    h = page.locator("#crop i.nw").bounding_box()
    page.mouse.move(h["x"] + 7, h["y"] + 7); page.mouse.down(); page.mouse.move(h["x"] + 90, h["y"] + 70, steps=4); page.mouse.up()
    page.keyboard.press("Escape")
    s = state(page)
    assert not s["editing"] and s["crop"] == [0, 0, 1, 1]


def test_export_is_the_framed_size(page):
    page.click("#frameReset")
    page.evaluate("() => { const e = document.getElementById('rot'); e.value = 15; e.dispatchEvent(new Event('input')); }")
    expected = page.evaluate("() => { const g = frameGeom(W, H, F.rot, F.crop); return [Math.round(g.Rw), Math.round(g.Rh)]; }")
    from PIL import Image
    with page.expect_download() as dl:
        page.click("#save")
    assert list(Image.open(dl.value.path()).size) == expected
    assert page.errors == []
