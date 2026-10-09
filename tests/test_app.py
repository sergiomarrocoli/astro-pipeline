import time
from pathlib import Path

from firstlight import app, enhance


def test_argv_from_options():
    argv = app.build_argv("/n/night", "/o", {"target": "NGC", "filter": "H", "denoise": True, "dry-run": True, "no-darks": True, "deconvolve": False})
    assert argv[:3] == ["/n/night", "-o", "/o"] and "--target" in argv and "--denoise" in argv and "--no-darks" in argv and "--dry-run" in argv
    assert "--deconvolve" not in argv
    assert app.build_argv("/n", None, {}) == ["/n"]


def test_every_enhancement_flag_is_offered_and_understood():
    flags = [s.flag.lstrip("-") for s in enhance.ORDER]
    argv = app.build_argv("/n", None, {f: True for f in flags})
    assert all(f"--{f}" in argv for f in flags)


def test_default_output_sits_beside_the_input():
    assert app.default_output("/data/2026-10-03") == "/data/2026-10-03-firstlight"


def test_summary_counts_frames_and_integration():
    m = {"targets": {"NGC 1": {"H": [{"exptime": 300.0}] * 3, "O": [{"exptime": 300.0}]}}, "calibration": {"dark": [1, 2], "flat": [], "bias": [], "darkflat": []}, "ignored": ["x"]}
    s = app.summarise(m)
    assert s["targets"][0]["filters"] == {"H": 3, "O": 1} and s["targets"][0]["seconds"] == 1200 and s["calibration"]["dark"] == 2 and s["ignored"] == 1


def test_results_lists_what_exists(tmp_path):
    t = tmp_path / "ngc_1"
    (t / "preview").mkdir(parents=True)
    (t / "preview" / "widget.html").write_text("x"); (t / "report.html").write_text("x")
    (tmp_path / "other").mkdir()
    r = app.results(tmp_path)
    assert len(r) == 1 and r[0]["target"] == "ngc_1" and set(r[0]) == {"target", "widget", "report"}
    assert app.results(tmp_path / "missing") == []


def test_scan_of_a_missing_folder_and_an_empty_one(tmp_path):
    b = app.Backend()
    assert "not a folder" in b.scan(str(tmp_path / "no"))["error"]
    assert "no light frames" in b.scan(str(tmp_path))["error"]


def wait(b, timeout=10):
    t = time.time()
    while b.state == "running" and time.time() - t < timeout:
        time.sleep(0.05)


def test_a_run_captures_output_and_reports_success(tmp_path, monkeypatch):
    from firstlight import cli
    monkeypatch.setattr(cli, "main", lambda argv: print("hello", argv[0]) or 0)
    b = app.Backend()
    assert b.start(str(tmp_path), str(tmp_path / "o"), {})["ok"]
    wait(b)
    p = b.poll()
    assert p["state"] == "done" and any("hello" in l for l in p["lines"]) and b.poll()["lines"] == []   # lines are delivered once


def test_a_failing_run_is_reported_and_a_second_start_is_refused_while_running(tmp_path, monkeypatch):
    from firstlight import cli
    def slow(argv):
        time.sleep(0.3); raise RuntimeError("boom")
    monkeypatch.setattr(cli, "main", slow)
    b = app.Backend()
    b.start(str(tmp_path), None, {})
    assert "already" in b.start(str(tmp_path), None, {})["error"]
    wait(b)
    p = b.poll()
    assert p["state"] == "failed" and any("boom" in l for l in p["lines"])


def test_scan_of_real_looking_headers(tmp_path):
    from astropy.io import fits
    import numpy as np
    for i in range(2):
        h = fits.Header(); h.update({"IMAGETYP": "Light", "OBJECT": "M42", "FILTER": "H", "EXPTIME": 60.0})
        fits.PrimaryHDU(np.zeros((4, 4), np.uint16), header=h).writeto(tmp_path / f"l{i}.fits")
    s = app.Backend().scan(str(tmp_path))
    assert s["targets"][0]["name"] == "M42" and s["targets"][0]["filters"] == {"H": 2} and s["output"].endswith("-firstlight")


def test_the_page_ships_with_the_package():
    assert Path(app.__file__).with_name("app_ui.html").exists()


def test_the_page_runs_against_the_backend_without_script_errors(tmp_path):
    """Load the real page in a browser with the real Backend behind a stand-in for pywebview's bridge."""
    import pytest
    pw = pytest.importorskip("playwright.sync_api")
    from astropy.io import fits
    import numpy as np
    for i in range(2):
        h = fits.Header(); h.update({"IMAGETYP": "Light", "OBJECT": "M42", "FILTER": "H", "EXPTIME": 60.0})
        fits.PrimaryHDU(np.zeros((4, 4), np.uint16), header=h).writeto(tmp_path / f"l{i}.fits")
    b = app.Backend()
    with pw.sync_playwright() as p:
        try:
            br = p.chromium.launch(channel="chrome")
        except Exception:
            pytest.skip("no browser")
        page = br.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        def call(name, *args):
            return getattr(b, name)(*args)
        page.expose_function("py_call", call)
        page.add_init_script("""window.pywebview = {api: new Proxy({}, {get: (_, n) => (...a) => window.py_call(n, ...a)})};""")
        page.goto(Path(app.__file__).with_name("app_ui.html").as_uri())
        page.evaluate("window.dispatchEvent(new Event('pywebviewready'))")
        page.wait_for_selector("#deconvolve", state="attached")
        page.fill("#folder", str(tmp_path))
        page.click("#scan")
        page.wait_for_selector("#opts:not(.hide)")
        assert "M42" in page.inner_text("#found") and page.input_value("#out").endswith("-firstlight")
        br.close()
    assert errors == []
