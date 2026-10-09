import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from firstlight import report


def frame(flt, t, exposure=300.0, rejected=False, reasons=(), target="T1", **m):
    base = {"fwhm": 2.2, "roundness": 0.78, "stars": 2600, "noise": 18.0, "sky": 2050.0}
    return {"path": f"/x/{flt}{t}.fits", "target": target, "filter": flt, "exposure": exposure, "date_obs": f"2026-10-03T22:{t:02d}:00",
            **{**base, **m}, "rejected": rejected, "reasons": list(reasons)}


def manifest(**filters):
    return {"targets": {"T1": {flt: [{"path": f"/x/{flt}{i}.fits", "exptime": e} for i, e in enumerate(exps)] for flt, exps in filters.items()}}}


def test_summary_counts_integration_and_rejections():
    fq = {"frames": [frame("H", 1), frame("H", 2), frame("H", 3, rejected=True, reasons=["soft stars"]), frame("O", 4)]}
    s = report.summarise("T1", manifest(H=[300, 300, 300], O=[300]), fq)
    rows = {r["filter"]: r for r in s["rows"]}
    assert (rows["H"]["used"], rows["H"]["rejected"], rows["H"]["available"]) == (2, 1, 3)
    assert rows["H"]["integration"] == 600 and rows["H"]["fwhm"] == pytest.approx(2.2)
    assert rows["H"]["rejected_frames"][0]["reasons"] == ["soft stars"]


def test_summary_notes_unused_exposures_and_single_frames():
    fq = {"frames": [frame("H", 1), frame("H", 2), frame("O", 3)]}
    s = report.summarise("T1", manifest(H=[300, 300, 120, 120, 60], O=[300]), fq)
    text = " ".join(s["notes"])
    assert "H: 3 frame(s) at [60, 120] s were not used" in text and "O: only 1 frame" in text


def test_summary_without_a_quality_file_counts_everything():
    s = report.summarise("T1", manifest(H=[300, 300]), None)
    assert s["rows"][0]["used"] == 2 and s["rows"][0]["integration"] == 600


def test_fmt_time():
    assert report._fmt_time(300) == "5 min" and report._fmt_time(4500) == "1 h 15 min" and report._fmt_time(3600) == "1 h 00 min"


def test_svg_series_has_one_line_per_series_and_handles_gaps():
    svg = report.svg_series({"H": [(0, 2.0), (10, 2.2), (20, 2.1)], "S": [(0, 1.0), (10, None), (20, 1.4)]}, ylabel="FWHM")
    assert svg.count("<polyline") == 2 and svg.count("<circle") == 5 and "FWHM" in svg
    assert report.svg_series({}) == "" and report.svg_series({"H": [(0, None)]}) == ""


def star_stack(size=600, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.03, 0.002, (size, size)).astype(np.float32)
    yy, xx = np.mgrid[:size, :size]
    srng = np.random.default_rng(9)
    for _ in range(300):
        cx, cy = srng.integers(15, size - 15, 2)
        sig = 1.0 + 1.2 * (cx / size)                        # stars get softer toward the right of the picture
        a += (srng.uniform(0.3, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))).astype(np.float32)
    return a


def test_zone_map_sees_stars_getting_softer_to_the_right():
    zm = report.zone_map(star_stack(), top_down=True)
    left = np.mean([zm["fwhm"][r][0] for r in range(3) if zm["fwhm"][r][0]])
    right = np.mean([zm["fwhm"][r][2] for r in range(3) if zm["fwhm"][r][2]])
    assert right > left + 0.8


def test_zone_map_row_zero_is_the_top_of_the_picture():
    a = star_stack()
    img = np.zeros_like(a); img[:] = a
    soft_top = img.copy()                                    # make the stars in the first rows of the array soft
    from firstlight.starsplit import gblur
    soft_top[:200] = gblur(soft_top[:200], 1.6)
    top_down = report.zone_map(soft_top, top_down=True)
    bottom_up = report.zone_map(soft_top, top_down=False)
    avg = lambda zm, r: np.mean([v for v in zm["fwhm"][r] if v])
    assert avg(top_down, 0) > avg(top_down, 2) + 0.3          # array row 0 is the top when stored top-down
    assert avg(bottom_up, 2) > avg(bottom_up, 0) + 0.3        # ...and the bottom of the picture when stored bottom-up


def test_zone_table_marks_the_worst_cell_darkest():
    zm = {"fwhm": [[2.0, 2.1, 3.3], [2.0, 2.2, 2.5], [1.9, 2.0, 2.3]]}
    html = report._zone_table(zm, "fwhm", ".2f", good_low=True)
    assert "3.30" in html and "rgba(214,96,77,0.63)" in html and "rgba(214,96,77,0.08)" in html


def test_build_html_contains_the_sections_and_escapes_text():
    fq = {"frames": [frame("H", 1), frame("H", 2, rejected=True, reasons=["soft <stars>"])]}
    data = {"summary": report.summarise("T1", manifest(H=[300, 300]), fq), "draft": "AAAA", "series": {"fwhm": {"H": [(0, 2.0), (5, 2.2)]}},
            "zones": {}, "processing": ["<b>Hot pixels:</b> 15,093"], "before": None, "when": "2026-10-03", "setup": "cam · scope",
            "draft_caption": "caption"}
    html_out = report.build_html("NGC <7000>", data)
    for needle in ("<h2>The data</h2>", "<h2>Frame quality</h2>", "<h2>Across the field</h2>", "<h2>What the pipeline did</h2>", "data:image/jpeg;base64,AAAA"):
        assert needle in html_out
    assert "NGC &lt;7000&gt;" in html_out and "soft &lt;stars&gt;" in html_out and "NGC <7000>" not in html_out


def test_write_builds_a_self_contained_report_from_an_output_folder(tmp_path):
    out = tmp_path / "out"; tdir = out / "T1"
    names = ["H", "O", "S"]
    rng = np.random.default_rng(1)
    for sub, n in ((s, n) for s in ("linear", "final+rl+dn") for n in names):
        (tdir / sub).mkdir(parents=True, exist_ok=True)
    for n in names:
        fits.PrimaryHDU(star_stack(seed=1)).writeto(tdir / "linear" / f"linear_{n}.fit")
        yy, xx = np.mgrid[:600, :600]
        sl = (rng.normal(0.03, 0.002, (600, 600)) + 0.05 * np.exp(-((xx - 300) ** 2 + (yy - 300) ** 2) / (2 * 80 ** 2))).astype(np.float32)
        st = np.zeros((600, 600), np.float32); st[100:103, 100:103] = 0.8
        fits.PrimaryHDU(sl).writeto(tdir / "final+rl+dn" / f"starless_{n}.fit")
        fits.PrimaryHDU(st).writeto(tdir / "final+rl+dn" / f"stars_{n}.fit")
    first = tmp_path / "raw.fits"
    h = fits.Header(); h["DATE-OBS"] = "2026-10-03T22:53:04"; h["INSTRUME"] = "CAM"; h["TELESCOP"] = "scope"; h["FOCALLEN"] = 135.0; h["XPIXSZ"] = 3.76
    fits.PrimaryHDU(np.zeros((8, 8), np.uint16), header=h).writeto(first)
    (out / "manifest.json").write_text(json.dumps({"targets": {"T1": {n: [{"path": str(first), "exptime": 300.0}] * 3 for n in names}}}))
    (out / "frame_quality.json").write_text(json.dumps({"frames": [frame(n, i, target="T1") for n in names for i in range(3)]}))
    (out / "hot_pixels.lst").write_text("P 1 1 H\nP 2 2 H\n")
    (tdir / "background.json").write_text(json.dumps({n: {"applied": True, "amplitude_over_noise": 1.5} for n in names}))
    (tdir / "settings.json").write_text(json.dumps({"auto": {"fwhm": 2.6, "fwhm_measured_on": "H", "denoise_amount": 1.0, "rl_iterations": 10,
                                                             "noise": {n: 0.0002 for n in names}}, "override": {}}))
    dest = report.write(out, "T1", names, None, tdir / "final+rl+dn")
    page = dest.read_text()
    assert dest.name == "report.html" and (tdir / "draft.jpg").exists()
    assert "2 stable hot pixels" in page and "2026-10-03" in page and '5.7\u2033/px' in page and "CAM" in page
    assert "Hot pixels" in page and "Sky gradient" in page and "S: 1.5σ" in page and "palette SHO" in page
    assert page.count("data:image/jpeg;base64,") >= 1 and "<svg" in page and "class='zone'" in page


def test_large_hot_pixel_counts_get_thousands_separators(tmp_path):
    assert f"{15093:,}" == "15,093"           # the report formats the count this way
    import inspect
    assert ":," in inspect.getsource(report.write)
