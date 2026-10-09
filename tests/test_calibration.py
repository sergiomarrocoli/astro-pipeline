import json
from pathlib import Path

import numpy as np
import pytest

from firstlight import calibration as cal
from firstlight.calibration import Conditions, Library


@pytest.fixture(autouse=True)
def fake_stamp(monkeypatch):
    import hashlib
    monkeypatch.setattr(cal, "stamp", lambda frames, script: hashlib.sha256((script + "".join(sorted(f["path"] for f in frames))).encode()).hexdigest())


def dark(i, exp=300.0, gain=120, offset=50, temp=-10.0, binning=1, inst="CAM"):
    return {"path": f"/d/{exp}_{gain}_{i}.fits", "exptime": exp, "gain": gain, "offset": offset, "temp": temp + 0.05 * (i % 3),
            "binning": binning, "instrument": inst, "kind": "dark"}


def entry(file="d.fit", exp=300.0, gain=120, offset=50, temp=-10.0, n=20, built="2026-10-01T00:00:00", binning=1, inst="CAM"):
    return {"kind": "dark", "file": file, "instrument": inst, "binning": binning, "gain": gain, "offset": offset, "exposure": exp,
            "temp": temp, "n_frames": n, "built": built, "source_stamp": file}


def lib(tmp_path, *entries):
    L = Library(tmp_path / "lib")
    for e in entries:
        (tmp_path / "lib").mkdir(parents=True, exist_ok=True)
        (tmp_path / "lib" / e["file"]).write_text("x")
        L.data["masters"].append(e)
    return L


def light(**kw):
    return {"instrument": "CAM", "binning": 1, "gain": 120, "offset": 50, "exposure": 300.0, "temp": -10.0, **kw}


# ---- matching ----------------------------------------------------------------------------

def test_exact_match_is_found(tmp_path):
    m, why = lib(tmp_path, entry()).find_dark(light())
    assert m["file"] == "d.fit" and "matches" in why


@pytest.mark.parametrize("change", [{"gain": 100}, {"offset": 10}, {"exposure": 120.0}, {"binning": 2}, {"instrument": "OTHER"}])
def test_any_mismatch_in_the_conditions_means_no_dark(tmp_path, change):
    m, why = lib(tmp_path, entry()).find_dark(light(**change))
    assert m is None and "no dark for" in why and "the library has" in why


def test_exposure_matches_within_half_a_second(tmp_path):
    L = lib(tmp_path, entry(exp=18.0))
    assert L.find_dark(light(exposure=18.08))[0] is not None       # 18.08 s light, 18 s dark
    assert L.find_dark(light(exposure=18.5))[0] is not None
    assert L.find_dark(light(exposure=18.6))[0] is None


def test_temperature_must_be_close_but_not_identical(tmp_path):
    L = lib(tmp_path, entry(temp=-10.0))
    assert L.find_dark(light(temp=-9.4))[0] is not None            # within 1.5 C
    m, why = L.find_dark(light(temp=-5.0))
    assert m is None and "-10.0 C" in why and "-5.0 C" in why and "tolerance" in why


def test_the_tolerance_is_adjustable(tmp_path):
    L = lib(tmp_path, entry(temp=-10.0))
    assert L.find_dark(light(temp=-7.0), tol=1.5)[0] is None
    assert L.find_dark(light(temp=-7.0), tol=4.0)[0] is not None


def test_the_closest_temperature_wins_then_more_frames_then_the_newest(tmp_path):
    L = lib(tmp_path, entry("far.fit", temp=-9.0), entry("near.fit", temp=-10.1, n=5), entry("also_near.fit", temp=-10.1, n=30))
    assert L.find_dark(light())[0]["file"] == "also_near.fit"       # same distance; more frames
    L = lib(tmp_path / "x", entry("old.fit", built="2026-01-01T00:00:00"), entry("new.fit", built="2026-09-01T00:00:00"))
    assert L.find_dark(light())[0]["file"] == "new.fit"


def test_the_reason_when_the_library_is_empty(tmp_path):
    assert Library(tmp_path / "none").find_dark(light()) == (None, "the library has no dark frames yet")


def test_missing_values_in_a_header_do_not_block_a_match(tmp_path):
    # e.g. a camera that does not write XBINNING: treat unknown as compatible rather than refusing
    assert lib(tmp_path, entry()).find_dark(light(binning=None))[0] is not None


# ---- grouping ------------------------------------------------------------------------------

def test_darks_are_grouped_by_conditions_and_temperature():
    frames = [dark(i) for i in range(6)] + [dark(i, exp=120.0) for i in range(4)] + [dark(i, temp=0.0) for i in range(5)]
    groups = cal.group_darks(frames)
    sizes = sorted(len(g) for g in groups.values())
    assert sizes == [4, 5, 6]                                         # 300s cold, 300s at 0 C, 120 s


def test_a_slow_temperature_drift_stays_in_one_group():
    frames = [dict(dark(i), temp=-10.0 + 0.1 * i) for i in range(8)]   # -10.0 .. -9.3
    assert len(cal.group_darks(frames)) == 1


def test_light_conditions_use_the_median_temperature():
    frames = [{"instrument": "CAM", "binning": 1, "gain": 120, "offset": 50, "exptime": 300.0, "temp": t} for t in (-10.1, -10.0, -9.2)]
    c = cal.light_conditions(frames)
    assert c["temp"] == -10.0 and c["exposure"] == 300.0 and c["gain"] == 120


# ---- building (Siril stubbed out) -----------------------------------------------------------

def fake_run(calls):
    def run(siril_bin, work, script, log):
        calls.append(script.read_text())
        (work / "master_dark.fit").write_text("master")
    return run


def test_masters_are_built_once_per_group_and_indexed(tmp_path):
    L = Library(tmp_path / "lib")
    frames = [dark(i) for i in range(12)] + [dark(i, exp=120.0) for i in range(10)]
    calls, logs = [], []
    built = cal.build_masters(frames, L, None, run=fake_run(calls), log=logs.append)
    assert len(built) == 2 and len(calls) == 2 and all("stack dark rej w 3 3 -nonorm" in c for c in calls)
    assert {e["exposure"] for e in L.masters()} == {300.0, 120.0}
    assert all((L.root / e["file"]).exists() for e in L.masters())
    assert not (L.root / ".work").exists() or not list((L.root / ".work").iterdir())     # temporary files are cleaned up
    again = cal.build_masters(frames, Library(tmp_path / "lib"), None, run=fake_run(calls), log=logs.append)
    assert again == [] and len(calls) == 2 and any("already in the library" in l for l in logs)  # same frames: not rebuilt


def test_too_few_frames_are_skipped_with_a_message(tmp_path):
    logs = []
    assert cal.build_masters([dark(i) for i in range(2)], Library(tmp_path / "l"), None, run=fake_run([]), log=logs.append) == []
    assert any("need at least 3" in l for l in logs)


def test_a_small_but_usable_set_builds_with_a_warning(tmp_path):
    logs = []
    built = cal.build_masters([dark(i) for i in range(5)], Library(tmp_path / "l"), None, run=fake_run([]), log=logs.append)
    assert len(built) == 1 and any("warning: only 5 frames" in l for l in logs)


def test_dry_run_builds_nothing(tmp_path):
    calls, logs = [], []
    L = Library(tmp_path / "l")
    cal.build_masters([dark(i) for i in range(12)], L, None, dry_run=True, run=fake_run(calls), log=logs.append)
    assert calls == [] and L.masters() == [] and any("would build" in l for l in logs)


def test_frames_without_exposure_or_gain_are_skipped(tmp_path):
    f = [dict(dark(i), gain=None) for i in range(5)]
    logs = []
    assert cal.build_masters(f, Library(tmp_path / "l"), None, run=fake_run([]), log=logs.append) == []
    assert any("without exposure or gain" in l for l in logs)


def test_the_library_path_prefers_the_argument_then_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("FIRSTLIGHT_LIBRARY", str(tmp_path / "env"))
    assert cal.library_path(tmp_path / "arg") == tmp_path / "arg" and cal.library_path() == tmp_path / "env"
    monkeypatch.delenv("FIRSTLIGHT_LIBRARY")
    assert cal.library_path() == cal.DEFAULT_LIBRARY


def test_row_order_is_matched_to_the_frame():
    a = np.arange(6).reshape(3, 2)
    assert cal.orient_like(a, True, True) is a and np.array_equal(cal.orient_like(a, True, False), a[::-1])


# ---- real Siril on small synthetic frames ----------------------------------------------------

def _siril():
    from firstlight import siril
    try:
        return siril.find_siril(None)
    except siril.SirilError:
        return None


@pytest.mark.skipif(_siril() is None, reason="siril-cli not installed")
def test_a_real_master_dark_is_built_and_removes_the_hot_pixels_when_applied(tmp_path):
    from astropy.io import fits
    from firstlight import classify, stack

    rng = np.random.default_rng(1)
    hot = (20, 30)
    def write(name, kind, sky, exp=60.0):
        a = rng.normal(sky, 8, (64, 64)) + 100
        a[hot] += 3000                                           # a hot pixel present in darks and lights
        h = fits.Header(); h.update({"IMAGETYP": kind, "EXPTIME": exp, "GAIN": 120, "OFFSET": 50, "CCD-TEMP": -10.0,
                                     "XBINNING": 1, "INSTRUME": "CAM", "FILTER": "H", "OBJECT": "T"})
        fits.PrimaryHDU(np.clip(a, 0, 65535).astype(np.uint16), header=h).writeto(tmp_path / name)
    darks = tmp_path / "darks"; darks.mkdir()
    for i in range(6):
        write(f"darks/d{i}.fits", "Dark", 0)
    manifest = classify.build_manifest(classify.scan(darks))
    lib = Library(tmp_path / "lib")
    built = cal.build_masters(manifest["calibration"]["dark"], lib, _siril(), log=lambda *_: None)
    assert len(built) == 1
    with fits.open(lib.path_of(built[0])) as hd:
        m = np.asarray(hd[0].data, dtype=float)
    assert m.shape == (64, 64) and m[tuple(reversed(hot))] > 5 * np.median(m) or m[hot] > 5 * np.median(m)   # hot pixel captured
    found, _ = lib.find_dark({"instrument": "CAM", "binning": 1, "gain": 120, "offset": 50, "exposure": 60.0, "temp": -10.2})
    assert found is not None
