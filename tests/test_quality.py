import json

import numpy as np
import pytest

from firstlight import quality


def star_frame(fwhm=2.4, n_stars=40, size=400, seed=0, noise=3.0, roundness=1.0, hot=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(2000, noise, (size, size)).astype(np.float32)
    yy, xx = np.mgrid[:size, :size]
    sy, sx = fwhm / 2.355, fwhm / 2.355 * roundness
    srng = np.random.default_rng(1234)                       # same star field in every frame
    for _ in range(n_stars):
        cx, cy = srng.integers(20, size - 20, 2)
        a += (srng.uniform(400, 3000) * np.exp(-((xx - cx) ** 2) / (2 * sx ** 2) - ((yy - cy) ** 2) / (2 * sy ** 2))).astype(np.float32)
    for _ in range(hot):
        a[rng.integers(5, size - 5), rng.integers(5, size - 5)] += 2500
    return a


@pytest.mark.parametrize("fwhm", [2.0, 3.0, 4.0])
def test_measures_star_width(fwhm):
    m = quality.measure_array(star_frame(fwhm))
    assert m["fwhm"] == pytest.approx(fwhm, rel=0.15) and m["roundness"] > 0.95 and m["measured"] > 10


@pytest.mark.parametrize("fwhm,rr", [(3.5, 0.6), (3.5, 0.8), (5.0, 0.7), (2.5, 0.9)])
def test_roundness_is_the_axis_ratio(fwhm, rr):
    # an elliptical window measures the true ratio; a circular one would report an elongated star as rounder
    assert quality.measure_array(star_frame(fwhm, roundness=rr, seed=2))["roundness"] == pytest.approx(rr, abs=0.06)


def test_hot_pixels_are_not_counted_as_stars_or_measured_as_stars():
    clean, hot = quality.measure_array(star_frame(2.4, hot=0)), quality.measure_array(star_frame(2.4, hot=300))
    assert hot["fwhm"] == pytest.approx(clean["fwhm"], rel=0.1)
    assert abs(hot["stars"] - clean["stars"]) <= 0.1 * clean["stars"] + 2


def test_more_stars_means_a_higher_count():
    assert quality.count_stars(star_frame(n_stars=80)) > 1.5 * quality.count_stars(star_frame(n_stars=30))


def group(n=6, bad=None):
    """metrics for n normal frames plus the listed bad ones, all in one group"""
    normal = {"fwhm": 2.1, "roundness": 0.89, "stars": 5000, "noise": 18.0, "sky": 2050.0, "measured": 250}
    metrics = {f"f{i}": {**normal, "fwhm": 2.1 + 0.03 * (i % 3)} for i in range(n)}
    for name, change in (bad or {}).items():
        metrics[name] = {**normal, **change}
    return metrics, {p: "G" for p in metrics}


def test_a_soft_frame_is_rejected_with_a_reason():
    metrics, groups = group(bad={"soft": {"fwhm": 2.8}})
    v = quality.judge(metrics, groups)
    assert v["soft"] and "soft" in v["soft"][0] and all(not v[f"f{i}"] for i in range(6))


def test_few_stars_and_elongation_are_rejected_too():
    metrics, groups = group(bad={"cloud": {"stars": 3200}, "wind": {"roundness": 0.70}})
    v = quality.judge(metrics, groups)
    assert "few stars" in v["cloud"][0] and "elongated" in v["wind"][0]


def test_gentle_trends_are_not_rejected():
    metrics, groups = group(bad={"late": {"stars": 4100, "fwhm": 2.25, "roundness": 0.87}, "dip": {"fwhm": 2.4}})
    v = quality.judge(metrics, groups)
    assert not v["late"] and not v["dip"]


def test_a_slow_decline_through_the_night_is_not_held_against_the_last_frames():
    # the real pattern: the sky brightens toward dawn and the star count falls to ~0.65 of where it started
    stars = [5600, 5500, 5400, 5200, 5000, 4700, 4400, 4100, 3800, 3600]
    metrics = {f"f{i}": {"fwhm": 2.1, "roundness": 0.8, "stars": s, "noise": 18.0, "sky": 2050.0, "measured": 250} for i, s in enumerate(stars)}
    groups = {p: "G" for p in metrics}
    order = {p: f"{i:02d}" for i, p in enumerate(metrics)}
    assert not any(quality.judge(metrics, groups, order=order).values())
    # ...but the same number of stars is a clear outlier when it is a sudden dip among normal neighbours
    metrics["f4"]["stars"] = 3300
    assert quality.judge(metrics, groups, order=order)["f4"]


def test_nan_free_measurements_for_every_frame_in_a_real_star_field():
    # a fit that fails on one star must not turn the whole frame's median into NaN
    m = quality.measure_array(star_frame(2.4, n_stars=60, seed=5))
    assert m["fwhm"] is not None and np.isfinite(m["fwhm"]) and np.isfinite(m["roundness"])


def test_small_groups_are_never_judged():
    metrics, groups = group(n=2, bad={"soft": {"fwhm": 9.0}})
    assert not quality.judge(metrics, groups)["soft"]            # three frames: not enough to know what normal is


def test_groups_are_judged_separately():
    a, ga = group(); b, gb = group()
    metrics = {**{f"a{k}": v for k, v in a.items()}, **{f"b{k}": {**v, "fwhm": v["fwhm"] * 0.7} for k, v in b.items()}}  # a sharper filter
    groups = {**{f"a{k}": "A" for k in a}, **{f"b{k}": "B" for k in b}}
    assert not any(quality.judge(metrics, groups).values())      # B is sharper than A, but each is normal for itself


def test_thresholds_can_be_overridden():
    metrics, groups = group(bad={"soft": {"fwhm": 2.4}})
    assert not quality.judge(metrics, groups)["soft"]
    assert quality.judge(metrics, groups, {"reject_fwhm_ratio": 1.05})["soft"]


def test_measurements_are_cached(tmp_path, monkeypatch):
    from astropy.io import fits
    p = tmp_path / "a.fits"; fits.PrimaryHDU(star_frame()).writeto(p)
    cache = tmp_path / "cache.json"
    first = quality.measure_all([p], cache)
    monkeypatch.setattr(quality, "measure_frame", lambda path: (_ for _ in ()).throw(AssertionError("measured again")))
    assert quality.measure_all([p], cache) == first and json.loads(cache.read_text())
    p.write_bytes(p.read_bytes() + b"\0" * 2880)             # a changed file is measured again
    with pytest.raises(AssertionError):
        quality.measure_all([p], cache)


def test_roundness_is_ignored_for_undersampled_stars():
    # stars this small cannot have their shape measured, so a "round" outlier there must not cost a frame
    normal = {"fwhm": 1.4, "roundness": 0.85, "stars": 5000, "noise": 20.0, "sky": 2070.0, "measured": 250}
    metrics = {f"f{i}": dict(normal) for i in range(6)}
    metrics["odd"] = {**normal, "roundness": 0.40}
    assert not quality.judge(metrics, {p: "G" for p in metrics})["odd"]


def spread(metrics):
    """a time order that puts the named bad frames among the normal ones, as real outliers are"""
    normal = [p for p in metrics if p.startswith("f")]
    bad = [p for p in metrics if not p.startswith("f")]
    order = {p: f"{2 * i:02d}" for i, p in enumerate(normal)}
    for j, p in enumerate(bad):
        order[p] = f"{2 * (2 * j + 1) + 1:02d}"
    return order


def test_cap_spares_the_least_bad_frames_when_too_many_would_go():
    metrics, groups = group(n=8, bad={"awful": {"fwhm": 4.0, "stars": 2000}, "bad": {"fwhm": 2.8}, "meh": {"fwhm": 2.7}})
    verdict = quality.judge(metrics, groups, order=spread(metrics))
    assert sum(1 for v in verdict.values() if v) == 3
    capped = quality.cap(verdict, groups, metrics, max_fraction=0.25)     # 11 frames: at most 2 may go
    assert sum(1 for v in capped.values() if v) == 2
    assert capped["awful"] and capped["bad"] and not capped["meh"]        # the two worst stay rejected


def test_cap_leaves_a_normal_night_alone():
    metrics, groups = group(n=8, bad={"soft": {"fwhm": 2.9}})
    verdict = quality.judge(metrics, groups)
    assert quality.cap(verdict, groups, metrics) == verdict
