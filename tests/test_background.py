import numpy as np
import pytest

from firstlight import background as bg


def scene(size=600, seed=0, noise=2.0, gradient=30.0, glow=40.0, nebula=0.0, stars=30):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    sky = rng.normal(100, noise, (size, size)).astype(np.float32)
    sky += gradient * (xx / size) * 0.7 + gradient * (yy / size) * 0.3                  # a tilted gradient
    sky += glow * np.exp(-((xx - 40) ** 2 + (yy - (size - 40)) ** 2) / (2 * 130 ** 2))    # a glow in one corner
    truth = sky - rng.normal(100, noise, (size, size)).astype(np.float32) * 0               # (noise-free reference below)
    nebula_map = np.zeros_like(sky)
    if nebula:
        nebula_map = nebula * np.exp(-((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2 * 90 ** 2))
        sky += nebula_map
    for _ in range(stars):
        cx, cy = rng.integers(10, size - 10, 2)
        sky += 800 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.3 ** 2))
    return sky.astype(np.float32), nebula_map


def corner_levels(a, k=80):
    h, w = a.shape
    c = lambda ys, xs: float(np.median(a[ys, xs]))
    return [c(slice(0, k), slice(0, k)), c(slice(0, k), slice(w - k, w)), c(slice(h - k, h), slice(0, k)),
            c(slice(h - k, h), slice(w - k, w)), c(slice(h // 2 - k, h // 2 + k), slice(w // 2 - k, w // 2 + k))]


def broad_glow(sigma, size=600, seed=0):
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    rng = np.random.default_rng(seed)
    return (rng.normal(100, 2, (size, size)).astype(np.float32)
            + 40 * np.exp(-((xx - 40) ** 2 + (yy - (size - 40)) ** 2) / (2 * sigma ** 2))).astype(np.float32)


def test_gradient_is_flattened():
    a, _ = scene(glow=0.0)
    out, info = bg.remove(a, block=30)
    assert info["applied"] and np.ptp(corner_levels(a)) > 15 and np.ptp(corner_levels(out)) < 4.0


def test_a_broad_corner_glow_is_only_reduced_and_a_compact_one_is_left_alone():
    # Documented limit: the fit never models a bright region, and a glow is a bright region (it looks like
    # nebulosity), so glows are reduced at best. Eating real signal is worse than leaving a little glow.
    broad, compact = broad_glow(300), broad_glow(130)
    assert np.ptp(corner_levels(bg.remove(broad, block=30)[0])) < 0.75 * np.ptp(corner_levels(broad))
    assert np.ptp(corner_levels(bg.remove(compact, block=30)[0])) > 0.9 * np.ptp(corner_levels(compact))


def test_level_is_preserved():
    a, _ = scene()
    out, _ = bg.remove(a, block=30)
    assert abs(float(np.median(out)) - float(np.median(a))) < 4.0


def test_nebula_is_mostly_kept():
    a, neb = scene(nebula=25.0, glow=0.0)
    out, info = bg.remove(a, block=30)
    centre = (slice(270, 330), slice(270, 330))
    kept = (np.median(out[centre]) - np.median(out[:60, 270:330])) / (np.median(a[centre]) - np.median(a[:60, 270:330]))
    assert info["applied"] and kept > 0.75          # at least three quarters of the nebula's excess survives


def test_emission_filling_half_the_frame_is_flagged_because_it_may_be_absorbed():
    # Documented limit: nothing can tell smooth emission over half the frame from a sky gradient by brightness,
    # so part of it is absorbed. The stage does not hide that: the correction is flagged as large, and the
    # warning tells the user to switch the step off for this target.
    a, _ = scene(gradient=10.0, glow=0.0, stars=0)
    a[:, a.shape[0] // 2:] += 30.0
    out, info = bg.remove(a, block=30)
    assert info["applied"] and info["large"] and info["amplitude_over_noise"] > bg.LARGE


def test_degree_zero_is_off_and_returns_a_copy():
    a, _ = scene()
    out, info = bg.remove(a, degree=0)
    assert not info["applied"] and np.array_equal(out, a) and out is not a


def test_exact_polynomial_is_recovered_without_noise():
    size = 480
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    u, v = (xx - size / 2) / (size / 2), (yy - size / 2) / (size / 2)
    truth = 50 + 8 * u - 5 * v + 3 * u * v + 4 * u ** 2
    out, info = bg.remove(truth.astype(np.float32), degree=2, block=24)
    assert np.abs(out - np.median(out)).max() < 0.05


def test_too_few_clean_blocks_means_no_correction():
    rng = np.random.default_rng(1)
    a = rng.normal(100, 2, (200, 200)).astype(np.float32)        # only 4 blocks: not enough to trust a cubic
    out, info = bg.remove(a, degree=3)
    assert not info["applied"] and np.array_equal(out, a)


def test_the_fitted_surface_has_the_expected_shape():
    params, info = bg.fit(scene()[0], block=30)
    s = bg.evaluate((600, 600), params)
    assert s.shape == (600, 600) and info["kept"] > 100 and np.isfinite(s).all()


def test_stage_writes_linear_files_from_aligned_ones_and_a_report(tmp_path):
    import json

    from astropy.io import fits

    tdir = tmp_path / "T"
    (tdir / "aligned").mkdir(parents=True)
    for n in ("H", "O"):
        a, _ = scene(size=960, glow=0.0, seed=1 if n == "H" else 2)
        h = fits.Header(); h["FILTER"] = n
        fits.PrimaryHDU(a, header=h).writeto(tdir / "aligned" / f"aligned_{n}.fit")
    assert bg.stage(tdir, ["H", "O"], 2) is True
    for n in ("H", "O"):
        out = fits.getdata(tdir / "linear" / f"linear_{n}.fit")
        assert out.shape == (960, 960) and fits.getheader(tdir / "linear" / f"linear_{n}.fit")["FILTER"] == n
    report = json.loads((tdir / "background.json").read_text())
    assert set(report) == {"H", "O"} and report["H"]["applied"]
    assert bg.stage(tdir, ["H", "O"], 2) is False                 # unchanged inputs and settings: skipped


def test_stage_with_degree_zero_still_produces_linear_files(tmp_path):
    from astropy.io import fits
    tdir = tmp_path / "T"
    (tdir / "aligned").mkdir(parents=True)
    a, _ = scene(size=300)
    fits.PrimaryHDU(a).writeto(tdir / "aligned" / "aligned_H.fit")
    bg.stage(tdir, ["H"], 0)
    assert np.array_equal(fits.getdata(tdir / "linear" / "linear_H.fit"), a)      # off means untouched, not missing
