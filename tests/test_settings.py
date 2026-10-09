import json

import numpy as np
import pytest

from firstlight import settings, starsplit


def star_image(fwhm, n=40, size=400, seed=3, noise=1.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(100, noise, (size, size)).astype(np.float32)
    y, x = np.mgrid[:size, :size]
    sig = fwhm / 2.355
    for _ in range(n):
        cx, cy = rng.integers(20, size - 20, 2)
        a += (rng.uniform(300, 2000) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig ** 2))).astype(np.float32)
    return a


@pytest.mark.parametrize("fwhm", [2.0, 3.0, 4.5])
def test_measure_fwhm_recovers_the_true_value(fwhm):
    assert settings.measure_fwhm(star_image(fwhm)) == pytest.approx(fwhm, rel=0.15)


def test_measure_fwhm_falls_back_when_there_are_no_stars():
    flat = np.random.default_rng(0).normal(100, 1, (200, 200)).astype(np.float32)
    assert settings.measure_fwhm(flat) == starsplit.DEFAULT_FWHM


def test_override_wins_and_survives_a_rewrite(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"auto": {"fwhm": 2.5}, "override": {"fwhm": 3.7}}))
    eff = settings.effective(tmp_path, {"fwhm": 2.2, "star_gate": 2.5})
    assert eff == {"fwhm": 3.7, "star_gate": 2.5}
    saved = json.loads((tmp_path / "settings.json").read_text())
    assert saved["auto"]["fwhm"] == 2.2 and saved["override"] == {"fwhm": 3.7}  # auto refreshed, override untouched


def test_no_existing_file_means_no_overrides(tmp_path):
    assert settings.effective(tmp_path, {"fwhm": 2.2}) == {"fwhm": 2.2}


def test_resolve_records_sky_noise_per_filter(tmp_path):
    from astropy.io import fits
    paths = {}
    for n, sigma in (("H", 1.0), ("O", 3.0)):
        p = tmp_path / f"linear_{n}.fit"
        fits.PrimaryHDU(star_image(2.5, size=200, noise=sigma).astype(np.float32)).writeto(p)
        paths[n] = p
    eff = settings.resolve(tmp_path / "t", paths)
    assert eff["noise"]["O"] > 2 * eff["noise"]["H"] and eff["fwhm_measured_on"] == "H"
    assert (tmp_path / "t" / "settings.json").exists()
