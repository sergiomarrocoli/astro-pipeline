from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from firstlight import enhance, siril
from firstlight.enhance import DECONVOLVE, DENOISE, ORDER, REMOVE_STARS, build_variant, normalise, variant_name


def test_flags_and_canonical_order():
    assert [s.key for s in ORDER] == ["rl", "dn", "sx"]
    assert [s.attr for s in ORDER] == ["deconvolve", "denoise", "remove_stars"]
    assert [s.flag for s in ORDER] == ["--deconvolve", "--denoise", "--remove-stars"]


def test_widget_assumes_deconvolve_and_denoise_but_not_star_removal():
    assert [s.key for s in ORDER if s.in_widget] == ["rl", "dn"]  # the widget needs the stars in its data


def test_normalise_orders_steps_and_drops_pointless_deconvolution():
    assert [s.key for s in normalise([DENOISE, DECONVOLVE])] == ["rl", "dn"]
    assert [s.key for s in normalise([DECONVOLVE, REMOVE_STARS])] == ["sx"]  # stars are discarded anyway
    assert normalise([]) == ()


def test_variant_names():
    assert variant_name([]) == "linear"
    assert variant_name(normalise([DENOISE, DECONVOLVE])) == "linear+rl+dn"


def test_layer_scripts_never_save_over_their_input():
    for script in (enhance.denoise_script("H"), enhance.deconvolve_script("H"), enhance.starnet_script("H")):
        lines = script.splitlines()
        assert lines[0].startswith("requires 1.4") and lines[1] == "load in_H" and lines[-1] == "close"
        assert not any(l == "save in_H" for l in lines)


def test_deconvolution_runs_on_the_stars_layer_with_a_psf_from_the_real_image():
    assert "starless" not in enhance.deconvolve_script("H")
    assert "makepsf load psf_H.fit" in enhance.deconvolve_script("H")
    assert "save stars_rl_H" in enhance.deconvolve_script("H")
    assert "makepsf stars -sym -savepsf=psf_H.fit" in enhance.psf_script("H")  # measured on the plain stack
    assert "save starless_dn_H" in enhance.denoise_script("H")


def test_starnet_splitter_availability(monkeypatch):
    monkeypatch.setattr(siril, "starnet_exe", lambda: None)
    assert "StarNet" in enhance.unavailable("starnet")
    assert enhance.unavailable("classical") is None
    monkeypatch.setattr(siril, "starnet_exe", lambda: Path("/bin/ls"))
    assert enhance.unavailable("starnet") is None


def test_starnet_exe_reads_siril_config(tmp_path, monkeypatch):
    cfg = tmp_path / ".config" / "siril"; cfg.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    (cfg / "config.1.4.ini").write_text("[core]\nstarnet_exe=\nstarnet_weights=\n")
    assert siril.starnet_exe() is None
    (cfg / "config.1.4.ini").write_text(f"[core]\nstarnet_exe={tmp_path}/missing\n")
    assert siril.starnet_exe() is None
    real = tmp_path / "starnet++"; real.write_text("x")
    (cfg / "config.1.4.ini").write_text(f"[core]\nstarnet_exe={real}\n")
    assert siril.starnet_exe() == real


@pytest.mark.parametrize("text,code,expected", [
    ("log: Script execution finished successfully.", 0, False),
    ("error: no suitable data in src fits\nlog: NL-Bayes execution time: 5 s", 0, False),  # denoise's harmless message
    ("log: Error in line 3 ('register'): invalid input sequence.\nScript execution failed.", 0, True),
    ("anything", 1, True),
])
def test_failure_detection(text, code, expected):
    assert siril.failed(code, text) is expected


# ---- orchestration, with Siril and the splitter stubbed out ----------------------------

@pytest.fixture
def calls(monkeypatch):
    log = []
    monkeypatch.setattr(enhance, "_stage", lambda d, name, *a, **k: log.append(("siril", d.name, name)))
    monkeypatch.setattr(enhance, "_py_stage", lambda d, name, *a, **k: log.append(("py", d.name, name)))
    monkeypatch.setattr(enhance, "recombine", lambda tdir, names, steps, dry, cfg=None: tdir / variant_name(steps))
    monkeypatch.setattr(enhance, "final_layers", lambda tdir, names, steps, dry, cfg=None: tdir / ("final" + variant_name(steps)[6:]))
    return log


def test_no_steps_is_the_plain_stack_and_does_no_work(tmp_path, calls):
    assert build_variant(tmp_path, ["H"], [], "classical", None) == tmp_path / "linear" and calls == []


def test_denoise_only_splits_then_denoises_the_starless_layer(tmp_path, calls):
    out = build_variant(tmp_path, ["H", "O"], [DENOISE], "classical", None)
    assert out == tmp_path / "linear+dn"
    assert calls == [("py", "layers", "split"), ("siril", "layers", "dn_H"), ("siril", "layers", "dn_O")]


def test_deconvolve_measures_psf_then_deconvolves_the_stars_layer(tmp_path, calls):
    build_variant(tmp_path, ["H"], [DECONVOLVE], "classical", None)
    assert calls == [("py", "layers", "split"), ("siril", "layers", "psf_H"), ("siril", "layers", "rl_H")]


def test_remove_stars_skips_deconvolution_and_uses_starnet_stage_when_asked(tmp_path, calls):
    out = build_variant(tmp_path, ["H"], [DECONVOLVE, REMOVE_STARS], "starnet", None)
    assert out == tmp_path / "linear+sx"
    assert calls == [("siril", "layers", "sn_H"), ("py", "layers", "stars")]  # no rl, no dn


# ---- recombination on real (tiny) FITS files --------------------------------------------

def _fits(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.asarray(arr, dtype=np.float32), header=fits.Header({"FILTER": "H"})).writeto(path, overwrite=True)


def _setup(tmp_path):
    for sub, name, v in (("linear", "linear_H", 1.0), ("layers", "starless_H", 2.0), ("layers", "stars_H", 3.0),
                         ("layers", "starless_dn_H", 20.0), ("layers", "stars_rl_H", -5.0)):
        _fits(tmp_path / sub / f"{name}.fit", np.full((4, 4), v))


FULL_DENOISE = {**enhance.default_cfg(), "denoise_amount": 1.0}


def _out(tmp_path, steps, cfg=FULL_DENOISE):
    d = enhance.recombine(tmp_path, ["H"], normalise(steps), False, cfg)
    return fits.getdata(d / "linear_H.fit"), fits.getheader(d / "linear_H.fit")


def test_recombine_picks_the_right_layers(tmp_path):
    _setup(tmp_path)
    assert _out(tmp_path, [DENOISE])[0].mean() == pytest.approx(20 + 3, abs=1e-3)   # denoised starless + stars
    assert _out(tmp_path, [REMOVE_STARS])[0].mean() == 2                            # starless only
    assert _out(tmp_path, [REMOVE_STARS, DENOISE])[0].mean() == 20                  # denoised starless only


def test_recombine_without_deconvolution_does_not_clip_negative_stars(tmp_path):
    _setup(tmp_path)
    _fits(tmp_path / "layers" / "stars_H.fit", np.full((4, 4), -0.5))  # noise can make the exact stars layer negative
    assert _out(tmp_path, [DENOISE])[0].mean() == pytest.approx(20 - 0.5, abs=1e-3)  # not clipped (only gated)


def test_recombine_clips_deconvolution_undershoot(tmp_path):
    _setup(tmp_path)  # stars_rl is all -5: negative lobes
    data, header = _out(tmp_path, [DECONVOLVE])
    assert data.mean() == 2.0  # starless + max(-5, 0): the dark rings cannot come back
    assert header["FILTER"] == "H"  # header carried over from the plain stack


def test_stars_layer_noise_is_gated_when_denoising_but_bright_stars_survive(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.normal(5, 1.0, (64, 64)).astype(np.float32)
    stars = rng.normal(0, 1.0, (64, 64)).astype(np.float32)   # masked-region noise
    stars[30, 30] = 500.0                                      # a real star
    _fits(tmp_path / "linear" / "linear_H.fit", base)
    _fits(tmp_path / "layers" / "starless_H.fit", np.zeros((64, 64)))
    _fits(tmp_path / "layers" / "starless_dn_H.fit", np.zeros((64, 64)))
    _fits(tmp_path / "layers" / "stars_H.fit", stars)
    gated = _out(tmp_path, [DENOISE])[0]
    rest = np.delete(gated.ravel(), 30 * 64 + 30)
    assert (rest == 0).mean() > 0.98 and rest.max() < 2.0               # noise-only pixels (sigma 1) shrink to 0 (rare >3.5 sigma ones keep a sliver)
    assert 495 < gated[30, 30] < 500                                    # the star keeps its flux
    assert np.allclose(_out(tmp_path, [REMOVE_STARS])[0], 0)            # --remove-stars never adds stars back


def test_denoise_amount_blends_the_denoised_and_plain_starless_layers(tmp_path):
    _setup(tmp_path)  # starless 2, denoised starless 20, stars 3
    cfg = {**enhance.default_cfg(), "denoise_amount": 0.5}
    assert _out(tmp_path, [REMOVE_STARS, DENOISE], cfg)[0].mean() == pytest.approx(0.5 * 20 + 0.5 * 2)


def test_final_layers_keeps_starless_and_stars_apart(tmp_path):
    _setup(tmp_path)
    d = enhance.final_layers(tmp_path, ["H"], normalise([DENOISE]), False, FULL_DENOISE)
    assert d.name == "final+dn"
    assert fits.getdata(d / "starless_H.fit").mean() == 20
    assert fits.getdata(d / "stars_H.fit").mean() == pytest.approx(3, abs=1e-3)
    d2 = enhance.final_layers(tmp_path, ["H"], normalise([REMOVE_STARS]), False, FULL_DENOISE)
    assert d2.name == "final+sx" and fits.getdata(d2 / "stars_H.fit").max() == 0   # stars discarded, layer kept (empty)


def test_final_layers_with_no_steps_is_just_the_split(tmp_path):
    _setup(tmp_path)
    d = enhance.final_layers(tmp_path, ["H"], (), False, FULL_DENOISE)
    assert d.name == "final"
    assert fits.getdata(d / "starless_H.fit").mean() == 2 and fits.getdata(d / "stars_H.fit").mean() == 3


def test_build_layers_always_splits_even_with_no_steps(tmp_path, calls):
    out = enhance.build_layers(tmp_path, ["H"], [], "classical", None)
    assert out.name == "final" and calls == [("py", "layers", "split")]


def test_rl_iterations_come_from_the_settings():
    assert "-iters=7" in enhance.deconvolve_script("H", 7)
    assert "-iters=4" in enhance.deconvolve_script("H")
