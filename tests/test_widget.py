import base64
import json
import re

import numpy as np

from firstlight.widget import (build_payload, channel_pack, default_assignment, downsample, quantise, render_html,
                               star_stretch_k)


def test_downsample_block_mean_and_shape():
    a = np.arange(16, dtype=float).reshape(4, 4)
    out = downsample(a, width=2)
    assert out.shape == (2, 2) and out[0, 0] == a[:2, :2].mean()


def test_downsample_small_image_is_untouched():
    assert downsample(np.ones((5, 7)), width=1000).shape == (5, 7)


def test_channel_pack_roundtrip_and_stats():
    rng = np.random.default_rng(1)
    a = rng.normal(100, 5, (64, 64))
    a[10, 10] = 1e5  # a "star"
    p = channel_pack(a)
    u = np.frombuffer(base64.b64decode(p["data"]), dtype="<u2")
    assert u.size == 64 * 64 and u.max() <= 65535
    assert 0 < p["bg"] < 0.05 and p["sigma"] > 0 and p["p99"] > 0


def test_channel_pack_flat_image_has_nonzero_sigma():
    assert channel_pack(np.full((8, 8), 3.0))["sigma"] > 0  # the page divides by sigma


def test_default_assignment_sho_hoo_and_fallback():
    assert default_assignment(["H", "O", "S"]) == ["S", "H", "O"]
    assert default_assignment(["Ha", "OIII"]) == ["Ha", "OIII", "OIII"]
    assert default_assignment(["H", "S"]) == ["H", "S", "S"]  # no O: an H/S bicolour, H red and S cyan
    assert default_assignment(["L", "R"]) == ["L", "R", "L"]  # unknown filters cycle in name order


def _layers(names, shape=(30, 40)):
    out = {}
    for i, n in enumerate(names):
        rng = np.random.default_rng(i)
        stars = np.zeros(shape); stars[5, 5] = 100 + i; stars[20, 30] = 50
        out[n] = (rng.normal(50, 2, shape), stars)
    return out


def test_star_stretch_k_puts_a_typical_star_at_the_target():
    q = np.linspace(0.002, 0.08, 200)   # real star pixels are faint compared with the brightest star
    k = star_stretch_k(q, 0.7)
    p95 = np.percentile(q, 95)
    assert abs(np.arcsinh(k * p95) / np.arcsinh(k) - 0.7) < 0.01
    assert star_stretch_k(np.zeros(10)) > 0  # no stars: still a valid number


def test_payload_has_both_layers_for_every_channel():
    payload = build_payload(_layers(["H", "O", "S"]))
    assert [c["name"] for c in payload["channels"]] == ["H", "O", "S"]
    assert payload["default"] == ["S", "H", "O"]
    assert (payload["width"], payload["height"]) == (40, 30)
    c = payload["channels"][0]
    n = 40 * 30
    assert len(base64.b64decode(c["data"])) == 2 * n and len(base64.b64decode(c["sdata"])) == 2 * n
    assert c["k"] > 0


def test_html_embeds_valid_json():
    html = render_html(build_payload(_layers(["H", "O"]), info="star FWHM 2.6 px"))
    blob = re.search(r"const P = (\{.*?\});\n", html, re.S).group(1)
    assert json.loads(blob)["default"] == ["H", "O", "O"] and "FWHM" in json.loads(blob)["info"]


def test_tiles_hold_both_layers_flipped_and_share_the_overview_scale(tmp_path):
    layers = _layers(["H", "O"], shape=(37, 50))
    payload = build_payload(layers, tile_dir=tmp_path / "widget_tiles", width=1000, tile_size=16)
    t = payload["tiles"]
    assert (t["w"], t["h"], t["cols"], t["rows"], t["shift"]) == (50, 37, 4, 3, 4)
    assert len(list((tmp_path / "widget_tiles").glob("t_*.js"))) == 12
    src = (tmp_path / "widget_tiles" / "t_0_0.js").read_text()
    m = re.match(r"__tile\((\d+),(\d+),(\d+),(\d+),(\{.*\})\);", src, re.S)
    assert m.group(3, 4) == ("16", "16")
    obj = json.loads(m.group(5))
    assert set(obj) == {"H", "H*", "O", "O*"}                       # starless and stars for each filter
    # Top-left tile is the TOP of the image: the last FITS rows, flipped. Both layers use the overview's vmax.
    sl, st = layers["H"]
    got = np.frombuffer(base64.b64decode(obj["H"]), "<u2").reshape(16, 16)
    assert (got == quantise(np.flipud(sl)[:16, :16], channel_pack(downsample(sl))["vmax"])).all()
    got_s = np.frombuffer(base64.b64decode(obj["H*"]), "<u2").reshape(16, 16)
    svmax = float(np.percentile(downsample(st), 99.99))
    assert (got_s == quantise(np.flipud(st)[:16, :16], svmax)).all()


def test_edge_tile_has_its_own_size(tmp_path):
    layers = {n: (np.ones((37, 50)), np.zeros((37, 50))) for n in "HO"}
    build_payload(layers, tile_dir=tmp_path / "t", tile_size=16)
    assert (tmp_path / "t" / "t_2_3.js").read_text().startswith("__tile(2,3,2,5,")  # 50 = 3*16 + 2 wide, 37 = 2*16 + 5 tall


def test_overview_factor_is_recorded_and_full_is_cropped_to_match(tmp_path):
    layers = {n: (np.ones((100, 100)), np.zeros((100, 100))) for n in "HO"}
    p = build_payload(layers, tile_dir=tmp_path / "t", width=30, tile_size=16)  # factor 3 -> 33x33 overview
    assert (p["factor"], p["width"], p["tiles"]["w"]) == (3, 33, 99)


def test_html_inlines_the_dsp_and_leaves_no_placeholders():
    html = render_html(build_payload(_layers(["H", "O"])))
    assert "function stretchP" in html and "function scnr" in html
    assert "__DSP__" not in html and "__PAYLOAD__" not in html


def test_faint_residue_does_not_drive_the_star_boost():
    rng = np.random.default_rng(0)
    faint = np.abs(rng.normal(0.0, 0.003, 5000)) + 0.011            # noise left in masked areas
    stars = np.concatenate([faint, rng.uniform(0.05, 0.6, 300)])    # plus real, moderately bright stars
    assert star_stretch_k(stars) < 0.2 * star_stretch_k(faint)       # keyed to the stars, not the residue


def test_black_point_uses_the_plain_stack_noise_when_given():
    layers = _layers(["H", "O"])
    vmax_h = channel_pack(downsample(layers["H"][0]))["vmax"]
    plain = build_payload(layers)["channels"][0]
    with_noise = build_payload(layers, noise={"H": 3.0, "O": 3.0})["channels"][0]
    assert with_noise["sigma"] == 3.0 / vmax_h and with_noise["sigma"] != plain["sigma"]
