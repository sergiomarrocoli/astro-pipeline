import numpy as np
import pytest

from firstlight import hotpixels as hp


def field(seed, stars, hot, size=160, n=6, noise=2.0):
    """n frames of one field: the same sensor hot pixels, stars at this field's positions."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    base = np.full((size, size), 100.0, np.float32)
    for cy, cx, amp in stars:                                   # FWHM ~2 px stars
        base += (amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 0.85 ** 2))).astype(np.float32)
    for y, x in hot:
        base[y, x] += 400
    return [base + rng.normal(0, noise, base.shape).astype(np.float32) for _ in range(n)]


HOT = [(20, 30), (50, 120), (100, 60), (140, 140), (75, 75)]


def test_candidates_flag_spikes_but_not_stars():
    frames = field(1, [(40, 100, 800), (110, 40, 600)], HOT)
    med = np.median(frames, axis=0)
    m = hp.candidates(med)
    assert all(m[y, x] for y, x in HOT)
    assert not m[40, 100] and not m[110, 40]                    # a star's neighbours are raised: not isolated


def test_two_fields_keep_only_what_both_share():
    # An extremely sharp star (FWHM ~1.3 px) in one field passes the loose isolation rule but cannot be a
    # sensor defect: it is not in the other field.
    a = np.median(field(1, [(40, 100, 800), (100, 40, 9000)], HOT), axis=0)
    b = np.median(field(2, [(60, 20, 800)], HOT), axis=0)
    ma, mb = hp.candidates(a), hp.candidates(b)
    both = ma & mb
    assert all(both[y, x] for y, x in HOT) and both.sum() == len(HOT)


def test_find_writes_nothing_for_a_clean_sensor():
    clean = np.median(field(3, [(40, 100, 800)], []), axis=0)
    assert not hp.candidates(clean).any()


def test_find_end_to_end_with_files(tmp_path):
    from astropy.io import fits
    fields = {}
    for name, seed, stars in (("A", 1, [(40, 100, 800)]), ("B", 2, [(60, 20, 800)])):
        paths = []
        for i, fr in enumerate(field(seed, stars, HOT)):
            p = tmp_path / f"{name}{i}.fits"
            fits.PrimaryHDU(fr.astype(np.float32)).writeto(p)
            paths.append(p)
        fields[name] = paths
    mask, info = hp.find(fields)
    assert info["count"] == len(HOT) and all(mask[y, x] for y, x in HOT) and info["fields"] == ["A", "B"]


def test_single_field_uses_the_stricter_rule_and_needs_three_frames(tmp_path):
    from astropy.io import fits
    paths = []
    for i, fr in enumerate(field(1, [(40, 100, 800)], HOT)):
        p = tmp_path / f"a{i}.fits"; fits.PrimaryHDU(fr.astype(np.float32)).writeto(p); paths.append(p)
    mask, info = hp.find({"A": paths})
    assert "1 of 1" in info["rule"] and all(mask[y, x] for y, x in HOT) and not mask[40, 100]
    _, none = hp.find({"A": paths[:2]})
    assert none["count"] == 0


def test_list_format_and_row_order():
    m = np.zeros((10, 20), bool); m[2, 5] = True
    assert hp.to_list(m, top_down=False) == "P 5 2 H\n"
    assert hp.to_list(m, top_down=True) == "P 5 7 H\n"          # y counted from the bottom: 10 - 1 - 2
    assert hp.to_list(np.zeros((4, 4), bool), True) == ""


def test_row_order_is_read_from_the_header(tmp_path):
    from astropy.io import fits
    for order, expect in (("TOP-DOWN", True), ("BOTTOM-UP", False), (None, False)):
        p = tmp_path / f"{order}.fits"
        h = fits.Header()
        if order:
            h["ROWORDER"] = order
        fits.PrimaryHDU(np.zeros((4, 4), np.float32), header=h).writeto(p)
        assert hp.row_order_top_down(p) is expect


def test_apply_replaces_hot_pixels_with_the_mean_of_clean_neighbours():
    a = np.full((8, 8), 10.0, np.float32); a[3, 3] = 500; a[3, 4] = 400     # two adjacent hot pixels
    m = np.zeros((8, 8), bool); m[3, 3] = m[3, 4] = True
    out = hp.apply(a, m)
    assert out[3, 3] == pytest.approx(10.0) and out[3, 4] == pytest.approx(10.0) and out[0, 0] == 10.0
    assert a[3, 3] == 500                                                   # the input is untouched
