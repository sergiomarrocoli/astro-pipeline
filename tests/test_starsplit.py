import numpy as np
import pytest

from firstlight import starsplit as ss


def sky(shape=(240, 240), seed=1, noise=1.0):
    return np.random.default_rng(seed).normal(0, noise, shape).astype(np.float32)


def add_star(a, cx, cy, amp, sig=1.2):
    y, x = np.mgrid[: a.shape[0], : a.shape[1]]
    a += (amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig * sig))).astype(np.float32)


def test_gblur_keeps_constants_and_mass():
    assert np.allclose(ss.gblur(np.full((64, 64), 3.0, np.float32), 2.5), 3.0, atol=1e-4)
    imp = np.zeros((64, 64), np.float32); imp[32, 32] = 1
    b = ss.gblur(imp, 2.0)
    assert abs(b.sum() - 1) < 1e-3 and b[32, 32] == b.max()


def test_dilate_grows_a_pixel_into_a_square():
    m = np.zeros((21, 21), np.float32); m[10, 10] = 1
    assert ss.dilate(m, 2).sum() == 25


def test_split_removes_stars_and_keeps_nebulosity():
    a = sky() + 5
    y, x = np.mgrid[:240, :240]
    a += (40 * np.exp(-((x - 170) ** 2 + (y - 160) ** 2) / (2 * 25.0 ** 2))).astype(np.float32)  # broad nebula
    for cx, cy, amp in ((60, 60, 2000), (120, 80, 600), (50, 190, 300)):
        add_star(a, cx, cy, amp)
    mask, out = ss.split({"H": a, "O": a * 0.8 + sky(seed=2)})
    starless, stars = out["H"]
    assert starless[60, 60] < 60 and starless[80, 120] < 60 and starless[190, 50] < 60  # stars gone (was 2000/600/300)
    assert abs(starless[160, 170] - a[160, 170]) < 6                                   # nebula untouched
    assert stars[60, 60] > 1500
    assert mask[160, 170] < 0.1                                                         # not masked


def test_recombination_returns_the_original_exactly():
    a = sky() + 5
    for cx, cy in ((60, 60), (150, 120)):
        add_star(a, cx, cy, 800)
    _, out = ss.split({"H": a, "O": a.copy()})
    starless, stars = out["H"]
    assert np.abs(starless + stars - a).max() < 1e-3  # nothing is lost or biased by splitting


def test_filled_regions_have_the_skys_grain():
    a = sky(noise=2.0)
    add_star(a, 120, 120, 3000, sig=2.0)
    _, out = ss.split({"H": a, "O": a.copy()})
    s = out["H"][0]
    inside, outside = s[110:130, 110:130].std(), s[20:60, 20:60].std()
    assert 0.5 * outside < inside < 1.8 * outside  # not a flat, noise-free patch


def test_same_input_gives_the_same_output():
    a = sky(); add_star(a, 100, 100, 900)
    s1 = ss.split({"H": a, "O": a.copy()})[1]["H"][0]
    s2 = ss.split({"H": a, "O": a.copy()})[1]["H"][0]
    assert np.array_equal(s1, s2)  # fixed seed, so stage stamps stay valid


def test_one_mask_is_shared_by_all_filters():
    h, o = sky(seed=3), sky(seed=4)
    add_star(h, 100, 100, 900); add_star(o, 100, 100, 700)
    mask, out = ss.split({"H": h, "O": o})
    assert out["H"][1][100, 100] > 300 and out["O"][1][100, 100] > 300  # star lands in both stars layers


def test_faint_stars_are_removed_but_nebula_knots_of_the_same_brightness_stay():
    rng = np.random.default_rng(7)
    a = sky(shape=(300, 300), seed=11, noise=1.0) + 10
    stars = []
    for _ in range(60):  # faint stars, peak ~25 sigma
        cx, cy = rng.integers(20, 280, 2)
        add_star(a, cx, cy, 25.0, sig=1.1); stars.append((cy, cx))
    knots = [(80, 80), (200, 220), (150, 150)]   # nebula-sized features (sigma 5 px) with the same peak
    for cy, cx in knots:
        add_star(a, cx, cy, 25.0, sig=5.0)
    _, out = ss.split({"H": a, "O": a.copy()})
    starless = out["H"][0]
    left = np.mean([starless[y, x] - 10 for y, x in stars if min(abs(y - ky) + abs(x - kx) for ky, kx in knots) > 30])
    kept = np.mean([starless[y, x] - 10 for y, x in knots])
    assert left < 6    # faint stars mostly gone (peak was 25 above sky)
    assert kept > 17   # the knots survive (peak was 25 above sky)


def test_a_star_visible_in_only_one_filter_is_still_removed():
    h, o, s = sky(seed=21), sky(seed=22), sky(seed=23)
    add_star(s, 100, 100, 60.0)          # a red star: bright only in S, nothing in H or O
    add_star(h, 160, 60, 60.0); add_star(o, 160, 60, 60.0); add_star(s, 160, 60, 60.0)   # a white one for comparison
    _, out = ss.split({"H": h, "O": o, "S": s})
    starless_s = out["S"][0]
    assert starless_s[100, 100] < 12 and starless_s[60, 160] < 12   # both gone from S, not left as an orange dot


def blob_scene(level_sigma, size=240, noise=1.0, star_amp=800.0, seed=3):
    """a smooth bright blob (peak level_sigma above the sky) with one star right in the middle, same in all filters"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    blob = (level_sigma * noise * np.exp(-((xx - 120) ** 2 + (yy - 120) ** 2) / (2 * 40 ** 2))).astype(np.float32)
    out = {}
    for i, n in enumerate("HOS"):
        a = rng.normal(100, noise, (size, size)).astype(np.float32) + blob
        add_star(a, 120, 120, star_amp * (0.8 + 0.1 * i), sig=1.06)
        out[n] = a
    return out, blob


def removed_fraction(level_sigma):
    chans, blob = blob_scene(level_sigma)
    _, out = ss.split(chans)
    star_flux_in_stars_layer = out["H"][1][116:125, 116:125].sum()
    total = (chans["H"][116:125, 116:125] - 100 - blob[116:125, 116:125]).sum()
    return star_flux_in_stars_layer / total


def test_a_star_on_a_faint_background_is_removed():
    assert removed_fraction(3.0) > 0.9


def test_a_star_in_a_very_bright_knot_is_left_in_the_starless_layer():
    # a flat fill cannot know the peak it hides, so it would leave a dimple; the star stays instead
    assert removed_fraction(80.0) < 0.1


def test_the_protection_fades_in_smoothly_with_brightness():
    levels = (5, 20, 30, 40, 45, 80)
    fracs = [removed_fraction(l) for l in levels]
    # (a few percent above 1 before protection starts: the flat fill's small dimple adds blob flux to the stars layer)
    assert all(a >= b - 0.05 for a, b in zip(fracs, fracs[1:]))      # never removes MORE as the nebula gets brighter
    assert fracs[0] > 0.9 and fracs[-1] < 0.1 and 0.1 < fracs[3] < 0.9   # a real transition in between


def test_split_stays_exact_when_stars_are_protected():
    chans, _ = blob_scene(80.0)
    _, out = ss.split(chans)
    for n, a in chans.items():
        assert np.abs(out[n][0] + out[n][1] - a).max() < 1e-3


def test_protected_knots_leave_no_dimple():
    chans, blob = blob_scene(80.0)
    _, out = ss.split(chans)
    centre = out["H"][0][118:123, 118:123].mean() - 100
    assert centre > 0.8 * (blob[118:123, 118:123].mean())            # the starless layer keeps the knot's brightness
