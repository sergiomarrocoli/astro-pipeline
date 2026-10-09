// Colour and stretch primitives for the preview widget. No DOM access, so they can be tested alone.
// The heavy processing (star split, denoise, deconvolution) happens in the pipeline, not here.

// Midtones transfer function, and the m that maps input x to output t.
const mtf = (m, x) => x <= 0 ? 0 : x >= 1 ? 1 : (m - 1) * x / ((2 * m - 1) * x - m);
const solveM = (x, t) => (x <= 0 || x >= 1) ? 0.5 : x * (t - 1) / (2 * t * x - t - x);

// Map a background-subtracted, scale-matched value to display [0,1].
//   c: { lo, span, m }   S: { contrast }
function stretchP(c, p, S) {
  let x = (p - c.lo) / c.span;
  x = x < 0 ? 0 : x > 1 ? 1 : x;
  let y = mtf(c.m, x);
  if (S.contrast) y += S.contrast * (y * y * (3 - 2 * y) - y);     // gentle S-curve
  return y;
}

// Star layer stretch: asinh keeps faint stars visible without blowing out bright ones. q in [0,1].
const starStretch = (k, q) => q <= 0 ? 0 : Math.asinh(k * q) / Math.asinh(k);

// SCNR (average neutral): cap green at the mean of red and blue. amount 0..1.
function scnr(r, g, b, amount) {
  const cap = (r + b) / 2;
  return g > cap ? g - amount * (g - cap) : g;
}

// Screen blend: the way a stars layer is combined with a stretched starless layer.
const screen = (a, b) => 1 - (1 - a) * (1 - b);

// SCNR amount that suits an image whose mean channel levels are r, g, b (0 when green is not dominant).
function autoScnr(r, g, b) {
  const ratio = g / Math.max((r + b) / 2, 1e-6);
  return ratio <= 1.05 ? 0 : Math.min(1, (ratio - 1.05) * 4);
}

// Separable Gaussian blur of a Float32Array(w*h), clamped edges.
function gblur(src, w, h, sigma) {
  const out = new Float32Array(w * h);
  if (!(sigma > 0.05)) { out.set(src); return out; }
  const r = Math.max(1, Math.ceil(sigma * 3)), k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = -r; j <= r; j++) { let xx = x + j; xx = xx < 0 ? 0 : xx >= w ? w - 1 : xx; s += k[j + r] * src[y * w + xx]; }
    tmp[y * w + x] = s;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = -r; j <= r; j++) { let yy = y + j; yy = yy < 0 ? 0 : yy >= h ? h - 1 : yy; s += k[j + r] * tmp[yy * w + x]; }
    out[y * w + x] = s;
  }
  return out;
}

// Blur only the colour (Cb = B-Y, Cr = R-Y) of display-space r,g,b, in place, leaving luminance Y untouched.
// Returns the smoothed chroma so a higher-resolution luminance can be paired with it later.
function chromaSmooth(r, g, b, w, h, sigma) {
  const n = w * h, Y = new Float32Array(n), cb = new Float32Array(n), cr = new Float32Array(n);
  for (let i = 0; i < n; i++) { Y[i] = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]; cb[i] = b[i] - Y[i]; cr[i] = r[i] - Y[i]; }
  const cb2 = gblur(cb, w, h, sigma), cr2 = gblur(cr, w, h, sigma);
  for (let i = 0; i < n; i++) { r[i] = Y[i] + cr2[i]; b[i] = Y[i] + cb2[i]; g[i] = (Y[i] - 0.299 * r[i] - 0.114 * b[i]) / 0.587; }
  return { cb: cb2, cr: cr2 };
}

// Colour from smoothed chroma + luminance from the pixel itself (the detail view's trick).
function withChroma(y, cb, cr, out) {
  const r = y + cr, b = y + cb;
  out[0] = r; out[1] = (y - 0.299 * r - 0.114 * b) / 0.587; out[2] = b;
}

// ---- Frame geometry (rotate + crop) ------------------------------------------------
// The image (w x h px, origin top-left, y down) is rotated clockwise by rot degrees about its centre.
// The rotated image sits in a bounding box; a crop is a rectangle in that box given as fractions
// [left, top, right, bottom]. These functions are pure so they can be tested without a page.

// Bounding box of the rotated image.
function bboxOf(w, h, rot) {
  const t = rot * Math.PI / 180, c = Math.abs(Math.cos(t)), s = Math.abs(Math.sin(t));
  return { Wb: w * c + h * s, Hb: w * s + h * c };
}

// Largest crop (as fractions of the bounding box) with the given width/height aspect, centred, that lies
// entirely inside the rotated image, so no empty corners remain.
function fitInside(w, h, rot, aspect) {
  const t = rot * Math.PI / 180, c = Math.abs(Math.cos(t)), s = Math.abs(Math.sin(t)), { Wb, Hb } = bboxOf(w, h, rot);
  // the four corners (+-a*q, +-q) must satisfy |x c + y s| <= w/2 and |x s + y c| <= h/2
  const q = Math.min(w / (2 * (aspect * c + s)), h / (2 * (aspect * s + c)));
  const hw = aspect * q, hh = q;
  return [(Wb / 2 - hw) / Wb, (Hb / 2 - hh) / Hb, (Wb / 2 + hw) / Wb, (Hb / 2 + hh) / Hb];
}

// Geometry of a crop: bounding box, crop rectangle in box pixels.
function frameGeom(w, h, rot, crop) {
  const { Wb, Hb } = bboxOf(w, h, rot);
  return { Wb, Hb, x0: crop[0] * Wb, y0: crop[1] * Hb, Rw: (crop[2] - crop[0]) * Wb, Rh: (crop[3] - crop[1]) * Hb };
}

// CSS matrix(a,b,c,d,e,f) taking image pixels to view pixels when the crop fills a view k times larger.
function frameMatrix(w, h, rot, g, k) {
  const t = rot * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  return [k * c, k * s, -k * s, k * c,
    k * (g.Wb / 2 - g.x0 - (c * w / 2 - s * h / 2)), k * (g.Hb / 2 - g.y0 - (s * w / 2 + c * h / 2))];
}

// The inverse: a view pixel back to image pixels [u, v].
function viewToImage(w, h, rot, g, k, vx, vy) {
  const t = rot * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  const rx = vx / k + g.x0 - g.Wb / 2, ry = vy / k + g.y0 - g.Hb / 2;
  return [rx * c + ry * s + w / 2, -rx * s + ry * c + h / 2];
}
