"""Generate CCT brand assets (banner + mark) as SVG.

The visual motif is the theory's own central parameter: kappa, the
von Mises-Fisher concentration of a multi-planet group's angular-momentum
vectors. Left-to-right the vectors go from scattered (kappa -> 0) to
aligned (kappa -> inf). This is the quantity the v3 test series scanned.

Deterministic: fixed seed, no randomness at render time.
Outputs are committed so GitHub can serve them directly.
"""
import math
import os

SEED = 1729
OUT_DIR = "assets"

# ---- CCT palette -----------------------------------------------------------
DEEP     = "#070B1F"   # deep space
MID      = "#111A44"   # indigo
VIOLET   = "#1E1240"   # violet shadow
SCATTER  = "#A78BFA"   # low-kappa accent
MIDTONE  = "#7DD3FC"
ALIGNED  = "#4EE1E0"   # high-kappa accent
AMBER    = "#FFB86B"   # stellar accent
TEXT     = "#E8ECFB"
MUTED    = "#8A93BF"


def lcg(seed):
    """Tiny deterministic PRNG so output never drifts between runs."""
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield state / (2**31)


def lerp_hex(c1, c2, t):
    c1 = c1.lstrip("#"); c2 = c2.lstrip("#")
    out = []
    for i in (0, 2, 4):
        a = int(c1[i:i+2], 16)
        b = int(c2[i:i+2], 16)
        out.append(int(round(a + (b - a) * t)))
    return "#%02X%02X%02X" % tuple(out)


def ramp(t):
    """violet -> sky -> cyan across the coherence axis."""
    if t < 0.5:
        return lerp_hex(SCATTER, MIDTONE, t / 0.5)
    return lerp_hex(MIDTONE, ALIGNED, (t - 0.5) / 0.5)


def starfield(rng, n, w, h, seed_skip=0):
    parts = []
    for _ in range(seed_skip):
        next(rng)
    for _ in range(n):
        x = next(rng) * w
        y = next(rng) * h
        r = 0.5 + next(rng) * 1.3
        o = 0.10 + next(rng) * 0.55
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}" fill="#FFFFFF" opacity="{o:.2f}"/>'
        )
    return "\n    ".join(parts)


def vector_strip(rng, x0, x1, y, n=26, length=30):
    """Angular-momentum vectors: scattered at left, aligned at right."""
    parts = []
    for i in range(n):
        t = i / (n - 1)
        cx = x0 + (x1 - x0) * t
        # deterministic jitter, damped as coherence rises
        jitter = (next(rng) - 0.5) * 2.0
        spread = (1.0 - t) ** 1.35
        ang = math.radians(jitter * 78.0 * spread)
        dx = math.sin(ang) * length
        dy = -math.cos(ang) * length
        col = ramp(t)
        op = 0.45 + 0.55 * t
        # shaft
        parts.append(
            f'<line x1="{cx:.1f}" y1="{y:.1f}" x2="{cx+dx:.1f}" y2="{y+dy:.1f}" '
            f'stroke="{col}" stroke-width="2" stroke-linecap="round" opacity="{op:.2f}"/>'
        )
        # arrowhead
        hx, hy = cx + dx, y + dy
        for side in (+1, -1):
            a2 = ang + side * math.radians(28)
            bx = hx - math.sin(a2) * 7.5
            by = hy + math.cos(a2) * 7.5
            parts.append(
                f'<line x1="{hx:.1f}" y1="{hy:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                f'stroke="{col}" stroke-width="2" stroke-linecap="round" opacity="{op:.2f}"/>'
            )
        # base node
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{y:.1f}" r="1.8" fill="{col}" opacity="{op*0.9:.2f}"/>'
        )
    return "\n    ".join(parts)


def build_banner():
    W, H = 1280, 470
    rng = lcg(SEED)
    stars = starfield(rng, 190, W, H)
    vecs = vector_strip(rng, 150, 1130, 366, length=28)

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Coherent Capture Theory">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{DEEP}"/>
      <stop offset="52%" stop-color="{MID}"/>
      <stop offset="100%" stop-color="{VIOLET}"/>
    </linearGradient>
    <radialGradient id="halo" cx="50%" cy="34%" r="62%">
      <stop offset="0%" stop-color="{ALIGNED}" stop-opacity="0.20"/>
      <stop offset="60%" stop-color="{SCATTER}" stop-opacity="0.07"/>
      <stop offset="100%" stop-color="{DEEP}" stop-opacity="0"/>
    </radialGradient>
    <!-- userSpaceOnUse: a horizontal line has a zero-height bbox, so the
         default objectBoundingBox units would leave the gradient undefined -->
    <linearGradient id="rule" gradientUnits="userSpaceOnUse" x1="300" y1="0" x2="980" y2="0">
      <stop offset="0%" stop-color="{SCATTER}" stop-opacity="0"/>
      <stop offset="22%" stop-color="{SCATTER}" stop-opacity="0.9"/>
      <stop offset="50%" stop-color="{MIDTONE}" stop-opacity="0.9"/>
      <stop offset="78%" stop-color="{ALIGNED}" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="{ALIGNED}" stop-opacity="0"/>
    </linearGradient>
  </defs>

  <rect width="{W}" height="{H}" fill="url(#sky)"/>
  <rect width="{W}" height="{H}" fill="url(#halo)"/>

  <g>
    {stars}
  </g>

  <!-- wordmark -->
  <text x="{W/2}" y="150" text-anchor="middle"
        font-family="Optima, Palatino, Georgia, serif" font-size="74" font-weight="600"
        letter-spacing="7" fill="{TEXT}">COHERENT CAPTURE</text>
  <text x="{W/2}" y="212" text-anchor="middle"
        font-family="Optima, Palatino, Georgia, serif" font-size="74" font-weight="300"
        letter-spacing="26" fill="{ALIGNED}" opacity="0.95">THEORY</text>

  <line x1="300" y1="250" x2="980" y2="250" stroke="url(#rule)" stroke-width="2.5"/>

  <text x="{W/2}" y="292" text-anchor="middle"
        font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
        font-size="17" letter-spacing="3.4" fill="{MUTED}">
    PRE-REGISTERED  &#183;  FALSIFIABLE  &#183;  REPRODUCIBLE
  </text>

  <!-- coherence axis -->
  <g>
    {vecs}
  </g>

  <line x1="150" y1="386" x2="1130" y2="386" stroke="{MUTED}" stroke-width="1" opacity="0.35"/>

  <text x="150" y="414" text-anchor="start"
        font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
        font-size="16" fill="{SCATTER}" opacity="0.95">&#954; &#8594; 0</text>
  <text x="150" y="436" text-anchor="start"
        font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
        font-size="12" fill="{MUTED}">scattered</text>

  <text x="{W/2}" y="424" text-anchor="middle"
        font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
        font-size="13" letter-spacing="1.6" fill="{MUTED}">
    angular-momentum coherence &#954; &#8212; the parameter under test
  </text>

  <text x="1130" y="414" text-anchor="end"
        font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
        font-size="16" fill="{ALIGNED}" opacity="0.95">&#954; &#8594; &#8734;</text>
  <text x="1130" y="436" text-anchor="end"
        font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
        font-size="12" fill="{MUTED}">aligned</text>
</svg>
'''


def build_mark():
    """Compact square mark: CCT monogram over the coherence fan."""
    S = 256
    rng = lcg(SEED + 7)
    stars = starfield(rng, 46, S, S)
    vecs = vector_strip(rng, 40, 216, 196, n=9, length=22)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{S}" height="{S}" viewBox="0 0 {S} {S}" role="img" aria-label="CCT">
  <defs>
    <linearGradient id="mg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{DEEP}"/>
      <stop offset="60%" stop-color="{MID}"/>
      <stop offset="100%" stop-color="{VIOLET}"/>
    </linearGradient>
    <radialGradient id="mh" cx="50%" cy="38%" r="60%">
      <stop offset="0%" stop-color="{ALIGNED}" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="{DEEP}" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="{S}" height="{S}" rx="52" fill="url(#mg)"/>
  <rect width="{S}" height="{S}" rx="52" fill="url(#mh)"/>
  <g>
    {stars}
  </g>
  <text x="{S/2}" y="132" text-anchor="middle"
        font-family="Optima, Palatino, Georgia, serif" font-size="72" font-weight="600"
        letter-spacing="2" fill="{TEXT}">CCT</text>
  <g>
    {vecs}
  </g>
</svg>
'''


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "cct-banner.svg"), "w") as f:
        f.write(build_banner())
    with open(os.path.join(OUT_DIR, "cct-mark.svg"), "w") as f:
        f.write(build_mark())
    print(f"wrote {OUT_DIR}/cct-banner.svg")
    print(f"wrote {OUT_DIR}/cct-mark.svg")
