# T18 audit — alpha tighter than s-process survives error subtraction

**Concern going in:** the T18 script computes raw per-cluster std for each
element and never subtracts measurement error. If GALAH's α-element errors
are systematically smaller than s-process errors (which they are: Mg/Fe ~0.024
dex, Si/Fe ~0.019, Ba/Fe ~0.052), the "α tighter than s-process" finding
might be a measurement-precision artifact rather than intrinsic-ISM signal.

**Result:** T18 survives cleanly. The intrinsic-scatter version drops by
only 2 percentage points (98.3 % → 96.4 %) and the Wilcoxon p stays at
~10⁻⁹⁷.

## Numbers

| version | median α scatter | median s-proc scatter | α/s-proc ratio | frac α<s-proc | Wilcoxon p |
|---|---|---|---|---|---|
| **raw std (the published T18)** | 0.0720 | 0.1817 | 0.39 | 98.3 % | 1.6 × 10⁻⁹⁸ |
| **intrinsic σ (error-subtracted)** | 0.0677 | 0.1712 | 0.39 | 96.4 % | 2.2 × 10⁻⁹⁷ |

GALAH errors used (median, in the cluster-matched sample): Mg/Fe 0.024,
Si/Fe 0.019, Ba/Fe 0.052 dex.

The error contribution to raw scatter: 33 % for α and 31 % for s-process —
i.e., both populations are dominated by intrinsic scatter, not measurement
noise. Only 0.5 % of clusters have α intrinsic scatter consistent with zero;
0.3 % for s-process.

## Caveat

The original T18 used Mg+Si for α and Ba+Ce for s-process; my rebuilt-FITS
audit uses Mg+Si for α and Ba alone for s-process (Ce not in the rebuilt
column set). The result with Ba+Ce vs Ba-only should be very similar — Ba
is the dominant s-process tracer in GALAH and the medians line up with
the published T18 figures. Could pull Ce from Data Central TAP for the 592
clusters if a stricter cross-check is wanted, but the 10⁻⁹⁷ headline is
clear enough that Ce wouldn't change the verdict.

## What it means

This is the first fully clean audit result this session. T18's
nucleosynthetic-hierarchy claim is real, not a measurement artifact:

- α-element scatter is ~2× tighter than s-process scatter in nearly every
  cluster, and the gap is much larger than measurement uncertainty in both
  channels.
- Consistent with the nucleosynthetic delay-time interpretation: CCSNe
  homogenize the birth cloud rapidly (~10 Myr); AGB enrichment is slower
  (~1–3 Gyr) and the s-process residual still carries inhomogeneous
  signatures at cluster-formation time.

## Files
- `t18_audit.py` — reproducible script
- `t18_audit_scatter.csv` — per-cluster raw + intrinsic scatter
