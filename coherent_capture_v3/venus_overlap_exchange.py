"""Preregistered Venus overlap-region stellar-exchange N-body sweep.

See VENUS_OVERLAP_EXCHANGE_PREREGISTRATION.md. Do not change locked grids or
success definitions after outcome inspection; revisions require a new preregistration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import qmc

G_AU3_MSUN_YR2 = 4.0 * math.pi**2
KMS_TO_AUYR = 31557600.0 / 149597870.7
M_VENUS_MSUN = 4.8675e24 / 1.98847e30
A_D_AU = 0.723332
A_VENUS_AU = 0.723332
Q_RATIOS = (2.0, 3.0, 4.0)
VINF_KMS = (0.10, 0.30, 0.50, 1.00)
BATCHES = 4
BATCH_SIZE = 128
BASE_SEED = 20260910


def hyperbola_elements(q_au: float, v_inf_kms: float, m_total_msun: float) -> tuple[float, float]:
    """Return hyperbolic semimajor axis a<0 and eccentricity e>1."""
    vinf = v_inf_kms * KMS_TO_AUYR
    mu = G_AU3_MSUN_YR2 * m_total_msun
    a = -mu / (vinf * vinf)
    e = 1.0 + q_au * vinf * vinf / mu
    return a, e


def true_anomaly_for_radius(a: float, e: float, radius: float, inbound: bool = True) -> float:
    p = a * (1.0 - e * e)
    c = (p / radius - 1.0) / e
    c = max(-1.0, min(1.0, c))
    f = math.acos(c)
    return -f if inbound else f


def hyperbolic_time_from_periapsis(a: float, e: float, f: float, mu: float) -> float:
    x = math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(f / 2.0)
    x = max(-1.0 + 1e-15, min(1.0 - 1e-15, x))
    H = 2.0 * math.atanh(x)
    M = e * math.sinh(H) - H
    n = math.sqrt(mu / ((-a) ** 3))
    return M / n


def specific_energy(r_rel: np.ndarray, v_rel: np.ndarray, mu: float) -> float:
    return 0.5 * float(np.dot(v_rel, v_rel)) - mu / float(np.linalg.norm(r_rel))


def classify_binding(e_sun: float, e_donor: float) -> str:
    if e_sun < 0.0 and e_donor >= 0.0:
        return "sun_exchange"
    if e_sun < 0.0 and e_donor < 0.0:
        return "double_bound"
    if e_sun >= 0.0 and e_donor < 0.0:
        return "donor_bound"
    return "unbound"


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / den
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    if k == 0:
        lo = 0.0
    return lo, hi


def is_cold_exchange(binding: str, persistent: bool, e: float, inc_deg: float) -> bool:
    return binding == "sun_exchange" and persistent and e <= 0.05 and inc_deg <= 3.4


def _orientation(u_cos: float, u_node: float) -> tuple[float, float]:
    cos_i = 2.0 * u_cos - 1.0
    inc = math.acos(max(-1.0, min(1.0, cos_i)))
    node = 2.0 * math.pi * u_node
    return inc, node


def _total_L(sim) -> np.ndarray:
    L = np.zeros(3)
    for p in sim.particles:
        L += p.m * np.cross(np.array(p.xyz, dtype=float), np.array(p.vxyz, dtype=float))
    return L


def _binding_energies(sim) -> tuple[float, float]:
    donor, planet, sun = sim.particles[0], sim.particles[1], sim.particles[2]
    rp = np.array(planet.xyz); vp = np.array(planet.vxyz)
    rd = np.array(donor.xyz); vd = np.array(donor.vxyz)
    rs = np.array(sun.xyz); vs = np.array(sun.vxyz)
    e_sun = specific_energy(rp-rs, vp-vs, sim.G * (sun.m + planet.m))
    e_donor = specific_energy(rp-rd, vp-vd, sim.G * (donor.m + planet.m))
    return e_sun, e_donor


def _build_sim(q_ratio: float, v_inf_kms: float, u: np.ndarray, epsilon: float):
    import rebound

    sim = rebound.Simulation()
    sim.units = ("AU", "yr", "Msun")
    sim.integrator = "ias15"
    try:
        sim.ri_ias15.epsilon = epsilon
    except Exception:
        pass

    sim.add(m=1.0)  # donor star
    donor = sim.particles[0]

    donor_inc, donor_Omega = _orientation(float(u[0]), float(u[1]))
    M_planet = 2.0 * math.pi * float(u[2])
    sim.add(
        m=M_VENUS_MSUN,
        a=A_D_AU,
        e=0.0,
        inc=donor_inc,
        Omega=donor_Omega,
        omega=0.0,
        M=M_planet,
        primary=donor,
    )

    q_au = q_ratio * A_D_AU
    m_total = 2.0 + M_VENUS_MSUN
    a_h, e_h = hyperbola_elements(q_au, v_inf_kms, m_total)
    R0 = max(100.0, 100.0 * A_D_AU, 20.0 * q_au)
    f0 = true_anomaly_for_radius(a_h, e_h, R0, inbound=True)
    enc_inc, enc_Omega = _orientation(float(u[3]), float(u[4]))
    enc_omega = 2.0 * math.pi * float(u[5])
    sim.add(
        m=1.0,
        a=a_h,
        e=e_h,
        inc=enc_inc,
        Omega=enc_Omega,
        omega=enc_omega,
        f=f0,
        primary=donor,
    )
    sim.move_to_com()
    mu_h = sim.G * m_total
    t0 = hyperbolic_time_from_periapsis(a_h, e_h, f0, mu_h)
    encounter_duration = 2.02 * abs(t0)
    return sim, R0, encounter_duration


def _single_pass(q_ratio: float, v_inf_kms: float, u: np.ndarray, epsilon: float) -> dict:
    sim, R0, duration = _build_sim(q_ratio, v_inf_kms, u, epsilon)
    E0 = sim.energy()
    L0 = _total_L(sim)
    sim.integrate(duration, exact_finish_time=1)

    e_sun, e_donor = _binding_energies(sim)
    binding = classify_binding(e_sun, e_donor)
    persistent = False
    a_helio = e_helio = i_helio = float("nan")

    if binding == "sun_exchange":
        planet, sun = sim.particles[1], sim.particles[2]
        orb = planet.orbit(primary=sun)
        a_helio, e_helio, i_helio = float(orb.a), float(orb.e), math.degrees(float(orb.inc))
        if a_helio > 0.0 and e_helio < 1.0:
            period = math.sqrt(a_helio**3 / (sun.m + planet.m))
            sim.integrate(sim.t + 100.0 * period, exact_finish_time=1)
            e_sun2, e_donor2 = _binding_energies(sim)
            binding2 = classify_binding(e_sun2, e_donor2)
            persistent = binding2 == "sun_exchange"
            if persistent:
                orb = sim.particles[1].orbit(primary=sim.particles[2])
                a_helio, e_helio, i_helio = float(orb.a), float(orb.e), math.degrees(float(orb.inc))
                e_sun, e_donor = e_sun2, e_donor2

    Ef = sim.energy()
    Lf = _total_L(sim)
    dE = abs(Ef-E0) / max(abs(E0), 1e-300)
    dL = float(np.linalg.norm(Lf-L0)) / max(float(np.linalg.norm(L0)), 1e-300)
    valid = dE <= 1e-10 and dL <= 1e-10

    return {
        "binding": binding,
        "persistent": persistent,
        "a_helio_au": a_helio,
        "e_helio": e_helio,
        "i_helio_deg": i_helio,
        "e_sun_spec": e_sun,
        "e_donor_spec": e_donor,
        "rel_energy_error": dE,
        "rel_L_error": dL,
        "numerically_valid": valid,
        "start_radius_au": R0,
    }


def run_trial(args: tuple) -> dict:
    q_ratio, v_inf_kms, batch, idx, u = args
    try:
        out = _single_pass(q_ratio, v_inf_kms, u, 1e-12)
        retried = False
        if not out["numerically_valid"]:
            out = _single_pass(q_ratio, v_inf_kms, u, 1e-13)
            retried = True
        out.update({
            "q_over_aD": q_ratio,
            "q_au": q_ratio * A_D_AU,
            "v_inf_kms": v_inf_kms,
            "batch": batch,
            "sobol_index": idx,
            "retried": retried,
        })
        out["exchange"] = bool(out["numerically_valid"] and out["binding"] == "sun_exchange" and out["persistent"])
        out["cold_exchange"] = bool(out["numerically_valid"] and is_cold_exchange(out["binding"], out["persistent"], out["e_helio"], out["i_helio_deg"]))
        out["venus_orbit_near"] = bool(out["exchange"] and abs(out["a_helio_au"]-A_VENUS_AU)/A_VENUS_AU <= 0.10)
        out["cold_near"] = bool(out["cold_exchange"] and out["venus_orbit_near"])
        out["status"] = "ok"
        return out
    except Exception as exc:
        return {
            "q_over_aD": q_ratio, "q_au": q_ratio*A_D_AU, "v_inf_kms": v_inf_kms,
            "batch": batch, "sobol_index": idx, "status": f"error:{type(exc).__name__}:{str(exc)[:160]}",
            "numerically_valid": False, "exchange": False, "cold_exchange": False,
            "venus_orbit_near": False, "cold_near": False,
        }


def _tasks(trials_per_cell: int) -> list[tuple]:
    if trials_per_cell % BATCHES != 0:
        raise ValueError("trials_per_cell must be divisible by 4")
    per_batch = trials_per_cell // BATCHES
    if per_batch & (per_batch - 1):
        raise ValueError("trials per batch must be a power of two for Sobol random_base2")
    tasks = []
    m = int(round(math.log2(per_batch)))
    for q_ratio in Q_RATIOS:
        for vinf in VINF_KMS:
            for batch in range(BATCHES):
                seed = BASE_SEED + int(q_ratio*1000) + int(vinf*10000) + batch*7919
                sampler = qmc.Sobol(d=6, scramble=True, seed=seed)
                points = sampler.random_base2(m=m)
                for idx, u in enumerate(points):
                    tasks.append((q_ratio, vinf, batch, idx, u))
    return tasks


def _summary(df: pd.DataFrame) -> dict:
    attempted = len(df)
    valid = int(df["numerically_valid"].fillna(False).sum())
    exch = int(df["exchange"].fillna(False).sum())
    cold = int(df["cold_exchange"].fillna(False).sum())
    near = int(df["venus_orbit_near"].fillna(False).sum())
    cold_near = int(df["cold_near"].fillna(False).sum())
    cells = []
    for (q, v), g in df.groupby(["q_over_aD", "v_inf_kms"], sort=True):
        n = len(g); x = int(g["exchange"].fillna(False).sum()); c = int(g["cold_exchange"].fillna(False).sum())
        cells.append({
            "q_over_aD": float(q), "v_inf_kms": float(v), "n": n,
            "exchange": x, "exchange_wilson95": wilson_interval(x, n),
            "cold_exchange": c, "cold_wilson95": wilson_interval(c, n),
        })

    numeric_fail = attempted - valid
    if numeric_fail / attempted > 0.01:
        verdict = "NUMERICALLY_UNRESOLVED"
    elif cold > 0:
        robust = False
        for v in VINF_KMS:
            counts = {q: next((c["cold_exchange"] for c in cells if c["q_over_aD"] == q and c["v_inf_kms"] == v), 0) for q in Q_RATIOS}
            if cold >= 5 and ((counts[2.0] > 0 and counts[3.0] > 0) or (counts[3.0] > 0 and counts[4.0] > 0)):
                robust = True
        verdict = "ROBUST_COLD_OVERLAP" if robust else "COLD_OVERLAP_OBSERVED"
    elif exch > 0:
        verdict = "EXCHANGE_ONLY_HOT"
    else:
        verdict = "EXCHANGE_VANISHES_IN_OVERLAP_REGION"

    return {
        "attempted": attempted, "numerically_valid": valid, "numerical_failures": numeric_fail,
        "persistent_exchanges": exch, "exchange_wilson95": wilson_interval(exch, attempted),
        "cold_exchanges": cold, "cold_exchange_wilson95": wilson_interval(cold, attempted),
        "venus_orbit_near": near, "cold_near": cold_near,
        "verdict": verdict, "cells": cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials-per-cell", type=int, default=512)
    ap.add_argument("--workers", type=int, default=max(1, min(2, os.cpu_count() or 1)))
    ap.add_argument("--output-dir", default="venus_overlap_results")
    ns = ap.parse_args()

    tasks = _tasks(ns.trials_per_cell)
    expected = len(Q_RATIOS) * len(VINF_KMS) * ns.trials_per_cell
    if len(tasks) != expected:
        raise RuntimeError(f"task denominator mismatch: {len(tasks)} != {expected}")

    with ProcessPoolExecutor(max_workers=ns.workers) as ex:
        rows = list(ex.map(run_trial, tasks, chunksize=8))
    df = pd.DataFrame(rows)
    outdir = Path(ns.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "venus_overlap_exchange_trials.csv"
    df.to_csv(csv_path, index=False)
    summary = _summary(df)

    config = {
        "a_D_AU": A_D_AU, "q_over_aD": Q_RATIOS, "v_inf_kms": VINF_KMS,
        "batches": BATCHES, "trials_per_cell": ns.trials_per_cell,
        "cold_e_max": 0.05, "cold_i_deg_max": 3.4, "base_seed": BASE_SEED,
    }
    summary["config"] = config
    summary["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (outdir / "venus_overlap_exchange_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Venus overlap stellar-exchange pilot result", "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Attempts: {summary['attempted']}",
        f"- Numerically valid: {summary['numerically_valid']}",
        f"- Persistent exchanges: {summary['persistent_exchanges']}",
        f"- Cold exchanges (e<=0.05, i<=3.4 deg): {summary['cold_exchanges']}",
        f"- Within 10% of Venus semimajor axis: {summary['venus_orbit_near']}",
        f"- Cold + near-Venus-a: {summary['cold_near']}", "",
        "| q/aD | v_inf km/s | N | exchange | cold |",
        "|---:|---:|---:|---:|---:|",
    ]
    for c in summary["cells"]:
        lines.append(f"| {c['q_over_aD']:.0f} | {c['v_inf_kms']:.2f} | {c['n']} | {c['exchange']} | {c['cold_exchange']} |")
    (outdir / "VENUS_OVERLAP_EXCHANGE_RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
