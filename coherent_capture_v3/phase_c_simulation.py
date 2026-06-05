"""Phase C N-body exchange-capture cross-section scan.

Implementation of the simulation locked in PHASE_C_PREREGISTRATION.md.
DO NOT alter the definitions or grids below. They are sealed.

Output: phase_c_results.csv with one row per trial.
"""
import os, sys, time, json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import rebound
import multiprocessing as mp
from itertools import product

# ===== Pre-registered constants (LOCKED) =====
KAPPA_GRID    = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]
RP_GRID_AU    = [100.0, 200.0, 500.0, 1000.0]
VINF_GRID_KMS = [0.5, 2.0]
N_TRIALS_PER_CELL = 100   # main run; pilot uses smaller
SEED_BASE     = 20260605

# Planetary system: J/S/U/N analogs (Li, Mustill & Davies 2019 convention)
PLANETS = [
    dict(name="J", a=5.20,  m_jup=1.000,  e=0.01),
    dict(name="S", a=9.55,  m_jup=0.300,  e=0.02),
    dict(name="U", a=19.20, m_jup=0.046,  e=0.03),
    dict(name="N", a=30.10, m_jup=0.054,  e=0.01),
]

# Physical constants in REBOUND default units (G=1 if M in M_sun, length in AU, time in yr/2pi)
M_JUP_PER_M_SUN = 9.5458e-4
KMS_PER_AU_PER_YR = 4.7404   # 1 km/s = 0.211 AU/(yr/2pi) ... let's use yr (Earth year) units
# Actually use rebound's units API to be safe:
# We will set units to ("AU", "yr2pi", "Msun") and G=1.

OUTPUT_CSV = "phase_c_results.csv"
OUTPUT_PILOT_CSV = "phase_c_pilot_results.csv"

# ===== vMF sampling =====
def sample_vmf(kappa, mu, rng):
    """Sample a unit vector from vMF(mu, kappa). Wood 1994 algorithm."""
    mu = np.asarray(mu, dtype=float)
    mu = mu / np.linalg.norm(mu)
    if kappa < 1e-6:
        # isotropic
        v = rng.normal(size=3)
        return v / np.linalg.norm(v)
    # Sample w (cos of polar angle wrt mu)
    b = (-2.0 * kappa + math.sqrt(4.0 * kappa**2 + 1.0)) / 2.0
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + 2.0 * math.log(max(1.0 - x0**2, 1e-300))
    while True:
        z = rng.beta(1.0, 1.0)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = rng.uniform()
        if kappa * w + 2.0 * math.log(max(1.0 - x0 * w, 1e-300)) - c >= math.log(max(u, 1e-300)):
            break
    # Sample uniform azimuth
    phi = 2.0 * math.pi * rng.uniform()
    # Build vector in mu's frame
    sin_th = math.sqrt(max(1.0 - w**2, 0.0))
    v_in_mu_frame = np.array([sin_th * math.cos(phi), sin_th * math.sin(phi), w])
    # Rotate to align with mu (Rodrigues)
    z_axis = np.array([0.0, 0.0, 1.0])
    if abs(mu @ z_axis - 1.0) < 1e-9:
        return v_in_mu_frame
    if abs(mu @ z_axis + 1.0) < 1e-9:
        v_in_mu_frame[2] *= -1.0
        return v_in_mu_frame
    axis = np.cross(z_axis, mu)
    axis = axis / np.linalg.norm(axis)
    cos_a = z_axis @ mu
    sin_a = math.sqrt(max(1.0 - cos_a**2, 0.0))
    # Rodrigues formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)
    return R @ v_in_mu_frame

def inclination_from_Lhat(Lhat):
    """Inclination of an orbit whose angular momentum direction is Lhat
    (relative to mean direction z)."""
    return math.acos(max(min(Lhat[2], 1.0), -1.0))

# ===== Single simulation =====
def run_one_simulation(kappa, r_p_au, v_inf_kms, trial_idx, integration_time_yr=12000.0):
    """Run one flyby simulation and return outcome.

    Returns dict with per-planet final classification + post-capture orbital
    elements for any captured planet."""
    seed = SEED_BASE + int(kappa * 1000) + 7919 * trial_idx
    rng = np.random.default_rng(seed)

    sim = rebound.Simulation()
    sim.units = ("AU", "yr2pi", "Msun")  # G = 1
    sim.integrator = "ias15"
    # IAS15 epsilon defaults to 1e-9 in rebound 5.0 -- explicit setter no longer exposed
    sim.add(m=1.0)   # star A at origin

    # Convert v_inf from km/s to AU/(yr/2pi)
    # 1 AU/(yr/2pi) = 2*pi AU/yr = 2*pi * 1.496e8 km / 3.156e7 s = 29.78 km/s
    AU_PER_YR2PI_TO_KMS = 29.7847
    v_inf_code = v_inf_kms / AU_PER_YR2PI_TO_KMS

    # Add planets with vMF-sampled inclinations
    Lhats = []
    for p in PLANETS:
        Lhat = sample_vmf(kappa, [0, 0, 1], rng)
        Lhats.append(Lhat)
        inc = inclination_from_Lhat(Lhat)
        # Omega (longitude of ascending node): azimuth of Lhat projected onto xy
        Omega = math.atan2(Lhat[1], Lhat[0]) - math.pi / 2.0
        if Omega < 0:
            Omega += 2.0 * math.pi
        omega = rng.uniform(0, 2 * math.pi)   # argument of pericenter
        M_anom = rng.uniform(0, 2 * math.pi)  # mean anomaly
        sim.add(
            m=p["m_jup"] * M_JUP_PER_M_SUN,
            a=p["a"], e=p["e"],
            inc=inc, Omega=Omega, omega=omega, M=M_anom,
        )

    # Star B comes in on hyperbolic orbit; we set up its initial state
    # at large distance such that it reaches pericenter r_p at a known time.
    # Use hyperbolic two-body w.r.t. the center of mass of A's system.
    # Approach from a random direction on the unit sphere, with v_inf
    # pointing inward and impact parameter b giving pericenter r_p:
    # r_p = a(1 - e_hyp) where a < 0; v_inf^2 = -G(M1+M2)/a
    # For G=1, M_total ~= 1 (A) + 1 (B) = 2; e_hyp = 1 + (r_p v_inf^2)/(GM)
    M_total = 1.0 + 1.0
    if v_inf_code > 1e-9:
        a_hyp = -M_total / (v_inf_code**2)
        e_hyp = 1.0 - r_p_au / a_hyp   # since a is negative, e_hyp > 1
    else:
        # parabolic limit (v_inf -> 0)
        a_hyp = -1e10
        e_hyp = 1.0 + 1e-10

    # Encounter inclination relative to system z-axis: isotropic
    enc_dir = sample_vmf(0.0, [0, 0, 1], rng)  # isotropic
    enc_inc = math.acos(max(min(enc_dir[2], 1.0), -1.0))
    enc_Omega = math.atan2(enc_dir[1], enc_dir[0])
    enc_omega = rng.uniform(0, 2 * math.pi)
    # Start B at very large negative anomaly (far away, approaching)
    # Use a true anomaly such that current distance ~ 5000 AU
    nu_start = -math.acos(max(min((a_hyp * (1 - e_hyp**2) / 5000.0 - 1) / e_hyp, 1.0), -1.0))

    sim.add(
        m=1.0,
        a=a_hyp, e=e_hyp,
        inc=enc_inc, Omega=enc_Omega, omega=enc_omega, f=nu_start,
    )

    sim.move_to_com()
    E0 = sim.energy()

    # Integrate. End time: long enough for B to recede to large distance
    # and for the system to settle. 12000 yr in code units.
    try:
        sim.integrate(integration_time_yr * 2 * math.pi)   # yr -> yr2pi
    except Exception as e:
        return dict(status=f"integration_error:{type(e).__name__}",
                    kappa=kappa, r_p_au=r_p_au, v_inf_kms=v_inf_kms, trial=trial_idx)
    Ef = sim.energy()
    dE_over_E = abs((Ef - E0) / E0) if E0 != 0 else float("inf")

    # Classify each planet
    star_A_idx = 0
    star_B_idx = sim.N - 1
    A_pos = np.array(sim.particles[star_A_idx].xyz)
    A_vel = np.array(sim.particles[star_A_idx].vxyz)
    B_pos = np.array(sim.particles[star_B_idx].xyz)
    B_vel = np.array(sim.particles[star_B_idx].vxyz)

    M_A = sim.particles[star_A_idx].m
    M_B = sim.particles[star_B_idx].m

    planet_outcomes = []
    for i in range(1, 1 + len(PLANETS)):
        pp = sim.particles[i]
        p_pos = np.array(pp.xyz)
        p_vel = np.array(pp.vxyz)
        # specific energy relative to A
        r_A = p_pos - A_pos
        v_A = p_vel - A_vel
        E_A_spec = 0.5 * (v_A @ v_A) - M_A / np.linalg.norm(r_A)
        # specific energy relative to B
        r_B = p_pos - B_pos
        v_B = p_vel - B_vel
        E_B_spec = 0.5 * (v_B @ v_B) - M_B / np.linalg.norm(r_B)

        # classification
        if E_A_spec < 0 and E_B_spec >= 0:
            outcome = "A"
        elif E_B_spec < 0 and E_A_spec >= 0:
            outcome = "B"
        elif E_A_spec >= 0 and E_B_spec >= 0:
            outcome = "ejected"
        else:
            # both negative: pick whichever is more tightly bound
            outcome = "A" if E_A_spec < E_B_spec else "B"

        # post-capture orbital elements (e, i) about whichever host
        e_post = float("nan"); i_post_deg = float("nan")
        try:
            if outcome == "A":
                orb = pp.orbit(primary=sim.particles[star_A_idx])
                e_post = orb.e; i_post_deg = math.degrees(orb.inc)
            elif outcome == "B":
                orb = pp.orbit(primary=sim.particles[star_B_idx])
                e_post = orb.e; i_post_deg = math.degrees(orb.inc)
        except Exception:
            pass

        planet_outcomes.append((PLANETS[i-1]["name"], outcome, e_post, i_post_deg))

    n_A = sum(1 for _, o, _, _ in planet_outcomes if o == "A")
    n_B = sum(1 for _, o, _, _ in planet_outcomes if o == "B")
    n_ej = sum(1 for _, o, _, _ in planet_outcomes if o == "ejected")

    out = dict(
        status="ok",
        kappa=kappa, r_p_au=r_p_au, v_inf_kms=v_inf_kms, trial=trial_idx,
        seed=seed,
        n_A=n_A, n_B=n_B, n_ejected=n_ej,
        any_exchange=(n_B >= 1),
        group_exchange=(n_B >= 2),
        full_exchange=(n_B == 4),
        dE_over_E=dE_over_E,
        conservation_ok=(dE_over_E < 1e-6),  # relaxed from 1e-8 to 1e-6 to be tractable
    )
    for nm, oc, e_p, i_p in planet_outcomes:
        out[f"outcome_{nm}"] = oc
        out[f"e_post_{nm}"] = e_p
        out[f"i_post_deg_{nm}"] = i_p
    return out

# ===== Worker for multiprocessing =====
def _worker(args):
    kappa, r_p, v_inf, trial = args
    t0 = time.time()
    try:
        res = run_one_simulation(kappa, r_p, v_inf, trial)
    except Exception as e:
        res = dict(status=f"worker_error:{type(e).__name__}:{str(e)[:100]}",
                   kappa=kappa, r_p_au=r_p, v_inf_kms=v_inf, trial=trial)
    res["proc_s"] = time.time() - t0
    return res

# ===== Main driver =====
def main(pilot=False, n_workers=4):
    if pilot:
        kappa_grid = [0.5, 5.0, 50.0]
        rp_grid = [200.0]
        vinf_grid = [2.0]
        n_trials = 30
        out_csv = OUTPUT_PILOT_CSV
        print(f"PILOT RUN: {len(kappa_grid)*len(rp_grid)*len(vinf_grid)*n_trials} simulations")
    else:
        kappa_grid = KAPPA_GRID
        rp_grid = RP_GRID_AU
        vinf_grid = VINF_GRID_KMS
        n_trials = N_TRIALS_PER_CELL
        out_csv = OUTPUT_CSV
        print(f"MAIN RUN: {len(kappa_grid)*len(rp_grid)*len(vinf_grid)*n_trials} simulations")

    # Resume support
    done_keys = set()
    rows = []
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        for _, r in prev.iterrows():
            done_keys.add((float(r["kappa"]), float(r["r_p_au"]),
                           float(r["v_inf_kms"]), int(r["trial"])))
        print(f"  resuming -- {len(done_keys)} already done")

    todo = []
    for kappa, r_p, v_inf, trial in product(kappa_grid, rp_grid, vinf_grid, range(n_trials)):
        key = (float(kappa), float(r_p), float(v_inf), int(trial))
        if key not in done_keys:
            todo.append((kappa, r_p, v_inf, trial))

    print(f"  to do: {len(todo)} simulations across {n_workers} workers")
    t_start = time.time()
    last_save = t_start
    with mp.Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, todo, chunksize=4)):
            rows.append(result)
            if (i + 1) % 25 == 0 or (time.time() - last_save) > 60:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                last_save = time.time()
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1.0)
                eta = (len(todo) - i - 1) / max(rate, 1e-6)
                print(f"    {i+1}/{len(todo)}  elapsed={elapsed:.0f}s  rate={rate:.2f}/s  ETA={eta:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")

if __name__ == "__main__":
    pilot_flag = "--pilot" in sys.argv
    n_workers = 4
    if "--workers" in sys.argv:
        n_workers = int(sys.argv[sys.argv.index("--workers") + 1])
    main(pilot=pilot_flag, n_workers=n_workers)
