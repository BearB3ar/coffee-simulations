import csv
import gc
import os
import random
import numpy as np
import openpnm as op
import base_realistic_run


# Keep setup aligned with run1.py
DOMAIN_SHAPE = [341, 341, 112]
POROSITY = 0.44
TEMPERATURE = 92
PARTICLE_SIZE_DIST = "twin_lognormal"
BREW_TIME_S = 240
POUR_RATE = 4
TIME_STEPS = 120
SHRINK_FACTOR = 1
FINE_SEED = 0
NET_SEED = 0
FIXED_F_FAST = 0.33
SEARCH_SEED = 17

RESULTS_CSV = "sweep_results_v3.csv"
CSV_FIELDNAMES = [
    "k_fast",
    "k_slow",
    "f_fast",
    "c_sat",
    "curve_score",
    "stable_score",
    "head_err",
    "tail_err",
    "yield_end",
    "c_100g",
    "c_250g",
    "max_brew_mass_g",
]

# Wider bounds for a more robust initial fit sweep.
K_FAST_BOUNDS = (0.2, 100)
K_SLOW_BOUNDS = (0.003, 1.5)
F_FAST_BOUNDS = (0.15, 0.85)
C_SAT_BOUNDS = (0.1, 100.0)

COARSE_RANDOM_SAMPLES = 160
REFINE_TOP_CANDIDATES = 6
REFINE_SAMPLES_PER_CANDIDATE = 10

# Approximate digitized points from the paper's red line:
# x = cumulative brewed mass (g), y = differential concentration (mg/g)
TARGET_RED_POINTS = np.array([
    [0.0, 95.0],
    [20.0, 70.0],
    [40.0, 50.0],
    [60.0, 40.0],
    [100.0, 28.0],
    [150.0, 20.0],
    [200.0, 15.0],
    [250.0, 12.0],
    [300.0, 10.0],
], dtype=float)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _clamp(val, bounds):
    lo, hi = bounds
    return float(np.clip(float(val), lo, hi))


def _sample_log_uniform(rng, bounds):
    lo, hi = bounds
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _param_key(k_fast_val, k_slow_val, f_fast_val, c_sat_val):
    return (
        round(float(k_fast_val), 7),
        round(float(k_slow_val), 7),
        round(float(f_fast_val), 7),
        round(float(c_sat_val), 7),
    )


def _append_result_csv(result):
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({key: result[key] for key in CSV_FIELDNAMES})


def _build_coarse_candidates(rng):
    # Seed from prior sweep space, but now with a few f_fast anchors.
    seeded = []
    for k_fast_val in [0.5, 1.5, 5.0, 10.0]:
        for k_slow_val in [0.02, 0.08, 0.2, 0.4]:
            for c_sat_val in [30.0, 60.0, 100.0, 160.0]:
                for f_fast_val in [0.25, FIXED_F_FAST, 0.5]:
                    if k_slow_val < k_fast_val:
                        seeded.append((k_fast_val, k_slow_val, f_fast_val, c_sat_val))

    candidates = list(seeded)
    attempts = 0
    max_attempts = COARSE_RANDOM_SAMPLES * 25
    seen = {_param_key(*p) for p in candidates}

    while len(candidates) < len(seeded) + COARSE_RANDOM_SAMPLES and attempts < max_attempts:
        attempts += 1
        k_fast_val = _sample_log_uniform(rng, K_FAST_BOUNDS)
        k_slow_val = _sample_log_uniform(rng, K_SLOW_BOUNDS)
        if k_slow_val >= 0.95 * k_fast_val:
            continue
        f_fast_val = float(rng.uniform(*F_FAST_BOUNDS))
        c_sat_val = _sample_log_uniform(rng, C_SAT_BOUNDS)
        key = _param_key(k_fast_val, k_slow_val, f_fast_val, c_sat_val)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((k_fast_val, k_slow_val, f_fast_val, c_sat_val))

    return candidates


def _build_refine_candidates(rng, best):
    candidates = []
    for _ in range(REFINE_SAMPLES_PER_CANDIDATE):
        k_fast_val = _clamp(best["k_fast"] * np.exp(rng.normal(0.0, 0.28)), K_FAST_BOUNDS)
        k_slow_val = _clamp(best["k_slow"] * np.exp(rng.normal(0.0, 0.30)), K_SLOW_BOUNDS)
        k_slow_val = min(k_slow_val, 0.90 * k_fast_val)
        k_slow_val = _clamp(k_slow_val, K_SLOW_BOUNDS)
        f_fast_val = _clamp(best["f_fast"] + rng.normal(0.0, 0.08), F_FAST_BOUNDS)
        c_sat_val = _clamp(best["c_sat"] * np.exp(rng.normal(0.0, 0.22)), C_SAT_BOUNDS)
        candidates.append((k_fast_val, k_slow_val, f_fast_val, c_sat_val))
    return candidates


def _compute_differential_curve(sim, solute="acids"):
    time_arr = np.asarray(sim.time_steps, dtype=float)
    extracted_hist = np.asarray(sim.extracted_mass_history_by_solute.get(solute, []), dtype=float)
    n = min(len(time_arr), len(extracted_hist))
    if n == 0:
        return np.array([]), np.array([])

    # Brewed water mass in grams.
    water_passed_so_far = sim.pour_rate * 1e-6 * 1e6 * time_arr[:n]
    coffee_mass = sim.initial_extractable_mass_by_solute[solute] * 1000.0 / 0.3
    retained_water_mass = 2.6 * coffee_mass
    beverage_mass_cum = np.maximum(0.0, water_passed_so_far - retained_water_mass)

    # Differential concentration c_brew = d(extracted)/d(beverage), converted to mg/g.
    d_extracted = np.diff(extracted_hist[:n], prepend=0.0)
    d_beverage = np.diff(beverage_mass_cum, prepend=0.0)
    c_diff = np.full(n, np.nan, dtype=float)
    valid = d_beverage > 0
    c_diff[valid] = 1e6 * (d_extracted[valid] / d_beverage[valid])

    valid_curve = np.isfinite(c_diff)
    return beverage_mass_cum[valid_curve], c_diff[valid_curve]


def setup_and_run(k_fast_val, k_slow_val, f_fast_val, c_sat_val, brew_time, time_steps):
    np.random.seed(NET_SEED)
    random.seed(NET_SEED)

    solute_cfg = {
        "acids": {
            "k_fast": float(k_fast_val),
            "k_slow": float(k_slow_val),
            "f_fast": float(f_fast_val),
            "concentration": 16e3,
            "c_sat": float(c_sat_val),
        }
    }
    sim = base_realistic_run.Simulation(
        domain_shape=DOMAIN_SHAPE,
        porosity=POROSITY,
        temperature=TEMPERATURE,
        particle_size_dist=PARTICLE_SIZE_DIST,
        solute_classes=solute_cfg,
    )

    wall_porosity_boost = 1.74 / (250 / (650e-6 / 1e-4) + 1.14) ** 2
    sim.generate_coffee_bed()
    #sim.wall_effect(wall_porosity_boost=wall_porosity_boost, decay_width=60)
    sim.extract_network()
    sim.add_geometry_models()
    sim.phase()
    sim.add_physics_models()
    sim.brew(
        brew_time=brew_time,
        pour_rate=POUR_RATE,
        time_steps=time_steps,
        shrink_factor=SHRINK_FACTOR,
        fines_rng_seed=FINE_SEED,
        store_snapshots=False,
    )

    solute = "acids"
    initial_extractable_mass = sim.initial_extractable_mass_by_solute[solute]
    extracted_mass = sim.total_extracted_by_solute[solute]
    bean_mass = initial_extractable_mass / 0.3
    yield_val = extracted_mass / bean_mass if bean_mass > 0 else np.nan
    brew_mass_g, c_diff_mg_g = _compute_differential_curve(sim, solute=solute)
    return {
        "yield": yield_val,
        "extracted_mass": extracted_mass,
        "initial_extractable_mass": initial_extractable_mass,
        "brew_mass_g": brew_mass_g,
        "c_diff_mg_g": c_diff_mg_g,
    }


def evaluate_pair(k_fast_val, k_slow_val, f_fast_val, c_sat_val):
    run = setup_and_run(
        k_fast_val=k_fast_val,
        k_slow_val=k_slow_val,
        f_fast_val=f_fast_val,
        c_sat_val=c_sat_val,
        brew_time=BREW_TIME_S,
        time_steps=TIME_STEPS,
    )
    op.Workspace().clear()
    gc.collect()

    sim_x = run["brew_mass_g"]
    sim_y = run["c_diff_mg_g"]
    target_x = TARGET_RED_POINTS[:, 0]
    target_y = TARGET_RED_POINTS[:, 1]

    valid_domain = target_x <= float(np.max(sim_x)) if sim_x.size else np.zeros_like(target_x, dtype=bool)
    if sim_x.size < 2 or sim_y.size < 2 or np.sum(valid_domain) < 3:
        interp_vals = np.full_like(target_y, np.nan, dtype=float)
        score = np.inf
        head_err = np.inf
        tail_err = np.inf
    else:
        interp_vals = np.interp(target_x[valid_domain], sim_x, sim_y)
        y_ref = target_y[valid_domain]
        core_rmse = _rmse(interp_vals, y_ref)
        head_err = abs(float(interp_vals[0]) - float(y_ref[0]))
        tail_err = abs(float(interp_vals[-1]) - float(y_ref[-1]))
        score = 0.5 * core_rmse + 0.3 * head_err + 0.2 * tail_err

    print(
        f"k_fast={float(k_fast_val):.3e}, k_slow={float(k_slow_val):.3e}, "
        f"f_fast={float(f_fast_val):.2f}, c_sat={float(c_sat_val):.3e}, "
        f"curve_score={score:.3f}, head_err={head_err:.3f}, tail_err={tail_err:.3f}, "
        f"yield_end={run['yield']:.2%}"
    )
    c100 = np.interp(100.0, sim_x, sim_y) if sim_x.size >= 2 else np.nan
    c250 = np.interp(250.0, sim_x, sim_y) if sim_x.size >= 2 else np.nan
    max_brew_mass = float(np.max(sim_x)) if sim_x.size else 0.0

    # Stabilize ranking by preferring runs that cover enough brewed mass
    # and land near a realistic extraction yield.
    coverage_penalty = max(0.0, 250.0 - max_brew_mass) / 25.0
    if np.isfinite(run["yield"]):
        yield_penalty = max(0.0, abs(float(run["yield"]) - 0.20) - 0.03) * 120.0
    else:
        yield_penalty = np.inf
    stable_score = float(score + coverage_penalty + yield_penalty)

    result = {
        "k_fast": float(k_fast_val),
        "k_slow": float(k_slow_val),
        "f_fast": float(f_fast_val),
        "c_sat": float(c_sat_val),
        "curve_score": float(score),
        "stable_score": stable_score,
        "head_err": float(head_err),
        "tail_err": float(tail_err),
        "yield_end": float(run["yield"]),
        "c_100g": float(c100) if np.isfinite(c100) else np.nan,
        "c_250g": float(c250) if np.isfinite(c250) else np.nan,
        "max_brew_mass_g": max_brew_mass,
    }
    _append_result_csv(result)
    return result


def run_sweep():
    rng = np.random.default_rng(SEARCH_SEED)
    coarse_candidates = _build_coarse_candidates(rng)
    print("=== Search plan ===")
    print(
        f"Stage 1: {len(coarse_candidates)} coarse points "
        "(seeded anchors + wide random search)."
    )
    print(
        f"Stage 2: refine top {REFINE_TOP_CANDIDATES} candidates with "
        f"{REFINE_SAMPLES_PER_CANDIDATE} local perturbations each."
    )
    print(
        "Bounds: "
        f"k_fast={K_FAST_BOUNDS}, k_slow={K_SLOW_BOUNDS}, "
        f"f_fast={F_FAST_BOUNDS}, c_sat={C_SAT_BOUNDS}"
    )

    print("\n=== Coarse sweep ===")
    coarse_results = []
    seen = set()
    for k_fast_val, k_slow_val, f_fast_val, c_sat_val in coarse_candidates:
        key = _param_key(k_fast_val, k_slow_val, f_fast_val, c_sat_val)
        if key in seen:
            continue
        seen.add(key)
        coarse_results.append(evaluate_pair(
            k_fast_val=k_fast_val,
            k_slow_val=k_slow_val,
            f_fast_val=f_fast_val,
            c_sat_val=c_sat_val,
        ))

    coarse_results.sort(key=lambda x: (x["stable_score"], x["curve_score"]))
    best_candidates = coarse_results[:REFINE_TOP_CANDIDATES]

    print("\n=== Refinement around top coarse candidates ===")
    fine_results = []
    for best in best_candidates:
        for k_fast_val, k_slow_val, f_fast_val, c_sat_val in _build_refine_candidates(rng, best):
            key = _param_key(k_fast_val, k_slow_val, f_fast_val, c_sat_val)
            if key in seen:
                continue
            seen.add(key)
            fine_results.append(evaluate_pair(
                k_fast_val=k_fast_val,
                k_slow_val=k_slow_val,
                f_fast_val=f_fast_val,
                c_sat_val=c_sat_val,
            ))
    fine_results.sort(key=lambda x: (x["stable_score"], x["curve_score"]))

    total_runs = len(coarse_results) + len(fine_results)
    print(f"\nTotal evaluations: {total_runs}")

    all_results = coarse_results + fine_results
    all_results.sort(key=lambda x: (x["stable_score"], x["curve_score"]))

    print("\n=== Top stable candidates ===")
    for r in all_results[:5]:
        print(
            f"k_fast={r['k_fast']:.3e}, k_slow={r['k_slow']:.3e}, "
            f"f_fast={r['f_fast']:.2f}, c_sat={r['c_sat']:.3e}, "
            f"stable_score={r['stable_score']:.3f}, curve_score={r['curve_score']:.3f}, "
            f"yield_end={r['yield_end']:.2%}, c_100g={r['c_100g']:.2f}, "
            f"c_250g={r['c_250g']:.2f}, max_brew_mass={r['max_brew_mass_g']:.1f}g"
        )

    if all_results:
        best = all_results[0]
        print("\n=== Recommended stable starting point ===")
        print(
            f"k_fast={best['k_fast']:.6g}, "
            f"k_slow={best['k_slow']:.6g}, "
            f"f_fast={best['f_fast']:.6g}, "
            f"c_sat={best['c_sat']:.6g}"
        )


if __name__ == "__main__":
    run_sweep()

