"""
HPC-ready parameter sweep for the coffee extraction simulation.

This script is a parallelized variant of `run_debug.py` intended for shared-memory
multi-core execution (single node) on HPC systems.

Environment portability notes:
1) Export from Windows/local machine:
   conda activate pnm; conda env export --from-history > environment_hpc.yml
2) Recreate on Linux HPC:
   conda env create -f environment_hpc.yml -n pnm
3) On CentOS 7 batch scripts, `source activate pnm` is often more reliable than
   `conda activate pnm` unless shell init is configured.
"""

import argparse
import csv
import gc
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
import numpy as np
import openpnm as op

import simulation_cylinder_half_bar

matplotlib.use("Agg")


# Keep setup aligned with run1.py / run_debug.py
DOMAIN_SHAPE = [562, 562, 112]
POROSITY = 0.44
TEMPERATURE = 92
BREW_TIME_S = 240
POUR_RATE = 4.17
TIME_STEPS = 120
SHRINK_FACTOR = 1
FINE_SEED = 0
NET_SEED = 0
FIXED_F_FAST = 0.33
SEARCH_SEED = 17

RESULTS_CSV = "sweep_results_hpc.csv"
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
    "c_pot_100g",
    "c_pot_250g",
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
# x = cumulative brewed mass (g), y = concentration (mg/g)
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


def _write_results_csv(results, output_csv):
    with open(output_csv, "w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for result in results:
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


def _compute_pot_concentration_curve(sim, solute="acids"):
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

    # Pot concentration = cumulative extracted / cumulative beverage, converted to mg/g.
    c_pot = np.full(n, np.nan, dtype=float)
    valid = beverage_mass_cum > 0
    c_pot[valid] = 1e6 * (extracted_hist[:n][valid] / beverage_mass_cum[valid])

    valid_curve = np.isfinite(c_pot)
    return beverage_mass_cum[valid_curve], c_pot[valid_curve]


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
    sim = simulation_cylinder_half_bar.Simulation(
        domain_shape=DOMAIN_SHAPE,
        porosity=POROSITY,
        temperature=TEMPERATURE,
        solute_classes=solute_cfg,
    )

    sim.generate_coffee_bed()
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
    brew_mass_g, c_pot_mg_g = _compute_pot_concentration_curve(sim, solute=solute)
    return {
        "yield": yield_val,
        "extracted_mass": extracted_mass,
        "initial_extractable_mass": initial_extractable_mass,
        "brew_mass_g": brew_mass_g,
        "c_pot_mg_g": c_pot_mg_g,
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
    sim_y = run["c_pot_mg_g"]
    target_x = TARGET_RED_POINTS[:, 0]
    target_y = TARGET_RED_POINTS[:, 1]

    valid_domain = target_x <= float(np.max(sim_x)) if sim_x.size else np.zeros_like(target_x, dtype=bool)
    if sim_x.size < 2 or sim_y.size < 2 or np.sum(valid_domain) < 3:
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
        f"yield_end={run['yield']:.2%}",
        flush=True,
    )
    c100 = np.interp(100.0, sim_x, sim_y) if sim_x.size >= 2 else np.nan
    c250 = np.interp(250.0, sim_x, sim_y) if sim_x.size >= 2 else np.nan
    max_brew_mass = float(np.max(sim_x)) if sim_x.size else 0.0

    # Stabilize ranking by preferring runs that land near a realistic extraction yield.
    if np.isfinite(run["yield"]):
        yield_penalty = max(0.0, abs(float(run["yield"]) - 0.20) - 0.03) * 120.0
    else:
        yield_penalty = np.inf
    stable_score = float(score + yield_penalty)

    return {
        "k_fast": float(k_fast_val),
        "k_slow": float(k_slow_val),
        "f_fast": float(f_fast_val),
        "c_sat": float(c_sat_val),
        "curve_score": float(score),
        "stable_score": stable_score,
        "head_err": float(head_err),
        "tail_err": float(tail_err),
        "yield_end": float(run["yield"]),
        "c_pot_100g": float(c100) if np.isfinite(c100) else np.nan,
        "c_pot_250g": float(c250) if np.isfinite(c250) else np.nan,
        "max_brew_mass_g": max_brew_mass,
    }


def _worker(params):
    k_fast_val, k_slow_val, f_fast_val, c_sat_val = params
    return evaluate_pair(
        k_fast_val=k_fast_val,
        k_slow_val=k_slow_val,
        f_fast_val=f_fast_val,
        c_sat_val=c_sat_val,
    )


def _evaluate_parallel(candidates, n_workers, stage_name):
    results = []
    total = len(candidates)
    if total == 0:
        return results

    print(f"{stage_name}: launching {total} evaluations on {n_workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_map = {executor.submit(_worker, c): c for c in candidates}
        completed = 0
        for future in as_completed(future_map):
            results.append(future.result())
            completed += 1
            print(f"{stage_name}: completed {completed}/{total}", flush=True)
    return results


def run_sweep(n_workers, output_csv):
    rng = np.random.default_rng(SEARCH_SEED)
    coarse_candidates = _build_coarse_candidates(rng)
    print("=== Search plan ===", flush=True)
    print(
        f"Stage 1: {len(coarse_candidates)} coarse points "
        "(seeded anchors + wide random search).",
        flush=True,
    )
    print(
        f"Stage 2: refine top {REFINE_TOP_CANDIDATES} candidates with "
        f"{REFINE_SAMPLES_PER_CANDIDATE} local perturbations each.",
        flush=True,
    )
    print(
        "Bounds: "
        f"k_fast={K_FAST_BOUNDS}, k_slow={K_SLOW_BOUNDS}, "
        f"f_fast={F_FAST_BOUNDS}, c_sat={C_SAT_BOUNDS}",
        flush=True,
    )

    # Preserve uniqueness checks from the debug script.
    seen = set()
    coarse_unique = []
    for p in coarse_candidates:
        key = _param_key(*p)
        if key in seen:
            continue
        seen.add(key)
        coarse_unique.append(p)

    print("\n=== Coarse sweep ===", flush=True)
    coarse_results = _evaluate_parallel(coarse_unique, n_workers=n_workers, stage_name="coarse")
    coarse_results.sort(key=lambda x: (x["stable_score"], x["curve_score"]))
    best_candidates = coarse_results[:REFINE_TOP_CANDIDATES]

    print("\n=== Refinement around top coarse candidates ===", flush=True)
    refine_candidates = []
    for best in best_candidates:
        for p in _build_refine_candidates(rng, best):
            key = _param_key(*p)
            if key in seen:
                continue
            seen.add(key)
            refine_candidates.append(p)

    fine_results = _evaluate_parallel(refine_candidates, n_workers=n_workers, stage_name="refine")
    fine_results.sort(key=lambda x: (x["stable_score"], x["curve_score"]))

    total_runs = len(coarse_results) + len(fine_results)
    print(f"\nTotal evaluations: {total_runs}", flush=True)

    all_results = coarse_results + fine_results
    all_results.sort(key=lambda x: (x["stable_score"], x["curve_score"]))
    _write_results_csv(all_results, output_csv=output_csv)
    print(f"Wrote {len(all_results)} rows to {output_csv}", flush=True)

    print("\n=== Top stable candidates ===", flush=True)
    for r in all_results[:5]:
        print(
            f"k_fast={r['k_fast']:.3e}, k_slow={r['k_slow']:.3e}, "
            f"f_fast={r['f_fast']:.2f}, c_sat={r['c_sat']:.3e}, "
            f"stable_score={r['stable_score']:.3f}, curve_score={r['curve_score']:.3f}, "
            f"yield_end={r['yield_end']:.2%}, c_pot_100g={r['c_pot_100g']:.2f}, "
            f"c_pot_250g={r['c_pot_250g']:.2f}, max_brew_mass={r['max_brew_mass_g']:.1f}g",
            flush=True,
        )

    if all_results:
        best = all_results[0]
        print("\n=== Recommended stable starting point ===", flush=True)
        print(
            f"k_fast={best['k_fast']:.6g}, "
            f"k_slow={best['k_slow']:.6g}, "
            f"f_fast={best['f_fast']:.6g}, "
            f"c_sat={best['c_sat']:.6g}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel HPC sweep for coffee extraction model.")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes (use physical cores, e.g. 24 on E5-2650 v4 node).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=RESULTS_CSV,
        help="CSV file path for final merged sweep results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_workers = max(1, int(args.workers))

    # One math-library thread per worker process avoids oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # Use spawn for consistency with Windows and safer behavior with threaded libs.
    mp.set_start_method("spawn", force=True)
    run_sweep(n_workers=n_workers, output_csv=args.output)


if __name__ == "__main__":
    main()

