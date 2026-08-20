"""
HPC-ready parameter sweep for the coffee extraction simulation.
Modularized for distributed PBS Array Jobs to bypass walltime and memory limits.
"""

import argparse
import csv
import gc
import math
import multiprocessing as mp
import os
import random
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
import numpy as np
import openpnm as op

import simulation_cone

matplotlib.use("Agg")

# --- Configuration (Kept identical to original) ---
DOMAIN_SHAPE = [915, 915, 793]
POROSITY = 0.44
TEMPERATURE = 92
BREW_TIME_S = 160
POUR_RATE = 4.17
TIME_STEPS = 120
SHRINK_FACTOR = 1
FINE_SEED = 0
NET_SEED = 0
FIXED_F_FAST = 0.33
SEARCH_SEED = 17

CSV_FIELDNAMES = [
    "k_fast", "k_slow", "f_fast", "c_sat", "curve_score", "stable_score",
    "head_err", "tail_err", "yield_end", "c_exit_60s", "c_exit_120s",
    "brew_time_end_s", "worker_error_type", "worker_error_message", "worker_traceback"
]

K_FAST_BOUNDS = (0.2, 100)
K_SLOW_BOUNDS = (0.003, 1.5)
F_FAST_BOUNDS = (0.15, 0.85)
C_SAT_BOUNDS = (0.1, 100.0)

COARSE_RANDOM_SAMPLES = 160
REFINE_TOP_CANDIDATES = 6
REFINE_SAMPLES_PER_CANDIDATE = 10

TARGET_EXIT_POINTS = np.array([
    [5.0, 195.0], [10.0, 150.0], [15.0, 100.0], [20.0, 60.0],
    [25.0, 40.0], [30.0, 25.0], [40.0, 20.0], [50.0, 15.0],
    [60.0, 10.0], [70.0, 8.0], [80.0, 7.0], [90.0, 6.0],
    [100.0, 5.0], [125.0, 3.0], [150.0, 1.0], [160.0, 0.0]
], dtype=float)

# --- Helper Functions ---
def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def _clamp(val, bounds):
    lo, hi = bounds
    return float(np.clip(float(val), lo, hi))

def _sample_log_uniform(rng, bounds):
    lo, hi = bounds
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

def _param_key(k_fast_val, k_slow_val, f_fast_val, c_sat_val):
    return (round(float(k_fast_val), 7), round(float(k_slow_val), 7),
            round(float(f_fast_val), 7), round(float(c_sat_val), 7))

# --- I/O Functions for Modular Workflow ---
def _save_tasks(tasks, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k_fast", "k_slow", "f_fast", "c_sat"])
        writer.writerows(tasks)
    print(f"Saved {len(tasks)} tasks to {filename}", flush=True)

def _load_tasks(filename):
    tasks = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append((float(row["k_fast"]), float(row["k_slow"]), 
                          float(row["f_fast"]), float(row["c_sat"])))
    return tasks

def _write_results_csv(results, output_csv):
    with open(output_csv, "w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in CSV_FIELDNAMES})

def _load_all_results(prefix):
    results = []
    for f in os.listdir("."):
        if f.startswith(prefix) and f.endswith(".csv"):
            with open(f, "r") as file_obj:
                reader = csv.DictReader(file_obj)
                for row in reader:
                    # Convert string values back to floats where appropriate
                    parsed_row = {}
                    for k, v in row.items():
                        try:
                            parsed_row[k] = float(v)
                        except ValueError:
                            parsed_row[k] = v
                    results.append(parsed_row)
    results.sort(key=lambda x: (x["stable_score"], x["curve_score"]))
    return results

def _get_chunk(lst, n_chunks, chunk_idx):
    """Slices a list for a specific PBS Array index (1-based)."""
    k, m = divmod(len(lst), n_chunks)
    idx = chunk_idx - 1
    start = idx * k + min(idx, m)
    end = (idx + 1) * k + min(idx + 1, m)
    return lst[start:end]

# --- Core Logic ---
def _build_coarse_candidates(rng):
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
        if k_slow_val >= 0.95 * k_fast_val: continue
        f_fast_val = float(rng.uniform(*F_FAST_BOUNDS))
        c_sat_val = _sample_log_uniform(rng, C_SAT_BOUNDS)
        key = _param_key(k_fast_val, k_slow_val, f_fast_val, c_sat_val)
        if key in seen: continue
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

def _compute_exit_concentration_curve(sim, solute="acids"):
    time_arr = np.asarray(sim.time_steps, dtype=float)
    c_exit = np.asarray(sim.exit_concentration_history_by_solute.get(solute, []), dtype=float)
    n = min(len(time_arr), len(c_exit))
    if n == 0: return np.array([]), np.array([])
    return time_arr[:n], c_exit[:n]

def setup_and_run(k_fast_val, k_slow_val, f_fast_val, c_sat_val, brew_time, time_steps):
    np.random.seed(NET_SEED)
    random.seed(NET_SEED)
    solute_cfg = {
        "acids": {
            "k_fast": float(k_fast_val), "k_slow": float(k_slow_val),
            "f_fast": float(f_fast_val), "concentration": 16e3, "c_sat": float(c_sat_val),
        }
    }
    sim = simulation_cylinder_half_bar.Simulation(
        domain_shape=DOMAIN_SHAPE, porosity=POROSITY,
        temperature=TEMPERATURE, solute_classes=solute_cfg,
    )
    sim.generate_coffee_bed()
    sim.extract_network()
    sim.add_geometry_models()
    sim.phase()
    sim.add_physics_models()
    sim.brew(brew_time=brew_time, pour_rate=POUR_RATE, time_steps=time_steps,
             shrink_factor=SHRINK_FACTOR, fines_rng_seed=FINE_SEED, store_snapshots=False)

    solute = "acids"
    initial_extractable_mass = sim.initial_extractable_mass_by_solute[solute]
    extracted_mass = sim.total_extracted_by_solute[solute]
    bean_mass = initial_extractable_mass / 0.3
    yield_val = extracted_mass / bean_mass if bean_mass > 0 else np.nan
    time_s, c_exit_kg_m3 = _compute_exit_concentration_curve(sim, solute=solute)
    
    return {
        "yield": yield_val, "extracted_mass": extracted_mass,
        "initial_extractable_mass": initial_extractable_mass,
        "time_s": time_s, "c_exit_kg_m3": c_exit_kg_m3,
    }

def evaluate_pair(k_fast_val, k_slow_val, f_fast_val, c_sat_val):
    run = setup_and_run(k_fast_val, k_slow_val, f_fast_val, c_sat_val, BREW_TIME_S, TIME_STEPS)
    op.Workspace().clear()
    gc.collect()

    sim_x = run["time_s"]
    sim_y = run["c_exit_kg_m3"]
    target_x = TARGET_EXIT_POINTS[:, 0]
    target_y = TARGET_EXIT_POINTS[:, 1]

    valid_domain = target_x <= float(np.max(sim_x)) if sim_x.size else np.zeros_like(target_x, dtype=bool)
    y_ref = target_y[valid_domain]
    finite_ref = np.isfinite(y_ref)
    
    if sim_x.size < 2 or sim_y.size < 2 or np.sum(valid_domain) < 3 or np.sum(finite_ref) < 3:
        score = np.inf; head_err = np.inf; tail_err = np.inf
    else:
        interp_vals = np.interp(target_x[valid_domain], sim_x, sim_y)
        y_ref_finite = y_ref[finite_ref]
        interp_finite = interp_vals[finite_ref]
        core_rmse = _rmse(interp_finite, y_ref_finite)
        head_err = abs(float(interp_vals[0]) - float(y_ref[0]))
        tail_err = abs(float(interp_vals[-1]) - float(y_ref[-1]))
        score = 0.5 * core_rmse + 0.3 * head_err + 0.2 * tail_err

    c60 = np.interp(60.0, sim_x, sim_y) if sim_x.size >= 2 else np.nan
    c120 = np.interp(120.0, sim_x, sim_y) if sim_x.size >= 2 else np.nan
    brew_time_end = float(np.max(sim_x)) if sim_x.size else 0.0

    if np.isfinite(run["yield"]):
        yield_penalty = max(0.0, abs(float(run["yield"]) - 0.30) - 0.03) * 1200.0
    else:
        yield_penalty = np.inf
    stable_score = float(score + yield_penalty)

    return {
        "k_fast": float(k_fast_val), "k_slow": float(k_slow_val),
        "f_fast": float(f_fast_val), "c_sat": float(c_sat_val),
        "curve_score": float(score), "stable_score": stable_score,
        "head_err": float(head_err), "tail_err": float(tail_err),
        "yield_end": float(run["yield"]),
        "c_exit_60s": float(c60) if np.isfinite(c60) else np.nan,
        "c_exit_120s": float(c120) if np.isfinite(c120) else np.nan,
        "brew_time_end_s": brew_time_end,
        "worker_error_type": "", "worker_error_message": "", "worker_traceback": "",
    }

def _worker(params):
    k_fast_val, k_slow_val, f_fast_val, c_sat_val = params
    try:
        return evaluate_pair(k_fast_val, k_slow_val, f_fast_val, c_sat_val)
    except Exception as exc:
        tb_str = traceback.format_exc()
        return {
            "k_fast": float(k_fast_val), "k_slow": float(k_slow_val),
            "f_fast": float(f_fast_val), "c_sat": float(c_sat_val),
            "curve_score": 1000, "stable_score": 1000,
            "head_err": 1000.0, "tail_err": 1000.0, "yield_end": 0.0,
            "c_exit_60s": 1000, "c_exit_120s": 1000, "brew_time_end_s": 0,
            "worker_error_type": type(exc).__name__,
            "worker_error_message": str(exc), "worker_traceback": tb_str.strip(),
        }

def _evaluate_parallel(candidates, n_workers, stage_name):
    results = []
    total = len(candidates)
    if total == 0: return results
    print(f"{stage_name}: launching {total} evaluations on {n_workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_map = {executor.submit(_worker, c): c for c in candidates}
        completed = 0
        for future in as_completed(future_map):
            results.append(future.result())
            completed += 1
            print(f"{stage_name}: completed {completed}/{total}", flush=True)
    return results

# --- Main execution modes ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["init_coarse", "run_chunk", "init_refine", "finalize"])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--task_file", type=str, default="tasks.csv")
    parser.add_argument("--output_prefix", type=str, default="results")
    parser.add_argument("--chunk_idx", type=int, default=os.environ.get("PBS_ARRAY_INDEX", 1))
    parser.add_argument("--num_chunks", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    mp.set_start_method("spawn", force=True)

    rng = np.random.default_rng(SEARCH_SEED)

    if args.mode == "init_coarse":
        coarse_candidates = _build_coarse_candidates(rng)
        seen = set()
        coarse_unique = []
        for p in coarse_candidates:
            key = _param_key(*p)
            if key not in seen:
                seen.add(key)
                coarse_unique.append(p)
        _save_tasks(coarse_unique, "coarse_tasks.csv")
        print(f"Initialization complete. Submit Job Array for coarse sweep.")

    elif args.mode == "run_chunk":
        tasks = _load_tasks(args.task_file)
        chunk_tasks = _get_chunk(tasks, args.num_chunks, int(args.chunk_idx))
        if not chunk_tasks:
            print(f"Chunk {args.chunk_idx} is empty. Exiting.")
            return
        
        results = _evaluate_parallel(chunk_tasks, args.workers, f"Chunk_{args.chunk_idx}")
        out_file = f"{args.output_prefix}_{args.chunk_idx}.csv"
        _write_results_csv(results, out_file)
        print(f"Chunk {args.chunk_idx} complete. Saved to {out_file}.")

    elif args.mode == "init_refine":
        coarse_results = _load_all_results("coarse_results_")
        if not coarse_results:
            raise RuntimeError("No coarse_results_*.csv files found.")
        
        best_candidates = coarse_results[:REFINE_TOP_CANDIDATES]
        seen = {_param_key(r["k_fast"], r["k_slow"], r["f_fast"], r["c_sat"]) for r in coarse_results}
        
        refine_candidates = []
        for best in best_candidates:
            for p in _build_refine_candidates(rng, best):
                key = _param_key(*p)
                if key not in seen:
                    seen.add(key)
                    refine_candidates.append(p)
        
        _save_tasks(refine_candidates, "refine_tasks.csv")
        print("Initialization complete. Submit Job Array for refine sweep.")

    elif args.mode == "finalize":
        all_results = _load_all_results("coarse_results_") + _load_all_results("refine_results_")
        all_results.sort(key=lambda x: (x["stable_score"], x["curve_score"]))
        _write_results_csv(all_results, "final_sweep_results.csv")
        
        print("\n=== Top stable candidates ===")
        for r in all_results[:5]:
            print(f"k_fast={r['k_fast']:.3e}, k_slow={r['k_slow']:.3e}, f_fast={r['f_fast']:.2f}, "
                  f"c_sat={r['c_sat']:.3e}, score={r['stable_score']:.3f}, yield={r['yield_end']:.2%}")

if __name__ == "__main__":
    main()