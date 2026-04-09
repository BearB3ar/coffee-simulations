import random
import numpy as np
import base_realistic_run


# Keep setup aligned with run1.py
DOMAIN_SHAPE = [300, 300, 235]
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
    sim.wall_effect(wall_porosity_boost=wall_porosity_boost, decay_width=60)
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

    return {
        "k_fast": float(k_fast_val),
        "k_slow": float(k_slow_val),
        "f_fast": float(f_fast_val),
        "c_sat": float(c_sat_val),
        "curve_score": float(score),
        "head_err": float(head_err),
        "tail_err": float(tail_err),
        "yield_end": float(run["yield"]),
        "c_100g": float(c100) if np.isfinite(c100) else np.nan,
        "c_250g": float(c250) if np.isfinite(c250) else np.nan,
        "max_brew_mass_g": float(np.max(sim_x)) if sim_x.size else 0.0,
    }


def run_sweep():
    # Reduced-size sweep with fixed f_fast to keep total evaluations <= 200.
    coarse_k_fast = [0.5, 1.5, 5.0, 10.0]
    coarse_k_slow = [0.02, 0.08, 0.2, 0.4]
    coarse_csat = [30.0, 60.0, 100.0, 160.0]

    print("=== Coarse sweep ===")
    coarse_results = []
    for k_fast_val in coarse_k_fast:
        for k_slow_val in coarse_k_slow:
            for c_sat_val in coarse_csat:
                coarse_results.append(evaluate_pair(
                    k_fast_val=k_fast_val,
                    k_slow_val=k_slow_val,
                    f_fast_val=FIXED_F_FAST,
                    c_sat_val=c_sat_val,
                ))

    coarse_results.sort(key=lambda x: x["curve_score"])
    best_candidates = coarse_results[:3]

    print("\n=== Refinement around top coarse candidates ===")
    fine_results = []
    for best in best_candidates:
        fine_k_fast = [best["k_fast"] * 0.7, best["k_fast"], best["k_fast"] * 1.3]
        fine_k_slow = [best["k_slow"] * 0.7, best["k_slow"], best["k_slow"] * 1.3]
        fine_csat = [best["c_sat"] * 0.8, best["c_sat"], best["c_sat"] * 1.2]
        for k_fast_val in fine_k_fast:
            for k_slow_val in fine_k_slow:
                for c_sat_val in fine_csat:
                    fine_results.append(evaluate_pair(
                        k_fast_val=k_fast_val,
                        k_slow_val=k_slow_val,
                        f_fast_val=FIXED_F_FAST,
                        c_sat_val=c_sat_val,
                    ))
    fine_results.sort(key=lambda x: x["curve_score"])

    total_runs = len(coarse_results) + len(fine_results)
    print(f"\nTotal evaluations: {total_runs} (max requested: 200)")

    print("\n=== Top candidates ===")
    for r in fine_results[:5]:
        print(
            f"k_fast={r['k_fast']:.3e}, k_slow={r['k_slow']:.3e}, "
            f"f_fast={r['f_fast']:.2f}, c_sat={r['c_sat']:.3e}, "
            f"curve_score={r['curve_score']:.3f}, yield_end={r['yield_end']:.2%}, "
            f"c_100g={r['c_100g']:.2f}, c_250g={r['c_250g']:.2f}"
        )


if __name__ == "__main__":
    run_sweep()

