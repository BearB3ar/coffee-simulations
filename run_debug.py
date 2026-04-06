import random
import numpy as np
import base_realistic_run


# Keep setup aligned with run1.py
DOMAIN_SHAPE = [265, 265, 205]
POROSITY = 0.44
TEMPERATURE = 95
PARTICLE_SIZE_DIST = "twin_lognormal"
BREW_TIME_S = 120
POUR_RATE = 2
TIME_STEPS_120S = 120
SHRINK_FACTOR = 1
FINE_SEED = 0
NET_SEED = 0

# Long-run/asymptotic estimate settings
ASYM_BREW_TIMES = [240, 480, 960, 1440]
ASYM_MIN_DELTA = 0.005  # Absolute yield change threshold between consecutive long runs

# Calibration targets (bean-mass basis)
YIELD_120_TARGET = (0.18, 0.22)
YIELD_ASYM_TARGET = (0.36, 0.44)


def setup_and_run(k_val, c_sat_val, brew_time, time_steps):
    np.random.seed(NET_SEED)
    random.seed(NET_SEED)

    solute_cfg = {"acids": {"k": float(k_val), "concentration": 16e3, "c_sat": float(c_sat_val)}}
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
    return yield_val, extracted_mass, initial_extractable_mass


def estimate_asymptotic_yield(k_val, c_sat_val):
    yields = []
    extracted = []
    for t in ASYM_BREW_TIMES:
        y_t, ext_t, _ = setup_and_run(k_val, c_sat_val, brew_time=t, time_steps=max(TIME_STEPS_120S, int(t)))
        yields.append(y_t)
        extracted.append(ext_t)
        if len(yields) >= 2 and abs(yields[-1] - yields[-2]) < ASYM_MIN_DELTA:
            return yields[-1], extracted[-1], t, True
    return yields[-1], extracted[-1], ASYM_BREW_TIMES[-1], False


def within(v, bounds):
    return bounds[0] <= v <= bounds[1]


def evaluate_pair(k_val, c_sat_val):
    y120, ext120, init_ext = setup_and_run(k_val, c_sat_val, brew_time=BREW_TIME_S, time_steps=TIME_STEPS_120S)
    yasym, ext_asym, t_asym, plateau = estimate_asymptotic_yield(k_val, c_sat_val)

    pass120 = within(y120, YIELD_120_TARGET)
    pass_asym = within(yasym, YIELD_ASYM_TARGET)
    status = "PASS" if (pass120 and pass_asym) else "FAIL"

    print(
        f"k={k_val:.3e}, c_sat={c_sat_val:.3e}, "
        f"yield_120={y120:.2%}, yield_asym={yasym:.2%}, "
        f"extracted_120={ext120:.3e}, extracted_asym={ext_asym:.3e}, "
        f"plateau_t={t_asym}s plateau={plateau}, status={status}"
    )
    return {
        "k": k_val,
        "c_sat": c_sat_val,
        "yield_120": y120,
        "yield_asym": yasym,
        "extracted_120": ext120,
        "extracted_asym": ext_asym,
        "initial_extractable_mass": init_ext,
        "plateau_t": t_asym,
        "plateau": plateau,
        "status": status,
        "score": abs(y120 - np.mean(YIELD_120_TARGET)) + abs(yasym - np.mean(YIELD_ASYM_TARGET)),
    }


def run_sweep():
    coarse_k = [3e-9, 1e-8, 3e-8]
    coarse_csat = [2e6, 5e6, 1e7]

    print("=== Coarse sweep ===")
    coarse_results = []
    for k_val in coarse_k:
        for c_sat_val in coarse_csat:
            coarse_results.append(evaluate_pair(k_val, c_sat_val))

    coarse_results.sort(key=lambda x: x["score"])
    best = coarse_results[0]

    print("\n=== Refinement around best coarse pair ===")
    fine_k = [best["k"] * 0.7, best["k"], best["k"] * 1.3]
    fine_csat = [best["c_sat"] * 0.7, best["c_sat"], best["c_sat"] * 1.3]
    fine_results = []
    for k_val in fine_k:
        for c_sat_val in fine_csat:
            fine_results.append(evaluate_pair(k_val, c_sat_val))
    fine_results.sort(key=lambda x: x["score"])

    print("\n=== Top candidates ===")
    for r in fine_results[:5]:
        print(
            f"k={r['k']:.3e}, c_sat={r['c_sat']:.3e}, "
            f"yield_120={r['yield_120']:.2%}, yield_asym={r['yield_asym']:.2%}, "
            f"status={r['status']}"
        )


if __name__ == "__main__":
    run_sweep()

