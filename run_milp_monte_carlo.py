"""
run_milp_monte_carlo.py
========================
Corre Monte Carlo (200 réplicas, MILP real por paso de tiempo) para los 7
escenarios de combinación de sustrato (glicerol/DHA/piruvato), usando
milp_thermokinetic_sim.py. Guarda los resultados en un pickle para que
las figuras se generen por separado sin tener que re-simular.
"""
import itertools
import pickle
import time

from milp_thermokinetic_sim import monte_carlo

N_REPS = 200
keys = ["glyc", "dha", "pyr"]
scenarios = [c for r in range(1, 4) for c in itertools.combinations(keys, r)]
SCENARIO_SEEDS = {combo: 1000 + i for i, combo in enumerate(scenarios)}

results = {}
t_start = time.time()
for i, combo in enumerate(scenarios):
    t0 = time.time()
    results[combo] = monte_carlo(list(combo), n_reps=N_REPS, seed=SCENARIO_SEEDS[combo])
    dt = time.time() - t0
    n_infeas = results[combo]["total_infeasible_steps"]
    print(
        f"[{i+1}/{len(scenarios)}] {'+'.join(combo):20s} "
        f"X_final={results[combo]['X_final_mean']:.4f}±{results[combo]['X_final_sd']:.4f}  "
        f"({dt:.1f}s, {n_infeas} pasos infactibles de {N_REPS*96})",
        flush=True,
    )

print(f"\nTotal: {time.time()-t_start:.1f}s")

with open("mc_results.pkl", "wb") as f:
    pickle.dump({"results": results, "scenarios": scenarios, "seeds": SCENARIO_SEEDS}, f)
print("Guardado: mc_results.pkl")
