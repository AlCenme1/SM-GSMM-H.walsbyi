"""
milp_thermokinetic_sim.py
==========================
"""

import json
import math
import numpy as np
import pulp

MODULE_JSON = "reduced_module_glyc_dha_pyr.json"

with open(MODULE_JSON) as f:
    _MOD = json.load(f)

REACTIONS = _MOD["reactions"]
INTERNAL_METS = _MOD["internal_metabolites"]
TEX_OF = _MOD["carbon_source_tex_reaction"]  # {'glyc':'GLYCtex', ...}

SUBSTRATES = ["glyc", "dha", "pyr"]

R_GAS = 8.314e-3  # kJ/(mol K)
T = 310.15        # K
RT = R_GAS * T
VARPI = 0.5        # kJ/mol, 
C_IN_ASSUMED = 0.01  # mM, 
W_PYR = 12.5        # ATP


KM_MEAN = {"glyc": 1.4, "dha": 0.35, "pyr": 0.6}
VMAX_MEAN = {"glyc": 164.0, "dha": 120.0, "pyr": 140.0}
KIN_CV = 0.25

K_MAX = 1.2
TAU_X = 0.004
NU_MAX = 0.15
ALPHA = 0.1
TAU_C = 0.002
C0 = 20.0
DT = 0.5
T_HORIZON = 48.0
X0 = 0.001


_FIXED_DG_RIDS = [
    rid for rid, rd in REACTIONS.items()
    if rd["dg_std"] is not None and rd.get("dynamic_dg_substrate") is None
]


def sample_params(rng, active_sources):
    
    dg_sampled = {}
    for rid in _FIXED_DG_RIDS:
        rd = REACTIONS[rid]
        err = rd["dg_error"] if rd["dg_error"] else 0.1 * abs(rd["dg_std"])
        dg_sampled[rid] = rng.normal(rd["dg_std"], err)

    KM = {}
    Vmax = {}
    for c in SUBSTRATES:
        KM[c] = rng.lognormal(np.log(KM_MEAN[c]), KIN_CV)
        Vmax[c] = rng.lognormal(np.log(VMAX_MEAN[c]), KIN_CV)

    eta_draws = []
    for i in (1, 2, 3):
        mu_ln = (i - 1)
        sd_ln = 0.1 / i
        eta_draws.append(rng.lognormal(mu_ln, sd_ln))
    eta = np.mean(eta_draws)
    eta_norm = eta / np.mean([np.exp((i - 1) + 0.5 * (0.1 / i) ** 2) for i in (1, 2, 3)])

    return dg_sampled, KM, Vmax, eta_norm


def solve_milp_step(active_sources, C_now, dg_sampled, KM, Vmax, eta):
    
    prob = pulp.LpProblem("step", pulp.LpMaximize)
    v = {}
    for rid, rd in REACTIONS.items():
        lb, ub = rd["bounds"]
        v[rid] = pulp.LpVariable(f"v_{rid}", lowBound=lb, upBound=ub)

    caps = {}
    for c in SUBSTRATES:
        tex_id = TEX_OF[c]
        if c in active_sources and C_now[c] > 0:
            cap = eta * Vmax[c] * C_now[c] / (KM[c] + C_now[c])
            v[tex_id].lowBound = 0
            v[tex_id].upBound = cap
            caps[c] = cap
        else:
            v[tex_id].lowBound = 0
            v[tex_id].upBound = 0
            caps[c] = 0.0

    for met in INTERNAL_METS:
        expr = pulp.lpSum(rd["stoich"].get(met, 0) * v[rid] for rid, rd in REACTIONS.items())
        prob += (expr == 0), f"mb_{met}"

    for rid, rd in REACTIONS.items():
        dg = dg_sampled.get(rid, rd["dg_std"])
        sub = rd.get("dynamic_dg_substrate")
        if sub is not None:
            if sub in active_sources and C_now[sub] > 0:
                c_out = max(C_now[sub], 1e-9)
                dg = rd["dg_std"] + RT * math.log(C_IN_ASSUMED / c_out)
            else:
                dg = None
        if dg is None:
            continue
        M = rd["big_m"]
        y = pulp.LpVariable(f"y_{rid}", cat="Binary")
        lb, ub = v[rid].lowBound, v[rid].upBound
        prob += v[rid] >= lb * (1 - y)
        prob += v[rid] <= ub * y
        prob += dg <= -VARPI + M * (1 - y)
        prob += dg >= VARPI - M * y

    prob += v["DM_atp_c"] + W_PYR * v["DM_pyr_c"]
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status_str = pulp.LpStatus[status]

    if status_str != "Optimal":
        return {c: 0.0 for c in SUBSTRATES}, 0.0, status_str

    uptake = {c: (v[TEX_OF[c]].value() or 0.0) for c in SUBSTRATES}
    atp_eq = pulp.value(prob.objective) or 0.0
    return uptake, atp_eq, status_str


def simulate(active_sources, rng, n_steps=None):
    if n_steps is None:
        n_steps = int(T_HORIZON / DT)
    dg_sampled, KM, Vmax, eta = sample_params(rng, active_sources)

    X = np.zeros(n_steps + 1)
    C = {c: np.zeros(n_steps + 1) for c in SUBSTRATES}
    u = {c: np.zeros(n_steps + 1) for c in SUBSTRATES}
    ATP = np.zeros(n_steps + 1)
    nu = np.zeros(n_steps + 1)
    n_infeasible = 0

    X[0] = X0
    for c in SUBSTRATES:
        C[c][0] = C0 if c in active_sources else 0.0

    for t in range(n_steps):
        C_now = {c: C[c][t] for c in SUBSTRATES}
        uptake, atp_eq, status = solve_milp_step(active_sources, C_now, dg_sampled, KM, Vmax, eta)
        if status != "Optimal":
            n_infeasible += 1
        for c in SUBSTRATES:
            u[c][t] = uptake[c]
        ATP[t] = atp_eq
        nu_t = min(NU_MAX, ALPHA * atp_eq)
        nu[t] = nu_t

        X[t + 1] = X[t] + nu_t * (1 - X[t] / K_MAX) * X[t] * DT - TAU_X * X[t] * DT
        X[t + 1] = max(X[t + 1], 0.0)
        for c in SUBSTRATES:
            C[c][t + 1] = (
                C[c][t] - u[c][t] * X[t] * DT
                + TAU_C * ((C0 if c in active_sources else 0.0) - C[c][t]) * DT
            )
            C[c][t + 1] = max(C[c][t + 1], 0.0)

    return {
        "time": np.arange(n_steps + 1) * DT,
        "X": X, "C": C, "u": u, "ATP": ATP, "nu": nu,
        "n_infeasible_steps": n_infeasible,
        "params": {
            **{f"dg_{k}": v_ for k, v_ in dg_sampled.items()},
            **{f"KM_{c}": KM[c] for c in SUBSTRATES},
            **{f"Vmax_{c}": Vmax[c] for c in SUBSTRATES},
            "eta": eta,
        },
    }


def monte_carlo(active_sources, n_reps=200, seed=42):
    rng = np.random.default_rng(seed)
    runs = [simulate(active_sources, rng) for _ in range(n_reps)]
    time = runs[0]["time"]
    X_mat = np.array([r["X"] for r in runs])
    total_infeasible = sum(r["n_infeasible_steps"] for r in runs)
    return {
        "time": time,
        "X_mean": X_mat.mean(axis=0),
        "X_lo": np.percentile(X_mat, 2.5, axis=0),
        "X_hi": np.percentile(X_mat, 97.5, axis=0),
        "X_final_mean": X_mat[:, -1].mean(),
        "X_final_sd": X_mat[:, -1].std(),
        "total_infeasible_steps": total_infeasible,
        "runs": runs,
    }
