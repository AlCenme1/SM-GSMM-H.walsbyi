"""
plot_milp_dynamics.py
======================
Genera las figuras A/B (dinámica discreta de biomasa y sustrato) a partir
de los resultados YA CALCULADOS por run_milp_monte_carlo.py (mc_results.pkl),
usando la simulación MILP real (Big-M) en vez del ODE con sigmoide ad hoc.
Mismo estilo visual que plot_discrete_dynamics.py (paleta Okabe-Ito,
muestreo cada 4h simulando cadencia experimental real).
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9, 'font.family': 'serif', 'axes.linewidth': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
})

OUT = '/mnt/user-data/outputs'
SAMPLE_EVERY_H = 4.0
DT = 0.5
SAMPLE_STRIDE = int(round(SAMPLE_EVERY_H / DT))

with open('mc_results.pkl', 'rb') as f:
    data = pickle.load(f)
mc = data['results']
scenarios = data['scenarios']

labels = {'glyc': 'Glicerol', 'dha': 'DHA', 'pyr': 'Piruvato'}
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7']
scenario_colors = OKABE_ITO
scenario_label = {c: ' + '.join(labels[k] for k in c) for c in scenarios}
markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
jitter = np.linspace(-0.6, 0.6, len(scenarios))
N_REPS = len(mc[scenarios[0]]['runs'])

# =========================================================================
# FIGURA A — crecimiento microbiano (MILP real)
# =========================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for combo, color, mk, jx in zip(scenarios, scenario_colors, markers, jitter):
    r = mc[combo]
    t_full = r['time']
    X_mat = np.array([run['X'] for run in r['runs']])

    idx = np.arange(0, len(t_full), SAMPLE_STRIDE)
    t_s = t_full[idx]
    X_s = X_mat[:, idx]
    X_mean = X_s.mean(0)
    X_sd = X_s.std(0)

    ax.plot(t_s + jx, X_mean, '-', color=color, lw=1.0, alpha=0.55, zorder=2)
    ax.errorbar(t_s + jx, X_mean, yerr=X_sd, fmt=mk, color=color, ms=4.5,
                mfc=color, mec='black', mew=0.4, elinewidth=1.0, capsize=2.0,
                label=scenario_label[combo], zorder=3)

ax.set_xlabel('Time (h)')
ax.set_ylabel(r'Biomass, $X$ (g L$^{-1}$)')
ax.set_title(
    f'A. MILP-based thermokinetic simulation — discrete sampling every {SAMPLE_EVERY_H:.0f} h '
    f'(mean $\\pm$ SD, n={N_REPS} MC reps)', loc='left', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, frameon=False, loc='upper left')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_biomass_dynamics_milp.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/fig_biomass_dynamics_milp.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# =========================================================================
# FIGURA B — dinámica de sustrato (carbono extracelular total)
# =========================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for combo, color, mk, jx in zip(scenarios, scenario_colors, markers, jitter):
    r = mc[combo]
    t_full = r['time']
    C_total_mat = np.zeros((N_REPS, len(t_full)))
    for c in combo:
        C_total_mat += np.array([run['C'][c] for run in r['runs']])

    idx = np.arange(0, len(t_full), SAMPLE_STRIDE)
    t_s = t_full[idx]
    C_s = C_total_mat[:, idx]
    C_mean = C_s.mean(0)
    C_sd = C_s.std(0)

    ax.plot(t_s + jx, C_mean, '-', color=color, lw=1.0, alpha=0.55, zorder=2)
    ax.errorbar(t_s + jx, C_mean, yerr=C_sd, fmt=mk, color=color, ms=4.5,
                mfc=color, mec='black', mew=0.4, elinewidth=1.0, capsize=2.0,
                label=scenario_label[combo], zorder=3)

ax.set_xlabel('Time (h)')
ax.set_ylabel('Total extracellular carbon, $\\sum_c C_{c,t}$ (mM)')
ax.set_title(
    f'B. MILP-based substrate depletion — discrete sampling every {SAMPLE_EVERY_H:.0f} h '
    f'(mean $\\pm$ SD, n={N_REPS} MC reps)', loc='left', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, frameon=False, loc='upper right')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_metabolite_dynamics_milp.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/fig_metabolite_dynamics_milp.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# =========================================================================
# Tabla resumen
# =========================================================================
import csv
with open(f'{OUT}/table_milp_scenario_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['scenario', 'X_final_mean', 'X_final_sd', 'n_reps', 'infeasible_steps', 'total_steps'])
    for combo in scenarios:
        r = mc[combo]
        n_steps = len(r['time']) - 1
        writer.writerow([
            '+'.join(combo), round(r['X_final_mean'], 5), round(r['X_final_sd'], 5),
            N_REPS, r['total_infeasible_steps'], N_REPS * n_steps,
        ])

print(f"Listo. Figuras y tabla en {OUT}")
