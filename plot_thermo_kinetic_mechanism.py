"""
plot_thermo_kinetic_mechanism.py
==================================
Genera las figuras que faltaban: el mecanismo TERMODINÁMICO (ΔG' dinámico
de captación, factibilidad Big-M, incertidumbre de los ΔG'° fijos) y
CINÉTICO (saturación tipo Michaelis-Menten de la captación) del módulo
reducido -- hasta ahora solo se habían mostrado las salidas (biomasa,
sustrato), no el mecanismo termocinético en sí que da nombre al modelo.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams.update({
    'font.size': 9, 'font.family': 'serif', 'axes.linewidth': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
})

OUT = '/mnt/user-data/outputs'
labels = {'glyc': 'Glycerol', 'dha': 'DHA', 'pyr': 'Pyruvate'}
colors = {'glyc': '#E69F00', 'dha': '#56B4E9', 'pyr': '#009E73'}  # Okabe-Ito

with open('mc_thermo_triple.pkl', 'rb') as f:
    triple = pickle.load(f)
singles = {}
for c in ['glyc', 'dha', 'pyr']:
    with open(f'mc_thermo_{c}.pkl', 'rb') as f:
        singles[c] = pickle.load(f)

VARPI = 0.5  # kJ/mol, margen de disipación estricto (mismo valor del MILP)

# =========================================================================
# FIGURA C — Mecanismo TERMODINÁMICO
# Panel A: ΔG' dinámico de captación (*tex) en el escenario triple-mix,
#          mostrando cómo se vuelve menos favorable al agotarse el sustrato
# Panel B: distribución (200 muestras MC) de los ΔG'° fijos con
#          restricción Big-M dura (GLYK, DHAPT, PYK), con su significancia
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

ax = axes[0]
for c in ['glyc', 'dha', 'pyr']:
    dg_mat = np.array([run['dg_tex'][c] for run in triple['runs']])
    t = triple['time']
    dg_mean = np.nanmean(dg_mat, axis=0)
    dg_sd = np.nanstd(dg_mat, axis=0)
    valid = ~np.isnan(dg_mean)
    ax.plot(t[valid], dg_mean[valid], '-', color=colors[c], lw=1.4, label=labels[c])
    ax.fill_between(t[valid], (dg_mean - dg_sd)[valid], (dg_mean + dg_sd)[valid],
                     color=colors[c], alpha=0.2, lw=0)
ax.axhline(-VARPI, color='grey', ls='--', lw=0.8, label=r"feasibility margin $-\varpi$")
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('Time (h)')
ax.set_ylabel(r"$\Delta_c G'_t$ (kJ mol$^{-1}$), uptake step")
ax.set_title('A. Dynamic uptake driving force\n(triple-mix, mean $\\pm$ SD, n=50)',
              loc='left', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, frameon=False, loc='upper right')

ax2 = axes[1]
fixed_rxns = [('GLYK', 'glyc', -15.9, 2.08), ('DHAPT', 'dha', -39.59, 2.21),
              ('PYK', None, -25.45, 0.42)]
positions = np.arange(len(fixed_rxns))
for i, (rid, sub, mean, sd) in enumerate(fixed_rxns):
    key = f'dg_{rid}'
    samples = np.array([run['params'][key] for run in triple['runs']])
    parts = ax2.violinplot([samples], positions=[i], widths=0.7, showmeans=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#0072B2')
        pc.set_alpha(0.6)
    ax2.scatter([i], [mean], color='black', zorder=5, s=20, marker='D')
ax2.axhline(0, color='black', lw=0.5)
ax2.set_xticks(positions)
ax2.set_xticklabels([r[0] for r in fixed_rxns])
ax2.set_ylabel(r"$\Delta_r G'^{\circ}$ (kJ mol$^{-1}$)")
ax2.set_title('B. Standard-state $\\Delta G\'^\\circ$ uncertainty\n(committed steps, hard Big-$M$ constraint, n=50)',
               loc='left', fontsize=9, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_thermodynamic_mechanism.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/fig_thermodynamic_mechanism.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# =========================================================================
# FIGURA D — Mecanismo CINÉTICO
# Curva de saturación Michaelis-Menten: capacidad de transporte (cota
# superior del MILP) vs. flujo de captación realmente resuelto, en función
# de la concentración extracelular simulada -- para los 3 escenarios de
# sustrato único (sin competencia por el mismo objetivo ATP-equivalente)
# =========================================================================
# =========================================================================
# FIGURA D — Mecanismo CINÉTICO
# Curva analítica de Michaelis-Menten (con KM_MEAN/VMAX_MEAN, la media de
# los priors) como referencia limpia, con el flujo de captación REALIZADO
# en cada réplica/paso disperso alrededor de ella -- la dispersión refleja
# tanto el muestreo de KM/Vmax/eta por réplica como la modulación
# termodinámica (el flujo real cae en o por debajo de la curva cinética,
# nunca por encima).
# =========================================================================
from milp_thermokinetic_sim import KM_MEAN, VMAX_MEAN

fig, ax = plt.subplots(figsize=(6.2, 4.6))
C_ref = np.linspace(0.01, 20, 300)
for c in ['glyc', 'dha', 'pyr']:
    r = singles[c]
    C_mat = np.array([run['C'][c] for run in r['runs']])
    u_mat = np.array([run['u'][c] for run in r['runs']])

    C_flat = C_mat[:, :-1].flatten()
    u_flat = u_mat[:, :-1].flatten()

    ax.scatter(C_flat[::5], u_flat[::5], s=3, alpha=0.15, color=colors[c])

    mm_ref = VMAX_MEAN[c] * C_ref / (KM_MEAN[c] + C_ref)
    ax.plot(C_ref, mm_ref, '-', color=colors[c], lw=1.6,
            label=f'{labels[c]} ($K_M$={KM_MEAN[c]:.2g} mM, $V_{{max}}$={VMAX_MEAN[c]:.0f})')

ax.set_xlabel('Extracellular substrate concentration (mM)')
ax.set_ylabel(r'Uptake flux (mmol gDW$^{-1}$ h$^{-1}$)')
ax.set_title('C. Michaelis-Menten transport kinetics\n(line: mean-parameter reference curve; points: realized MILP flux across MC replicates)',
              loc='left', fontsize=8.5, fontweight='bold')
ax.legend(fontsize=6.5, frameon=False, loc='lower right')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_kinetic_mechanism.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/fig_kinetic_mechanism.png', dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"Listo. Figuras termodinámica y cinética en {OUT}")
