"""
Consulta la viabilidad termodinámica de las reacciones del modelo HQ
usando eQuilibrator (component-contribution).

IMPORTANTE: este script necesita conexión a internet (descarga ~cientos
de MB de base de datos de compuestos desde Zenodo la primera vez que se
ejecuta, y la cachea localmente para corridas futuras). No se puede
ejecutar dentro del sandbox de Claude, que no tiene salida de red hacia
zenodo.org / doi.org.

Instalación:
    pip install equilibrator-api cobra pandas openpyxl

Genera dos archivos:
    HQ_equilibrator_queries.xlsx   -> TODAS las reacciones metabólicas
                                       internas Y de transporte (excluye
                                       solo los intercambios EX_). Las
                                       reacciones de transporte usan
                                       multicompartmental_standard_dg_prime
                                       (ver E_POTENTIAL_DIFFERENCE / OUTER_*
                                       más abajo) y solo reportan ΔG'°
                                       (no hay ΔG'm para transporte puro).
    HQ_glicerol_dha_piruvato.xlsx  -> subconjunto de reacciones que
                                       involucran glicerol, glicerol-3-P,
                                       DHA, DHAP o piruvato

Condiciones fisiológicas usadas por defecto (ajustables abajo):
    pH = 7.0, fuerza iónica = 0.25 M, pMg = 3.0, T = 298.15 K

NOTA IMPORTANTE sobre H. walsbyi: es un halófilo extremo que vive en
condiciones de salinidad cercanas a saturación (~5 M NaCl). Las
condiciones "estándar" de eQuilibrator (I = 0.25 M) NO representan el
citoplasma real de este organismo, que mantiene una fuerza iónica
intracelular muy alta (típicamente usando KCl como soluto compatible).
Los valores de ΔG aquí son un punto de partida; si tu paper ya trabaja
con una fuerza iónica específica para el citoplasma halofílico (como
mencionas en el modelo termocinético), cambia ionic_strength más abajo
y vuelve a correr.
"""

import cobra
import pandas as pd
from equilibrator_api import ComponentContribution, Q_

MODEL_PATH = "newHQ_final_annotated.xml"
OUTPUT_ALL = "HQ_equilibrator_queries.xlsx"
OUTPUT_MODULE = "HQ_glicerol_dha_piruvato.xlsx"

# ---- condiciones fisiológicas (ajustar si corresponde) --------------
PH = 7.0
IONIC_STRENGTH = "0.25M"   # ver nota arriba sobre halófilos extremos
PMG = 3.0
TEMPERATURE = "298.15K"

# -- condiciones del lado "externo" para reacciones de transporte -----
# (multicompartmental_standard_dg_prime evalúa el lado interno con
# PH/IONIC_STRENGTH/PMG de arriba, y el lado externo con estos otros).
# Por defecto se dejan IGUALES a las internas -- pero un halófilo
# extremo como H. walsbyi vive en salinidad cercana a saturación
# (~5 M), así que si tu paper ya tiene valores de pH/fuerza iónica
# periplasmática o extracelular medidos/asumidos, cámbialos aquí.
OUTER_PH = PH
OUTER_IONIC_STRENGTH = IONIC_STRENGTH
OUTER_PMG = PMG

# Diferencia de potencial electroestático entre el compartimento externo
# e interno (Δψ, típicamente negativo hacia adentro en procariontes,
# del orden de -100 a -150 mV, pero varía por organismo/condición). Se
# deja en 0 V por defecto (es decir, se ignora la contribución
# electroquímica) porque no tengo un valor medido para H. walsbyi --
# AJUSTAR si tienes uno de la literatura o tus propias mediciones.
E_POTENTIAL_DIFFERENCE = "0V"

# orden de compartimentos de más interno a más externo, usado para
# decidir cuál "mitad" de una reacción de transporte es la interna
COMPARTMENT_ORDER = ["C_c", "C_p", "C_e"]

# umbrales de clasificación (kJ/mol), convención habitual en literatura
# de viabilidad termodinámica de rutas metabólicas
FAVORABLE_THRESHOLD = -20.0
UNFAVORABLE_THRESHOLD = 20.0

# metabolitos "core" del módulo glicerol / DHA / piruvato (se excluyen a
# propósito los glicerolípidos de membrana -- fosfatidilglicerol, CDP-
# diacilglicerol, etc. -- que contienen "glycerol" en el nombre pero
# pertenecen a la biosíntesis de membrana, no a este módulo catabólico)
MODULE_METABOLITES = {
    "glyc_c", "glyc_e", "glyc_p",
    "glyc3p_c", "glyc3p_e", "glyc3p_p",
    "dha_c", "dhap_c",
    "pyr_c",
}

EXCLUDE_PREFIXES = ("EX_",)  # excluir reacciones de intercambio


def build_equilibrator_formula(reaction):
    """Construye la fórmula de reacción en formato eQuilibrator, usando
    el namespace bigg.metabolite (cobertura 100% en este modelo).

    IMPORTANTE: eQuilibrator indexa los compuestos SIN sufijo de
    compartimento (ej. "atp", no "atp_c") -- el compartimento no cambia
    la especie química, solo el contexto de pH/fuerza iónica en que se
    evalúa. Por eso aquí se quita el sufijo "_<compartimento>" del id
    bigg.metabolite (que sí lo incluye, correctamente, para MEMOTE)
    antes de construir la fórmula."""
    left, right = [], []
    for met, coef in reaction.metabolites.items():
        bigg_id = met.annotation.get("bigg.metabolite")
        if not bigg_id:
            return None  # no se puede construir sin id
        short_code = met.compartment.rsplit("_", 1)[-1]  # "C_c" -> "c"
        suffix = f"_{short_code}"
        base_id = bigg_id[: -len(suffix)] if bigg_id.endswith(suffix) else bigg_id
        term = f"{abs(coef)} bigg.metabolite:{base_id}"
        if coef < 0:
            left.append(term)
        else:
            right.append(term)
    if not left or not right:
        return None
    return " + ".join(left) + " = " + " + ".join(right)


def build_half_reaction(reaction, compartment):
    """Construye la 'media reacción' (formato eQuilibrator) con solo los
    metabolitos de un compartimento dado, para usar en
    multicompartmental_standard_dg_prime. Ver nota en build_equilibrator_formula
    sobre por qué se quita el sufijo de compartimento del id bigg."""
    left, right = [], []
    for met, coef in reaction.metabolites.items():
        if met.compartment != compartment:
            continue
        bigg_id = met.annotation.get("bigg.metabolite")
        if not bigg_id:
            return None
        short_code = met.compartment.rsplit("_", 1)[-1]  # "C_c" -> "c"
        suffix = f"_{short_code}"
        base_id = bigg_id[: -len(suffix)] if bigg_id.endswith(suffix) else bigg_id
        term = f"{abs(coef)} bigg.metabolite:{base_id}"
        if coef < 0:
            left.append(term)
        else:
            right.append(term)
    return " + ".join(left) + " = " + " + ".join(right)


def classify(dg_kj):
    if dg_kj is None:
        return "No calculable"
    if dg_kj <= FAVORABLE_THRESHOLD:
        return "Favorable"
    if dg_kj >= UNFAVORABLE_THRESHOLD:
        return "Desfavorable"
    return "Cerca del equilibrio"


def compute_transport_dg(cc, reaction, compartments):
    """Calcula ΔG'° de una reacción de transporte (2 compartimentos) con
    multicompartmental_standard_dg_prime, siguiendo el mismo patrón del
    ejemplo oficial de eQuilibrator (tutorial): se separa la reacción en
    una 'media reacción' interna y otra externa según el compartimento
    de cada metabolito, y se evalúa el ΔG combinado incluyendo el
    término electroquímico si hay diferencia de potencial de membrana.

    Solo maneja transporte entre EXACTAMENTE 2 compartimentos; si la
    reacción toca 3+ compartimentos a la vez, se marca como no evaluada.
    """
    known = [c for c in COMPARTMENT_ORDER if c in compartments]
    if len(known) != len(compartments) or len(compartments) != 2:
        return {
            "ΔG'° (kJ/mol)": None, "Error ΔG'° (kJ/mol)": None,
            "ΔG'm (kJ/mol, fisiológico)": None, "Error ΔG'm (kJ/mol)": None,
            "Clasificación": "No evaluado (más de 2 compartimentos o compartimento desconocido)",
        }
    inner_comp, outer_comp = known[0], known[1]  # ya ordenados de más interno a más externo

    inner_formula = build_half_reaction(reaction, inner_comp)
    outer_formula = build_half_reaction(reaction, outer_comp)
    if inner_formula is None or outer_formula is None:
        return {
            "ΔG'° (kJ/mol)": None, "Error ΔG'° (kJ/mol)": None,
            "ΔG'm (kJ/mol, fisiológico)": None, "Error ΔG'm (kJ/mol)": None,
            "Clasificación": "No calculable (falta ID de compuesto)",
        }

    try:
        rxn_inner = cc.parse_reaction_formula(inner_formula)
        rxn_outer = cc.parse_reaction_formula(outer_formula)
        dg0 = cc.multicompartmental_standard_dg_prime(
            rxn_inner,
            rxn_outer,
            e_potential_difference=Q_(E_POTENTIAL_DIFFERENCE),
            p_h_outer=Q_(OUTER_PH),
            ionic_strength_outer=Q_(OUTER_IONIC_STRENGTH),
            p_mg_outer=Q_(OUTER_PMG),
        )
        dg0_val = dg0.value.m_as("kJ/mol")
        dg0_err = dg0.error.m_as("kJ/mol")
        # el transporte puro no tiene una noción estándar de "condición
        # fisiológica" separada (esa corrección aplica a concentraciones
        # de metabolitos reactivos, no al gradiente de transporte en sí),
        # así que se reporta solo el ΔG'° multicompartimental
        return {
            "ΔG'° (kJ/mol)": round(dg0_val, 2),
            "Error ΔG'° (kJ/mol)": round(dg0_err, 2),
            "ΔG'm (kJ/mol, fisiológico)": None,
            "Error ΔG'm (kJ/mol)": None,
            "Clasificación": classify(dg0_val),
        }
    except Exception as exc:
        return {
            "ΔG'° (kJ/mol)": None, "Error ΔG'° (kJ/mol)": None,
            "ΔG'm (kJ/mol, fisiológico)": None, "Error ΔG'm (kJ/mol)": None,
            "Clasificación": f"Error: {exc}",
        }


def main():
    print("Cargando ComponentContribution (puede tardar la primera vez)...")
    cc = ComponentContribution()
    cc.p_h = Q_(PH)
    cc.ionic_strength = Q_(IONIC_STRENGTH)
    cc.p_mg = Q_(PMG)
    cc.temperature = Q_(TEMPERATURE)

    model = cobra.io.read_sbml_model(MODEL_PATH)

    rows = []
    for rxn in model.reactions:
        if rxn.id.startswith(EXCLUDE_PREFIXES):
            continue
        # reacciones de transporte puro (mismo metabolito, distinto
        # compartimento) también se excluyen de la consulta principal,
        # pero se marcan para referencia
        compartments = {m.compartment for m in rxn.metabolites}
        is_transport = len(compartments) > 1

        formula = build_equilibrator_formula(rxn)
        row = {
            "ID": rxn.id,
            "Nombre": rxn.name,
            "Ecuación": rxn.build_reaction_string(use_metabolite_names=False),
            "Tipo": "Transporte" if is_transport else "Metabólica",
            "Fórmula eQuilibrator": formula or "",
        }

        if is_transport:
            row.update(compute_transport_dg(cc, rxn, compartments))
            rows.append(row)
            continue

        if formula is None:
            row.update({
                "ΔG'° (kJ/mol)": None, "Error ΔG'° (kJ/mol)": None,
                "ΔG'm (kJ/mol, fisiológico)": None, "Error ΔG'm (kJ/mol)": None,
                "Clasificación": "No calculable (falta ID de compuesto)",
            })
            rows.append(row)
            continue

        try:
            rxn_obj = cc.parse_reaction_formula(formula)
            if not rxn_obj.is_balanced():
                row.update({
                    "ΔG'° (kJ/mol)": None, "Error ΔG'° (kJ/mol)": None,
                    "ΔG'm (kJ/mol, fisiológico)": None, "Error ΔG'm (kJ/mol)": None,
                    "Clasificación": "No calculable (desbalanceada)",
                })
                rows.append(row)
                continue

            dg0 = cc.standard_dg_prime(rxn_obj)
            dgm = cc.physiological_dg_prime(rxn_obj)

            dg0_val = dg0.value.m_as("kJ/mol")
            dg0_err = dg0.error.m_as("kJ/mol")
            dgm_val = dgm.value.m_as("kJ/mol")
            dgm_err = dgm.error.m_as("kJ/mol")

            row.update({
                "ΔG'° (kJ/mol)": round(dg0_val, 2),
                "Error ΔG'° (kJ/mol)": round(dg0_err, 2),
                "ΔG'm (kJ/mol, fisiológico)": round(dgm_val, 2),
                "Error ΔG'm (kJ/mol)": round(dgm_err, 2),
                "Clasificación": classify(dgm_val),
            })
        except Exception as exc:  # eQuilibrator lanza varios tipos de error
            row.update({
                "ΔG'° (kJ/mol)": None, "Error ΔG'° (kJ/mol)": None,
                "ΔG'm (kJ/mol, fisiológico)": None, "Error ΔG'm (kJ/mol)": None,
                "Clasificación": f"Error: {exc}",
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(OUTPUT_ALL, sheet_name="eQuilibrator", index=False)
    print(f"{len(df)} reacciones consultadas -> {OUTPUT_ALL}")

    # ---- subconjunto glicerol / DHA / piruvato -----------------------
    module_rxn_ids = {
        r.id for r in model.reactions
        if any(m.id in MODULE_METABOLITES for m in r.metabolites)
    }
    df_module = df[df["ID"].isin(module_rxn_ids)].copy()
    df_module.to_excel(OUTPUT_MODULE, sheet_name="Glicerol_DHA_Piruvato", index=False)
    print(f"{len(df_module)} reacciones del módulo glicerol/DHA/piruvato -> {OUTPUT_MODULE}")
    print("\nMetabolitos 'core' usados para definir el módulo:")
    print("  ", sorted(MODULE_METABOLITES))


if __name__ == "__main__":
    main()
