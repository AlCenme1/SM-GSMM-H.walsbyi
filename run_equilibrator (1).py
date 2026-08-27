
import cobra
import pandas as pd
from equilibrator_api import ComponentContribution, Q_

MODEL_PATH = "iHQW-TK.xml"
OUTPUT_ALL = "HQ_equilibrator_queries.xlsx"
OUTPUT_MODULE = "HQ_equilibrator_glycerol_dha_pyruvate.xlsx"


PH = 7.0
IONIC_STRENGTH = "4.5M"   
PMG = 3.0
TEMPERATURE = "310.15K"


OUTER_PH = PH
OUTER_IONIC_STRENGTH = IONIC_STRENGTH
OUTER_PMG = PMG


E_POTENTIAL_DIFFERENCE = "0V"

#
COMPARTMENT_ORDER = ["C_c", "C_p", "C_e"]


FAVORABLE_THRESHOLD = -20.0
UNFAVORABLE_THRESHOLD = 20.0

#  module glycerol / DHA / piruvato 
MODULE_METABOLITES = {
    "glyc_c", "glyc_e", "glyc_p",
    "glyc3p_c", "glyc3p_e", "glyc3p_p",
    "dha_c", "dhap_c",
    "pyr_c",
}

EXCLUDE_PREFIXES = ("EX_",)  


def build_equilibrator_formula(reaction):
    
    left, right = [], []
    for met, coef in reaction.metabolites.items():
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
    if not left or not right:
        return None
    return " + ".join(left) + " = " + " + ".join(right)


def build_half_reaction(reaction, compartment):
    
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
    print("loading")
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
        
        compartments = {m.compartment for m in rxn.metabolites}
        is_transport = len(compartments) > 1

        formula = build_equilibrator_formula(rxn)
        row = {
            "ID": rxn.id,
            "Name": rxn.name,
            "Equation": rxn.build_reaction_string(use_metabolite_names=False),
            "Type": "Transporte" if is_transport else "Metabolic",
            "Formula eQuilibrator": formula or "",
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
