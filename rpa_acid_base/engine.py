import numpy as np
from scipy.constants import Avogadro as avogadro
import SCFT_tools
from SCFT_tools import RPA_DZ as RPA
import random
import argparse
from pathlib import Path

RESULTS_DIR = Path("results/rpa")

WATER_MODELS = {
    "single": {
        "force_field_dir": Path("force_fields/current"),
        "output_file": "OUT.txt",
        "seq": "GGAGGGGAGGGGAGGGGAGG",
        "chain_sequences": ["AGGGGAGGGGAGGGGAGGGG"],
        "species_kwargs": {"dw": 1e-5, "max_conc": 0.09},
        "species_beadtypes": [['HOH', 0.31, 0], ['ARO', 0.75, 0], ['GLU', 0.75, -1], ['TMA', 0.31, 1], ['OH', 0.31, -1]],
        "hoh_weight": 18.01528,
        "oh_weight": 17.007,
        "h3o_weight": 19.0232,
        "hoh_volume": 0.03001009271548933,
        "ion_volume": 0.03001009271548933,
        "solvent_radius": 0.31,
        "glup_charge_fully_protonated": 0,
    },
    "six-water": {
        "force_field_dir": Path("force_fields/clustered_water"),
        "output_file": "RPA_clustered_water.txt",
        "seq": "AAAAGGGGGGGGGGGGGGGG",
        "chain_sequences": ["AGAGAGAGAGAGAGAGAGAG"],
        "species_kwargs": {"dw": 1e-7, "max_conc": 0.2, "precision": 0.0005},
        "species_beadtypes": [['HOH', 0.75, 0], ['ARO', 0.75, 0], ['GLU', 0.75, -1], ['TMA', 0.75, 1], ['OH', 0.75, -1]],
        "hoh_weight": 18.01528 * 6,
        "oh_weight": 17.007 + 5 * 18.01528,
        "h3o_weight": 19.0232 + 5 * 18.01528,
        "hoh_volume": 0.03001009271548933 * 6,
        "ion_volume": 0.03001009271548933 * 6,
        "solvent_radius": 0.75,
        "glup_charge_fully_protonated": 0,
    },
}

def get_model_settings(water_model):
    return WATER_MODELS[water_model]

def normalize_water_model(value):
    model = value.strip().lower().replace("_", "-")
    aliases = {
        "single": "single",
        "single-water": "single",
        "1-water": "single",
        "six-water": "six-water",
        "6-water": "six-water",
        "six": "six-water",
        "6": "six-water",
        "clustered": "six-water",
        "clustered-water": "six-water",
    }
    if model not in aliases:
        valid = ", ".join(WATER_MODELS)
        raise ValueError(f"Unknown water_model '{value}'. Expected one of: {valid}.")
    return aliases[model]

def parse_bool(value):
    normalized = value.strip().lower()
    if normalized in ("true", "t", "yes", "y", "1", "on"):
        return True
    if normalized in ("false", "f", "no", "n", "0", "off"):
        return False
    raise ValueError(f"Could not parse boolean value '{value}'.")

def validate_chain_sequence(sequence, key_name="chain_sequence"):
    sequence = sequence.strip().upper()
    if not sequence or any(monomer not in "AG" for monomer in sequence):
        raise ValueError(f"{key_name} must contain only A and G monomers.")
    if "G" not in sequence:
        raise ValueError(f"{key_name} must contain at least one G monomer.")
    return sequence

def resolve_output_path(output_file):
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = RESULTS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def read_input_file(path):
    input_path = Path(path)
    config = {}
    if not input_path.exists():
        return config

    for line_number, raw_line in enumerate(input_path.read_text().splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"Could not parse {input_path}:{line_number}: {raw_line}")
            key, value = parts
        key = key.strip().lower()
        value = value.strip()
        if key in ("chain_sequences", "sequences"):
            sequences = [validate_chain_sequence(sequence, key) for sequence in value.split(",") if sequence.strip()]
            if not sequences:
                raise ValueError("chain_sequences must contain at least one sequence.")
            config["chain_sequences"] = sequences
        elif key == "max_conc":
            config["max_conc"] = float(value)
        elif key == "tmah_conc":
            config["tmah_conc"] = float(value)
        elif key in ("water_model", "model"):
            config["water_model"] = normalize_water_model(value)
        elif key == "output_file":
            if not value:
                raise ValueError(f"output_file cannot be empty in {input_path}:{line_number}.")
            config["output_file"] = value
        elif key == "stop_at_init_spinodal":
            config["stop_at_init_spinodal"] = parse_bool(value)
        else:
            raise ValueError(f"Unknown input key '{key}' in {input_path}:{line_number}.")
    return config

def build_run_config(input_config, water_model_override=None):
    water_model = input_config.get("water_model", "single")
    if water_model_override is not None:
        water_model = normalize_water_model(water_model_override)
    settings = get_model_settings(water_model)
    species_kwargs = dict(settings["species_kwargs"])
    if "max_conc" in input_config:
        species_kwargs["max_conc"] = input_config["max_conc"]
    if "tmah_conc" in input_config:
        species_kwargs["tmah_conc"] = input_config["tmah_conc"]
    chain_sequences = input_config.get("chain_sequences")
    if chain_sequences is None:
        chain_sequences = settings["chain_sequences"]
    output_file = input_config.get("output_file")
    return {
        "water_model": water_model,
        "settings": settings,
        "chain_sequences": chain_sequences,
        "species_kwargs": species_kwargs,
        "output_file": output_file,
        "stop_at_init_spinodal": input_config.get("stop_at_init_spinodal", False),
    }

def write_spinodal_summary(path, spinodal_rows):
    with open(path, "w") as f:
        f.write("# sequence fA polymer_weight_fraction_spinodal spinodal_type\n")
        for sequence, aromatic_fraction, polymer_wt_spinodal, spinodal_type in spinodal_rows:
            f.write(f"{sequence} {aromatic_fraction:.10f} {polymer_wt_spinodal:.10f} {int(spinodal_type)}\n")

def find_initial_spinodal(stability_data):
    for idx in range(1, len(stability_data)):
        previous_status = int(stability_data[idx - 1][-1])
        current_status = int(stability_data[idx][-1])
        if previous_status == 0 and current_status in (1, 2):
            return idx, stability_data[idx][3], current_status
    return None

def select_spinodal(stability_data, require_initial_spinodal=False):
    if require_initial_spinodal:
        initial_spinodal = find_initial_spinodal(stability_data)
        if initial_spinodal is not None:
            return initial_spinodal
    else:
        for idx in range(1, len(stability_data)):
            previous_status = int(stability_data[idx - 1][-1])
            current_status = int(stability_data[idx][-1])
            if current_status != previous_status:
                return idx, stability_data[idx][3], current_status

    if not stability_data:
        raise ValueError("No stability data were generated.")
    fallback_idx = len(stability_data) - 1
    return fallback_idx, stability_data[fallback_idx][3], int(stability_data[fallback_idx][-1])

def calc_conc(total_volume,n_molecules):
    return (n_molecules/avogadro)/(total_volume*(1/10**21)*0.001) # in units of mol/L

def calc_mol_count(total_volume,concentration): # total_volume [=] nm3, concentration [=] mol/L
    return int(round(total_volume*(1/10**21)*0.001*concentration*avogadro)) # in units of # of molecules

def get_species_counts(chain_sequence,dw=1e-6,max_conc=0.25,tmah_conc=0.0238,precision=0.001,water_model="single"):
    settings = get_model_settings(water_model)
    chain_len = len(chain_sequence)
    aro_comp_fraction = sum([1 if i == 'A' else 0 for i in chain_sequence])/len(chain_sequence)
    glu_comp_fraction = 1-aro_comp_fraction
    num_glu_per_chain = round(chain_len*glu_comp_fraction)
    chain_mol_beads = [1 if i =='A' else 2 for i in chain_sequence]
    beadtypes = settings["species_beadtypes"]
    moltypes = [['HOH',[0]], ['TMA',[3]], ['OH',[4]], ['Chain',chain_mol_beads]]
    molbonds = [['HOH',[]], ['TMA',[]], ['OH',[]], ['Chain',[[idx,idx+1] for idx in range(chain_len-1)]]]
    molweights = [['HOH',18.01528],['TMA',74.14],['OH',17.007],['Chain',aro_comp_fraction*chain_len*161.2004+(1-aro_comp_fraction)*chain_len*128.1060]] # chain is 5 ARO and 5 GLU
    #molvols = [['HOH',0.029978],['TMA',0.06753],['OH',0.029978],['Chain',0.135253*chain_len]]
    chain_vol = num_glu_per_chain*0.130306509511330 + (chain_len-num_glu_per_chain)*0.22259864959526
    molvols = [['HOH',0.03001009271548933],['TMA',0.11820332057803033],['OH',0.03001009271548933],['Chain',chain_vol]]
    comps = []
    chain_w = molweights[-1][-1]
    tot_w = chain_w/dw
    # — # deprotected monomers/ # total GLU monomers
    deprotection_ratios = list(i*(1/num_glu_per_chain) for i in range(num_glu_per_chain+1))
    if precision < dw:
        print(f"dw must be smaller than the desired precision.")
        print("Exiting program.")
        exit()
    stride = int(round(precision/dw))
    C0 = np.linspace(dw,max_conc,round(max_conc/(dw*stride)))
    DS = {}
    i = 0
    for j in range(len(C0)):
        C_Chain = C0[j]
        n_chains = round(C_Chain*tot_w/chain_w)
        tot_GLU_mono = round(chain_len*glu_comp_fraction*n_chains)
        C_SOL = 1-C_Chain
        C_W = (1-tmah_conc)*C_SOL
        n_waters = round(C_W*tot_w/18.01528)
        C_TMA = tmah_conc*C_SOL*(molweights[1][-1]/(molweights[1][-1]+molweights[2][-1]))
        n_TMA = round(C_TMA*tot_w/74.14)
        C_OH = tmah_conc*C_SOL*(molweights[2][-1]/(molweights[1][-1]+molweights[2][-1]))
        # OH- and HA (GLU) neutralization
        n_OH = round(C_OH*tot_w/17.007)
        if n_OH > tot_GLU_mono:
            n_GLUd_post_neutraliz = tot_GLU_mono
            n_GLUp_post_neutraliz = tot_GLU_mono - n_GLUd_post_neutraliz
        else:       
            n_GLUd_post_neutraliz = n_OH
            n_GLUp_post_neutraliz = tot_GLU_mono - n_GLUd_post_neutraliz
        n_OH -= n_GLUd_post_neutraliz
        n_waters += n_GLUd_post_neutraliz
        #print(f'{j} n_HOH = {n_waters}, n_OH = {n_OH}, n_TMA = {n_TMA}, tot_GLU_mono = {tot_GLU_mono}, n_GLUd = {n_GLUd_post_neutraliz}, n_GLUp = {n_GLUp_post_neutraliz}')
        
        # convert to mol/L concentrations
        total_volume = n_chains*molvols[-1][-1] + n_TMA*molvols[1][-1] + n_OH*molvols[2][-1] + n_waters*molvols[0][-1]
        C_OH0 = calc_conc(total_volume,n_OH)
        C_HOH0 = calc_conc(total_volume,n_waters)
        C_HA0 = calc_conc(total_volume,n_GLUp_post_neutraliz) # protonated GLUp
        C_A0 = calc_conc(total_volume,n_GLUd_post_neutraliz) # deprotonated GLUd
        C_TMA0 = calc_conc(total_volume,n_TMA)

        if n_OH == 0: # if strong base has been fully depleted, begin considering acid equilibrium
            #Ka = 4.71 # GLUp <-> H+ + GLUd equilibrium constant HA <-> H+ + A-
            Ka = 1.95*10**-5
            # solve quadratic equation for [H3O+]: x^2 + ([A-]0 + Ka) * x - Ka * [HA]0 = 0
            coeff = [1, C_A0+Ka, -Ka*C_HA0]
            x = np.roots(coeff)[-1]
            C_H3O = x
            C_A = C_A0 + x
            C_HA = C_HA0 - x
            C_OH = 0
            C_HOH = C_HOH0 - x
            if C_HOH < 0:
                break
        else: 
            C_H3O = 0
            C_A = C_A0
            C_HA = C_HA0
            C_OH = C_OH0
            C_HOH = C_HOH0
        C_TMA = C_TMA0
        #print(f"{j+1}. C_H3O = {C_H3O}, C_A = {C_A}, C_HA = {C_HA}, C_OH = {C_OH}, C_HOH = {C_HOH}")
        n_OH = calc_mol_count(total_volume,C_OH)
        n_HOH = calc_mol_count(total_volume,C_HOH)
        n_H3O = calc_mol_count(total_volume,C_H3O)
        n_GLUd = calc_mol_count(total_volume,C_A)
        n_GLUp = calc_mol_count(total_volume,C_HA)
        n_TMA = calc_mol_count(total_volume,C_TMA)
        n_chains = round((n_GLUd+n_GLUp)/(chain_len*glu_comp_fraction))
        #print(f"GLUd, GLUp {n_GLUd,n_GLUp}")
        # determine linear combination of polymer chain compositions that reproduce deprotonation fraction while maximizing uniformity (assumption).
        # for instance, if the overall deprotonation fraction is 0.836, we want to find a linear combination of n_chains with compositions of
        # 0.8 and 0.9 (in the case of 10 GLU residues per chain) that best reproduces this fraction.

        # Determine the lower and upper bound of deprotonation fractions that be linearly combined to overall deprotonation fraction
        has_h3o = n_H3O > 0.0
        bounds = tuple([1.0,has_h3o])
        if n_GLUd + n_GLUp > 0:
            prot_frac = (n_GLUp/(n_GLUp+n_GLUd))
            #print(1-prot_frac)
            if prot_frac > 0:
                # work in terms of deprotection
                deprot_frac = 1 - prot_frac
                # determine l, r bounds
                l, r = 0, 1
                while r < len(deprotection_ratios) and not (deprotection_ratios[l] <= deprot_frac <= deprotection_ratios[r]):
                    l, r = l+1, r+1
                lb, ub = deprotection_ratios[l], deprotection_ratios[r]
                #print((lb,ub))
                bounds = (lb,ub,has_h3o)
                # find linear combo
                a = np.array([[lb/n_chains, ub/n_chains],[1, 1]])
                b = np.array([deprot_frac,n_chains])
                x = np.linalg.solve(a,b)
                n_type1_chains, n_type2_chains = round(x[0]), round(x[1])
                #print(n_type1_chains, n_type2_chains)
        # list in a dictionary in dictionary data-structure (DS)

        # DS = {
        # (1.0 deproton. fraction, bool for H3O+) : { j : [n_chain1, nHOH, nTMA, nOH], j+1 : [n_chain1, nHOH, nTMA, nOH, nH3O], ... , j+N : [n_chain1, nHOH, nTMA, nO, nH3O]},
        # (deprot frac 1, deprot frac 2,bool for H3O+) : { j : [n_chain1, n_chain2, nHOH, nTMA, nOH, nH3O], j+1 : [n_chain1, chain2, nHOH, nTMA, nOH, nH3O], ... , j+N : [n_chain1, chain2, nHOH, nTMA, nOH, nH3O]},
        # ...
        # } 
        
            if bounds not in DS.keys():
                #print(f"adding {bounds} to D-S.")
                DS[bounds] = {}
                #print(bounds)
                
            if len(bounds) == 2:
                net_charge = -1*n_chains*chain_len*glu_comp_fraction + n_TMA - n_OH + n_H3O
                #print(net_charge)
                if net_charge < 9e-9:
                    DS[bounds][j] = [n_chains,n_HOH,n_TMA,n_OH,n_H3O]
                else:
                    print("NOT ELECTRO-NEUTRAL")
                    exit()
            else:
                lb_deprot_frac, ub_deprot_frac = bounds[0], bounds[1]
                #print(DS)
                net_charge = n_TMA - n_OH + n_H3O - n_type1_chains*lb_deprot_frac*chain_len*glu_comp_fraction - n_type2_chains*ub_deprot_frac*chain_len*glu_comp_fraction 
                #print(net_charge)
                if net_charge < 9e-9:
                    #print(f"Added {n_type1_chains+n_type2_chains} chains.")
                    DS[bounds][j] = [n_type1_chains,n_type2_chains,n_HOH,n_TMA,n_OH,n_H3O]
                else:
                    print("NOT ELECTRO-NEUTRAL")
                    exit()
        # Compute Final Weight Fractions
        total_weight = n_chains*chain_w + n_TMA*74.14 + n_OH*17.007 + n_waters*18.01528 # ensure that chain_w is weighted in way that accounts for GLU dissoc
        Cw_W = n_waters*18.01528/total_weight
        Cw_TMA = n_TMA*74.14/total_weight
        Cw_OH = n_OH*17.007/total_weight
        Cw_CHAIN = n_chains*chain_w/total_weight

    # iterate through each item in overarching DS, initialize and run RPA for each pair of deprotonation fraction chains, accumulate all the outputs at the end.
    #print(DS[(0.9375, 1.0)])
    
    for key in DS.keys():
        if len(key) == 2: # fully deprotonated case
            deprot_frac = key[0]
            #print(deprot_frac)
        else:
            deprot_frac1, deprot_frac2 = key[0], key[1]
            #print(deprot_frac1, deprot_frac2)
    return DS

# for determining protonation states
def choose_even_cuts(indices, x):
    cuts = []
    res = []
    n = len(indices)
    for k in range(1, x + 1):
        cut = (k * n) // (x + 1)
        cuts.append(cut)
    for i in range(len(indices)):
        if i in cuts:
            res.append(indices[i])
    return res


def apply_six_water_clustering(DS):
    n_h2o_per_cluster = 6
    for key in list(DS.keys()):
        if len(key) > 2: # multiple chain types
            # [n_chain1, n_chain2, n_h2o, n_tma, n_oh, n_h3o]
            for idx in list(DS[key]):
                species_arr = DS[key][idx]
                n_chain1, n_chain2, n_h2o, n_tma, n_oh, n_h3o = species_arr
                n_h2o -= (n_h2o_per_cluster-1)*n_oh
                n_h2o -= (n_h2o_per_cluster-1)*n_h3o
                n_h2o /= n_h2o_per_cluster
                DS[key][idx] = [n_chain1, n_chain2, n_h2o, n_tma, n_oh, n_h3o]
        else: # single chain type
            # [n_chain, n_h2o, n_tma, n_oh, n_h3o]
            for idx in list(DS[key]):
                species_arr = DS[key][idx]
                n_chain, n_h2o, n_tma, n_oh, n_h3o = species_arr
                n_h2o -= (n_h2o_per_cluster-1)*n_oh
                n_h2o -= (n_h2o_per_cluster-1)*n_h3o
                n_h2o /= n_h2o_per_cluster
                DS[key][idx] = [n_chain, n_h2o, n_tma, n_oh, n_h3o]


def parse_args():
    parser = argparse.ArgumentParser(description="Run RPA acid/base stability calculations.")
    parser.add_argument(
        "--input",
        default="input.in",
        help="Input file containing chain_sequences, max_conc, tmah_conc, water_model, and optional output controls.",
    )
    parser.add_argument(
        "--water-model",
        choices=WATER_MODELS.keys(),
        default=None,
        help="Override water_model from the input file.",
    )
    return parser.parse_args()


def main(input_file="input.in", water_model_override=None):
    run_config = build_run_config(read_input_file(input_file), water_model_override)
    water_model = run_config["water_model"]
    settings = run_config["settings"]
    species_kwargs = run_config["species_kwargs"]
    force_field_dir = settings["force_field_dir"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seq = settings["seq"]
    chain_sequences = run_config["chain_sequences"]
    write_detailed_stability = len(chain_sequences) == 1
    stop_at_init_spinodal = run_config["stop_at_init_spinodal"]
    requested_output_file = run_config["output_file"]
    if requested_output_file:
        output_path = resolve_output_path(requested_output_file)
    elif write_detailed_stability:
        output_path = resolve_output_path(settings["output_file"])
    else:
        output_path = resolve_output_path("initial_spinodals.txt")
    default_detailed_output_path = resolve_output_path(settings["output_file"])
    first_spindoals = []
    output_data = []
    spinodal_rows = []
    for chain_sequence in chain_sequences:
        chain_len = len(chain_sequence)
        aro_comp_fraction = sum([1 if i == 'A' else 0 for i in chain_sequence])/len(chain_sequence)
        glu_comp_fraction = 1-aro_comp_fraction
        DS = get_species_counts(chain_sequence,water_model=water_model,**species_kwargs)
        if water_model == "six-water":
            apply_six_water_clustering(DS)
        phase_transitions = {}
        stability_data = []
        RPA_OBJECTS = {}
        initial_spinodal = None
        for key, species_counts_data in DS.items():
            aro_comp_fraction = sum([1 if i == 'A' else 0 for i in chain_sequence])/len(chain_sequence)
            glu_comp_fraction = 1-aro_comp_fraction
            chain_mol_beads = [1 if i =='A' else 2 for i in chain_sequence]
            if len(key) == 2 and key[0] == 1.0: # fully deprotonated case
                deprot_frac = key[0]
                num_deprot_monos, num_prot_monos = chain_len*glu_comp_fraction*deprot_frac, chain_len*glu_comp_fraction*(1-deprot_frac)
                #if num_deprot_monos > 0: # fully protonated case; key -- 1:ARO, 2:GLUd, 3:GLUp
                #    chain_mol_beads = [1 if i =='A' else 3 for i in chain_sequence]
                res = [data_col for data_col in species_counts_data.values()] # [n_chain1, nHOH, nTMA, nOH]
                print("Running 100% deprotonated peptoid chains.")
                RPA_OBJECTS[key] = RPA.RPAsystem()
                RS = RPA_OBJECTS[key]
                RS.ff_file = str(force_field_dir / 'cg_ff_GLUd_only.dat')
                RS.out = 'ARO_GLU'
                RS.bondtype = 'offset'
                RS.r_max = 10.
                RS.n_r = 10000
                RS.k_start = 0.001
                RS.k_end = 25.0
                RS.n_k = 10000
                n = 1000
                RS.beadtypes = [['HOH', settings["solvent_radius"], 0], ['ARO', 0.75, 0], ['GLUd', 0.75, -1], ['TMA', settings["solvent_radius"], 1], ['OH', settings["solvent_radius"], -1]]
                RS.moltypes = [['HOH',[0]], ['TMA',[3]], ['OH',[4]], ['Chain',chain_mol_beads]]
                RS.molbonds = [['HOH',[]], ['TMA',[]], ['OH',[]], ['Chain',[[idx,idx+1] for idx in range(chain_len-1)]]]
                RS.molweights = [['HOH',settings["hoh_weight"]],['TMA',74.14],['OH',settings["oh_weight"]],['Chain',aro_comp_fraction*chain_len*161.2004+(1-aro_comp_fraction)*chain_len*128.1060]] # chain is 5 ARO and 5 GLU
                chain_vol = (1-aro_comp_fraction)*chain_len*0.130306509511330 + (aro_comp_fraction*chain_len)*0.22259864959526
                RS.molvols = [['HOH',settings["hoh_volume"]],['TMA',0.11820332057803033],['OH',settings["ion_volume"]],['Chain',chain_vol]]
                RS.comps = []
                for data_col in res:
                    n_chains, nHOH, nTMA, nOH, nH3O = data_col
                    total_weight = n_chains*RS.molweights[-1][-1] + nOH*RS.molweights[-2][-1] + nTMA*RS.molweights[-3][-1] + nHOH*RS.molweights[-4][-1]
                    C_CHAIN, C_OH, C_TMA, C_W, C_H3O = n_chains*RS.molweights[-1][-1]/total_weight, nOH*RS.molweights[-2][-1]/total_weight, nTMA*RS.molweights[-3][-1]/total_weight, nHOH*RS.molweights[-4][-1]/total_weight, 0
                    #if C_OH > 0:
                    RS.comps.append([C_W,C_TMA,C_OH,C_CHAIN])
                RS.Initialize()
                print("COMPS:",RS.comps)
                stability_status = RS.ComputeMatrixDeterminants()
                for k,ss in enumerate(stability_status):
                    if ss != -1:
                        comp = RS.comps[k].tolist() + [0.0] #[C_W,C_TMA,C_OH,C_CHAIN,C_H3O]
                        #rho = RS.rho_species[k].tolist()
                        stability_data.append(comp + [ss])
                        print(ss)
                    if k == 0:
                        prev_ss, curr_ss = ss, ss
                    else:
                        tmp = curr_ss
                        prev_ss,curr_ss = tmp,ss
                        print(k,prev_ss,curr_ss)
                        if prev_ss != curr_ss:
                            lb_idx, ub_idx = k-1, k
                            transition = (res[lb_idx], res[ub_idx],[prev_ss,curr_ss],comp[3])
                            if key not in phase_transitions.keys():
                                phase_transitions[key] = []
                            phase_transitions[key].append(transition)
                #print(stability_data)
            elif len(key) == 2 and key[0] == 0.0: # fully protonoated case
                deprot_frac = key[0]
                num_deprot_monos, num_prot_monos = chain_len*glu_comp_fraction*deprot_frac, chain_len*glu_comp_fraction*(1-deprot_frac)
                #if num_deprot_monos > 0: # fully protonated case; key -- 1:ARO, 2:GLUd, 3:GLUp
                chain_mol_beads = [1 if i =='A' else 2 for i in chain_sequence]
                res = [data_col for data_col in species_counts_data.values()] # [n_chain1, nHOH, nTMA, nOH]
                print("Running 100% protonated peptoid chains.")
                RPA_OBJECTS[key] = RPA.RPAsystem()
                RS = RPA_OBJECTS[key]
                RS.ff_file = str(force_field_dir / 'cg_ff_GLUd_only.dat')
                #RS.ff_file = 'force_fields/previous/cg_ff_GLUd_only_prev.dat'
                RS.out = 'ARO_GLU'
                RS.bondtype = 'offset'
                RS.r_max = 10.
                RS.n_r = 10000
                RS.k_start = 0.001
                RS.k_end = 25.0
                RS.n_k = 10000
                n = 1000
                RS.beadtypes = [['HOH', settings["solvent_radius"], 0], ['ARO', 0.75, 0], ['GLUp', 0.75, settings["glup_charge_fully_protonated"]], ['TMA', settings["solvent_radius"], 1], ['OH', settings["solvent_radius"], -1]]
                RS.moltypes = [['HOH',[0]], ['TMA',[3]], ['OH',[4]], ['Chain',chain_mol_beads]]
                RS.molbonds = [['HOH',[]], ['TMA',[]], ['OH',[]], ['Chain',[[idx,idx+1] for idx in range(chain_len-1)]]]
                RS.molweights = [['HOH',settings["hoh_weight"]],['TMA',74.14],['OH',settings["oh_weight"]],['Chain',aro_comp_fraction*chain_len*161.2004+(1-aro_comp_fraction)*chain_len*129.114]] # chain is 5 ARO and 5 GLU
                #RS.molvols = [['HOH',0.029978],['TMA',0.06753],['OH',0.029978],['Chain',0.135253*chain_len]]
                chain_vol = (1-aro_comp_fraction)*chain_len*0.130306509511330 + (aro_comp_fraction*chain_len)*0.22259864959526
                RS.molvols = [['HOH',settings["hoh_volume"]],['TMA',0.11820332057803033],['OH',settings["ion_volume"]],['Chain',chain_vol]]
                RS.comps = []
                for data_col in res:
                    n_chains, nHOH, nTMA, nOH, nH3O = data_col
                    total_weight = n_chains*RS.molweights[-1][-1] + nOH*RS.molweights[-2][-1] + nTMA*RS.molweights[-3][-1] + nHOH*RS.molweights[-4][-1]
                    C_CHAIN, C_OH, C_TMA, C_W, C_H3O = n_chains*RS.molweights[-1][-1]/total_weight, nOH*RS.molweights[-2][-1]/total_weight, nTMA*RS.molweights[-3][-1]/total_weight, nHOH*RS.molweights[-4][-1]/total_weight, 0
                    RS.comps.append([C_W,C_TMA,C_OH,C_CHAIN])
                RS.Initialize()
                print("COMPS:",RS.comps)
                stability_status = RS.ComputeMatrixDeterminants()
                for k,ss in enumerate(stability_status):
                    if ss != -1:
                        comp = RS.comps[k].tolist() + [0.0] #[C_W,C_TMA,C_OH,C_CHAIN,C_H3O]
                        rho = RS.rho_species[k].tolist()
                        stability_data.append(comp + [ss])
                    if k == 0:
                        prev_ss, curr_ss = ss, ss
                    else:
                        tmp = curr_ss
                        prev_ss,curr_ss = tmp,ss
                        print(k,prev_ss,curr_ss)
                        if prev_ss != curr_ss:
                            lb_idx, ub_idx = k-1, k
                            transition = (res[lb_idx], res[ub_idx],[prev_ss,curr_ss],comp[3])
                            if key not in phase_transitions.keys():
                                phase_transitions[key] = []
                            phase_transitions[key].append(transition)

            else: # intermediate case with two chain types
                deprot_frac1, deprot_frac2, has_h3o = key
                num_deprot_monos1, num_prot_monos1 = chain_len*glu_comp_fraction*deprot_frac1, chain_len*glu_comp_fraction*(1-deprot_frac1)
                num_deprot_monos2, num_prot_monos2 = chain_len*glu_comp_fraction*deprot_frac2, chain_len*glu_comp_fraction*(1-deprot_frac2)
                chain_mol_beads1, chain_mol_beads2 = [1 if i =='A' else 2 for i in chain_sequence], [1 if i =='A' else 2 for i in chain_sequence]
                indices_of_twos = [i for i, x in enumerate(chain_mol_beads) if x == 2]
                chosen_indices1, chosen_indices2 = random.sample(indices_of_twos, round(num_prot_monos1)), random.sample(indices_of_twos, round(num_prot_monos2))
                chosen_indices1, chosen_indices2 = choose_even_cuts(indices_of_twos,round(num_prot_monos1)), choose_even_cuts(indices_of_twos,round(num_prot_monos2))
                if len(chosen_indices1) > 0:
                    for idx in chosen_indices1:
                        chain_mol_beads1[idx] = 3
                if len(chosen_indices2) > 0:
                    for idx in chosen_indices2:
                        chain_mol_beads2[idx] = 3
                res = [data_col for data_col in species_counts_data.values()] # [n_chain1, n_chain2, nHOH, nTMA, nOH, nH3O]
                print(f"Running two chain types w/ deprots of {deprot_frac1, deprot_frac2}.")
                RPA_OBJECTS[key] = RPA.RPAsystem()
                RS = RPA_OBJECTS[key]
                RS.ff_file = str(force_field_dir / 'cg_ff_GLUd_GLUp.dat')
                RS.out = 'ARO_GLU'
                RS.bondtype = 'offset'
                RS.r_max = 10.
                RS.n_r = 10000
                RS.k_start = 0.001
                RS.k_end = 25.0
                RS.n_k = 10000
                n = 1000
                if has_h3o:
                    RS.beadtypes = [['HOH', settings["solvent_radius"], 0], ['ARO', 0.75, 0], ['GLUd', 0.75, -1],['GLUp', 0.75, 0], ['TMA', settings["solvent_radius"], 1], ['H3O', settings["solvent_radius"], 1]]
                    RS.moltypes = [['HOH',[0]], ['TMA',[4]], ['H3O',[5]], ['Chain1',chain_mol_beads1],['Chain2',chain_mol_beads2]]
                    RS.molbonds = [['HOH',[]], ['TMA',[]], ['H3O',[]], ['Chain1',[[idx,idx+1] for idx in range(chain_len-1)]],['Chain2',[[idx,idx+1] for idx in range(chain_len-1)]]]
                    RS.molweights = [['HOH',settings["hoh_weight"]],['TMA',74.14],['H3O',settings["h3o_weight"]],['Chain1',aro_comp_fraction*chain_len*161.2004+num_deprot_monos1*128.1060+num_prot_monos1*129.114],['Chain2',aro_comp_fraction*chain_len*161.2004+num_deprot_monos2*128.1060+num_prot_monos2*129.114]] # chain is 5 ARO and 5 GLU
                    #RS.molvols = [['HOH',0.029978],['TMA',0.06753],['H3O',0.029978],['Chain1',0.135253*chain_len],['Chain2',0.135253*chain_len]]
                    chain_vol = (1-aro_comp_fraction)*chain_len*0.130306509511330 + (aro_comp_fraction*chain_len)*0.22259864959526
                    RS.molvols = [['HOH',settings["hoh_volume"]],['TMA',0.11820332057803033],['H3O',settings["ion_volume"]],['Chain1',chain_vol],['Chain2',chain_vol]]
                    RS.comps = []
                    for data_col in res:
                        n_chains1, n_chains2, nHOH, nTMA, nOH, nH3O = data_col
                        total_weight = n_chains2*RS.molweights[-1][-1] + n_chains1*RS.molweights[-2][-1] + nH3O*RS.molweights[-3][-1] + nTMA*RS.molweights[1][-1] + nHOH*RS.molweights[0][-1]
                        C_CHAIN1,C_CHAIN2, C_OH, C_TMA, C_W, C_H3O = n_chains1*RS.molweights[-2][-1]/total_weight, n_chains2*RS.molweights[-1][-1]/total_weight, 0, nTMA*RS.molweights[1][-1]/total_weight, nHOH*RS.molweights[0][-1]/total_weight, nH3O*RS.molweights[-3][-1]/total_weight
                        RS.comps.append([C_W,C_TMA,C_H3O,C_CHAIN1,C_CHAIN2])
                        print([C_W,C_TMA,C_H3O,C_CHAIN1,C_CHAIN2])
                    print(f"NOW RUNNING RPA {key}")
                    RS.Initialize()
                    print("COMPS:",RS.comps)
                    stability_status = RS.ComputeMatrixDeterminants()
                    for k,ss in enumerate(stability_status):
                        if ss != -1:
                            cw, ctma, ch3o, cchain1, cchain2 = RS.comps[k].tolist() #[C_W,C_TMA,C_H3O,C_CHAIN1, C_CHAIN2] --> #[C_W,C_TMA,C_OH,C_CHAIN,C_H3O]
                            comp = [cw,ctma,0.00000000,cchain1+cchain2,ch3o]
                            rho = RS.rho_species[k].tolist()
                            stability_data.append(comp + [ss])
                        if k == 0:
                            prev_ss, curr_ss = ss, ss
                        else:
                            tmp = curr_ss
                            prev_ss,curr_ss = tmp,ss
                            #print(k,prev_ss,curr_ss)
                            if prev_ss != curr_ss:
                                lb_idx, ub_idx = k-1, k
                                transition = (res[lb_idx], res[ub_idx],[prev_ss,curr_ss],comp[3])
                                if key not in phase_transitions.keys():
                                    phase_transitions[key] = []
                                phase_transitions[key].append(transition)
                else:
                    RS.beadtypes = [['HOH', settings["solvent_radius"], 0], ['ARO', 0.75, 0], ['GLUd', 0.75, -1],['GLUp', 0.75, 0], ['TMA', settings["solvent_radius"], 1]]
                    RS.moltypes = [['HOH',[0]], ['TMA',[4]], ['Chain1',chain_mol_beads1],['Chain2',chain_mol_beads2]]
                    RS.molbonds = [['HOH',[]], ['TMA',[]], ['Chain1',[[idx,idx+1] for idx in range(chain_len-1)]],['Chain2',[[idx,idx+1] for idx in range(chain_len-1)]]]
                    RS.molweights = [['HOH',settings["hoh_weight"]],['TMA',74.14],['Chain1',aro_comp_fraction*chain_len*161.2004+num_deprot_monos1*128.1060+num_prot_monos1*129.114],['Chain2',aro_comp_fraction*chain_len*161.2004+num_deprot_monos2*128.1060+num_prot_monos2*129.114]] # chain is 5 ARO and 5 GLU
                    chain_vol = (1-aro_comp_fraction)*chain_len*0.130306509511330 + (aro_comp_fraction*chain_len)*0.22259864959526
                    RS.molvols = [['HOH',settings["hoh_volume"]],['TMA',0.11820332057803033],['Chain1',chain_vol],['Chain2',chain_vol]]
                    RS.comps = []
                    for data_col in res:
                        n_chains1, n_chains2, nHOH, nTMA, nOH, nH3O = data_col
                        total_weight = n_chains2*RS.molweights[-1][-1] + n_chains1*RS.molweights[-2][-1] + nTMA*RS.molweights[1][-1] + nHOH*RS.molweights[0][-1]
                        C_CHAIN1,C_CHAIN2, C_OH, C_TMA, C_W = n_chains1*RS.molweights[-2][-1]/total_weight, n_chains2*RS.molweights[-1][-1]/total_weight, 0, nTMA*RS.molweights[1][-1]/total_weight, nHOH*RS.molweights[0][-1]/total_weight
                        RS.comps.append([C_W,C_TMA,C_CHAIN1,C_CHAIN2])
                        print([C_W,C_TMA,C_CHAIN1,C_CHAIN2])
                    print(f"NOW RUNNING RPA {key}")
                    RS.Initialize()
                    print("COMPS:",RS.comps)
                    stability_status = RS.ComputeMatrixDeterminants()
                    for k,ss in enumerate(stability_status):
                        if ss != -1:
                            cw, ctma, cchain1, cchain2 = RS.comps[k].tolist() #[C_W,C_TMA,C_H3O,C_CHAIN1, C_CHAIN2] --> #[C_W,C_TMA,C_OH,C_CHAIN,C_H3O]
                            comp = [cw,ctma,0.00000000,cchain1+cchain2,0.00000000]
                            rho = RS.rho_species[k].tolist()
                            stability_data.append(comp + [ss])
                        if k == 0:
                            prev_ss, curr_ss = ss, ss
                        else:
                            tmp = curr_ss
                            prev_ss,curr_ss = tmp,ss
                            #print(k,prev_ss,curr_ss)
                            if prev_ss != curr_ss:
                                lb_idx, ub_idx = k-1, k
                                transition = (res[lb_idx], res[ub_idx],[prev_ss,curr_ss],comp[3])
                                if key not in phase_transitions.keys():
                                    phase_transitions[key] = []
                                phase_transitions[key].append(transition)

            if stop_at_init_spinodal:
                initial_spinodal = find_initial_spinodal(stability_data)
                if initial_spinodal is not None:
                    print(f"Initial spinodal reached for {chain_sequence}; moving to next sequence.")
                    break

        if write_detailed_stability:
            with open(output_path, "w") as f: # [C_W,C_TMA,C_OH,C_CHAIN,C_H3O]
                f.write("# weight fractions/chain concentrations : HOH   TMA   OH   Chain  H3O\n")
                f.write("# 0:stable, 1:macophase instability, 2:mesophase instability\n")
                for p in stability_data:
                    for i in p[0:-1]:
                        f.write(f"{format(i, '.10f'):12.10}")
                    f.write("{}\n".format(p[-1]))
        else:
            default_detailed_output_path.unlink(missing_ok=True)
        #print("RPA_OBJECTS:",RPA_OBJECTS)
        # LOCATE INITIAL SPINODAL
        if initial_spinodal is None:
            initial_spinodal = select_spinodal(stability_data, require_initial_spinodal=stop_at_init_spinodal)
        _, polymer_wt_spinodal, spinodal_type = initial_spinodal
        first_spindoals.append((len(chain_sequence),f"fA={aro_comp_fraction}",polymer_wt_spinodal,spinodal_type))
        output_data.append([len(chain_sequence),aro_comp_fraction,polymer_wt_spinodal,spinodal_type])
        spinodal_rows.append((chain_sequence, aro_comp_fraction, polymer_wt_spinodal, spinodal_type))
        print("FINISHED.")
    print(first_spindoals)
    if write_detailed_stability:
        output_data = np.array(output_data)
        print(output_data)
        np.savetxt(RESULTS_DIR / "blocklength_fA20.csv", output_data, delimiter=",")
    else:
        write_spinodal_summary(output_path, spinodal_rows)
        print(spinodal_rows)

def cli():
    args = parse_args()
    main(args.input, args.water_model)


if __name__ == "__main__":
    cli()
