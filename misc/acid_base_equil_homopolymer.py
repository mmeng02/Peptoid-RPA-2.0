import numpy as np
from scipy.constants import Avogadro as avogadro
import SCFT_tools
from SCFT_tools import RPA_DZ as RPA
import random
from pathlib import Path

def calc_conc(total_volume,n_molecules):
    return (n_molecules/avogadro)/(total_volume*(1/10**21)*0.001) # in units of mol/L

def calc_mol_count(total_volume,concentration): # total_volume [=] nm3, concentration [=] mol/L
    return int(round(total_volume*(1/10**21)*0.001*concentration*avogadro)) # in units of # of molecules

def get_species_counts(chain_sequence,desired_dw=0.0001,max_conc=0.99,tmah_conc=0.0238):
    chain_len = len(chain_sequence)
    aro_comp_fraction = sum([1 if i == 'A' else 0 for i in chain_sequence])/len(chain_sequence)
    glu_comp_fraction = 1-aro_comp_fraction
    num_glu_per_chain = round(chain_len*glu_comp_fraction)
    chain_mol_beads = [1 if i =='A' else 2 for i in chain_sequence]
    beadtypes = [['HOH', 0.31, 0], ['ARO', 0.75, 0], ['GLU', 0.75, -1], ['TMA', 0.31, 1], ['OH', 0.31, -1]]
    moltypes = [['HOH',[0]], ['TMA',[3]], ['OH',[4]], ['Chain',chain_mol_beads]]
    molbonds = [['HOH',[]], ['TMA',[]], ['OH',[]], ['Chain',[[idx,idx+1] for idx in range(chain_len-1)]]]
    molweights = [['HOH',18.01528],['TMA',74.14],['OH',17.007],['Chain',aro_comp_fraction*chain_len*161.2004+(1-aro_comp_fraction)*chain_len*128.1060]] # chain is 5 ARO and 5 GLU
    #molvols = [['HOH',0.029978],['TMA',0.06753],['OH',0.029978],['Chain',0.135253*chain_len]]
    chain_vol = num_glu_per_chain*0.130306509511330 + (chain_len-num_glu_per_chain)*0.22259864959526
    molvols = [['HOH',0.03001009271548933],['TMA',0.11820332057803033],['OH',0.03001009271548933],['Chain',chain_vol]]
    comps = []
    chain_w = molweights[-1][-1]
    tot_w = chain_w/desired_dw
    # — # deprotected monomers/ # total GLU monomers
    deprotection_ratios = list(i*(1/num_glu_per_chain) for i in range(num_glu_per_chain+1))
    C0 = np.linspace(desired_dw,max_conc,round(max_conc/desired_dw))
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
                print(bounds)
            if len(bounds) == 2:
                net_charge = -1*n_chains*chain_len*glu_comp_fraction + n_TMA - n_OH + n_H3O
                #print("Net Charge", net_charge, [n_chains,n_HOH,n_TMA,n_OH,n_H3O])
                if net_charge < 1e-10:
                    DS[bounds][j] = [n_chains,n_HOH,n_TMA,n_OH,n_H3O]
                else:
                    print("NOT ELECTRO-NEUTRAL")
                    exit()
            else:
                lb_deprot_frac, ub_deprot_frac = bounds[0], bounds[1]
                #print(DS)
                net_charge = n_TMA - n_OH + n_H3O - n_type1_chains*lb_deprot_frac*chain_len*glu_comp_fraction - n_type2_chains*ub_deprot_frac*chain_len*glu_comp_fraction 
                #print("Net Charge", net_charge, [n_type1_chains,n_type2_chains,n_HOH,n_TMA,n_OH,n_H3O])
                if net_charge < 1e-10:
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
            print(deprot_frac)
        else:
            deprot_frac1, deprot_frac2 = key[0], key[1]
            print(deprot_frac1, deprot_frac2)
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

#RPA_OBJECTS = {}

def main():
    Path("results/rpa").mkdir(parents=True, exist_ok=True)
    chain_sequences = [str("A"*i)+str("G"*(30-i)) for i in range(1,30)]
    chain_sequences = ["G"*50]
    first_spindoals = []
    for chain_sequence in chain_sequences:
        is_homopolymer_A, is_homopolymer_G = False, False
        if 'A' not in chain_sequence: # homopolymer case
            is_homopolymer_G = True
        elif 'G' not in chain_sequence:
            is_homopolymer_A = True
        chain_len = len(chain_sequence)
        aro_comp_fraction = sum([1 if i == 'A' else 0 for i in chain_sequence])/len(chain_sequence)
        glu_comp_fraction = 1-aro_comp_fraction
        if is_homopolymer_G:
            DS = get_species_counts(chain_sequence)
            print(DS)
        phase_transitions = {}
        stability_data = []
        RPA_OBJECTS = {}
        if is_homopolymer_G:
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
                    RS.ff_file = './force_fields/current/cg_ff_GLUd_only.dat'
                    RS.out = 'ARO_GLU'
                    RS.bondtype = 'offset'
                    RS.r_max = 10.
                    RS.n_r = 10000
                    RS.k_start = 0.0005
                    RS.k_end = 25.0
                    RS.n_k = 10000
                    n = 1000
                    RS.beadtypes = [['HOH', 0.31, 0], ['GLUd', 0.75, -1], ['TMA', 0.31, 1], ['OH', 0.31, -1]]
                    RS.moltypes = [['HOH',[0]], ['TMA',[2]], ['OH',[3]], ['Chain',[1 for i in range(len(chain_sequence))]]]
                    RS.molbonds = [['HOH',[]], ['TMA',[]], ['OH',[]], ['Chain',[[idx,idx+1] for idx in range(chain_len-1)]]]
                    RS.molweights = [['HOH',18.01528],['TMA',74.14],['OH',17.007],['Chain',aro_comp_fraction*chain_len*161.2004+(1-aro_comp_fraction)*chain_len*128.1060]] # chain is 5 ARO and 5 GLU
                    chain_vol = (1-aro_comp_fraction)*chain_len*0.130306509511330 + (aro_comp_fraction*chain_len)*0.22259864959526
                    RS.molvols = [['HOH',0.03001009271548933],['TMA',0.11820332057803033],['OH',0.03001009271548933],['Chain',chain_vol]]
                    RS.comps = []
                    for data_col in res:
                        n_chains, nHOH, nTMA, nOH, nH3O = data_col
                        total_weight = n_chains*RS.molweights[-1][-1] + nOH*RS.molweights[-2][-1] + nTMA*RS.molweights[-3][-1] + nHOH*RS.molweights[-4][-1]
                        C_CHAIN, C_OH, C_TMA, C_W, C_H3O = n_chains*RS.molweights[-1][-1]/total_weight, nOH*RS.molweights[-2][-1]/total_weight, nTMA*RS.molweights[-3][-1]/total_weight, nHOH*RS.molweights[-4][-1]/total_weight, 0
                        #if C_OH > 0:
                        RS.comps.append([C_W,C_TMA,C_OH,C_CHAIN])
                    RS.Initialize()
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
                    RS.ff_file = './force_fields/current/cg_ff_GLUd_only.dat'
                    #RS.ff_file = 'force_fields/previous/cg_ff_GLUd_only_prev.dat'
                    RS.out = 'ARO_GLU'
                    RS.bondtype = 'offset'
                    RS.r_max = 10.
                    RS.n_r = 10000
                    RS.k_start = 0.0005
                    RS.k_end = 1.0
                    RS.n_k = 10000
                    n = 1000
                    RS.beadtypes = [['HOH', 0.31, 0], ['GLUp', 0.75, -1], ['TMA', 0.31, 1], ['OH', 0.31, -1]]
                    RS.moltypes = [['HOH',[0]], ['TMA',[3]], ['OH',[4]], ['Chain',chain_mol_beads]]
                    RS.molbonds = [['HOH',[]], ['TMA',[]], ['OH',[]], ['Chain',[[idx,idx+1] for idx in range(chain_len-1)]]]
                    RS.molweights = [['HOH',18.01528],['TMA',74.14],['OH',17.007],['Chain',aro_comp_fraction*chain_len*161.2004+(1-aro_comp_fraction)*chain_len*129.114]] # chain is 5 ARO and 5 GLU
                    #RS.molvols = [['HOH',0.029978],['TMA',0.06753],['OH',0.029978],['Chain',0.135253*chain_len]]
                    chain_vol = (1-aro_comp_fraction)*chain_len*0.130306509511330 + (aro_comp_fraction*chain_len)*0.22259864959526
                    RS.molvols = [['HOH',0.03001009271548933],['TMA',0.11820332057803033],['OH',0.03001009271548933],['Chain',chain_vol]]
                    RS.comps = []
                    for data_col in res:
                        n_chains, nHOH, nTMA, nOH, nH3O = data_col
                        total_weight = n_chains*RS.molweights[-1][-1] + nOH*RS.molweights[-2][-1] + nTMA*RS.molweights[-3][-1] + nHOH*RS.molweights[-4][-1]
                        C_CHAIN, C_OH, C_TMA, C_W, C_H3O = n_chains*RS.molweights[-1][-1]/total_weight, nOH*RS.molweights[-2][-1]/total_weight, nTMA*RS.molweights[-3][-1]/total_weight, nHOH*RS.molweights[-4][-1]/total_weight, 0
                        RS.comps.append([C_W,C_TMA,C_OH,C_CHAIN])
                    RS.Initialize()
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
                    chain_mol_beads1, chain_mol_beads2 = [1 for i in range(len(chain_sequence))], [1 for i in range(len(chain_sequence))]
                    indices_of_twos = [i for i, x in enumerate(chain_mol_beads1) if x == 1]
                    chosen_indices1, chosen_indices2 = choose_even_cuts(indices_of_twos,round(num_prot_monos1)), choose_even_cuts(indices_of_twos,round(num_prot_monos2))
                    if len(chosen_indices1) > 0:
                        for idx in chosen_indices1:
                            chain_mol_beads1[idx] = 2
                    if len(chosen_indices2) > 0:
                        for idx in chosen_indices2:
                            chain_mol_beads2[idx] = 2
                    res = [data_col for data_col in species_counts_data.values()] # [n_chain1, n_chain2, nHOH, nTMA, nOH, nH3O]
                    print(f"Running two chain types w/ deprots of {deprot_frac1, deprot_frac2}.")
                    RPA_OBJECTS[key] = RPA.RPAsystem()
                    RS = RPA_OBJECTS[key]
                    RS.ff_file = './force_fields/current/cg_ff_GLUd_GLUp.dat'
                    RS.out = 'ARO_GLU'
                    RS.bondtype = 'offset'
                    RS.r_max = 10.
                    RS.n_r = 10000
                    RS.k_start = 0.0005
                    RS.k_end = 1.0
                    RS.n_k = 10000
                    n = 1000
                    if has_h3o:
                        RS.beadtypes = [['HOH', 0.31, 0], ['GLUd', 0.75, -1],['GLUp', 0.75, 0], ['TMA', 0.31, 1], ['H3O', 0.31, 1]]
                        RS.moltypes = [['HOH',[0]], ['TMA',[3]], ['H3O',[4]], ['Chain1',chain_mol_beads1],['Chain2',chain_mol_beads2]]
                        RS.molbonds = [['HOH',[]], ['TMA',[]], ['H3O',[]], ['Chain1',[[idx,idx+1] for idx in range(chain_len-1)]],['Chain2',[[idx,idx+1] for idx in range(chain_len-1)]]]
                        RS.molweights = [['HOH',18.01528],['TMA',74.14],['H3O',19.0232],['Chain1',aro_comp_fraction*chain_len*161.2004+num_deprot_monos1*128.1060+num_prot_monos1*129.114],['Chain2',aro_comp_fraction*chain_len*161.2004+num_deprot_monos2*128.1060+num_prot_monos2*129.114]] # chain is 5 ARO and 5 GLU
                        #RS.molvols = [['HOH',0.029978],['TMA',0.06753],['H3O',0.029978],['Chain1',0.135253*chain_len],['Chain2',0.135253*chain_len]]
                        chain_vol = (1-aro_comp_fraction)*chain_len*0.130306509511330 + (aro_comp_fraction*chain_len)*0.22259864959526
                        RS.molvols = [['HOH',0.03001009271548933],['TMA',0.11820332057803033],['H3O',0.03001009271548933],['Chain1',chain_vol],['Chain2',chain_vol]]
                        RS.comps = []
                        for data_col in res:
                            n_chains1, n_chains2, nHOH, nTMA, nOH, nH3O = data_col
                            total_weight = n_chains2*RS.molweights[-1][-1] + n_chains1*RS.molweights[-2][-1] + nH3O*RS.molweights[-3][-1] + nTMA*RS.molweights[1][-1] + nHOH*RS.molweights[0][-1]
                            C_CHAIN1,C_CHAIN2, C_OH, C_TMA, C_W, C_H3O = n_chains1*RS.molweights[-2][-1]/total_weight, n_chains2*RS.molweights[-1][-1]/total_weight, 0, nTMA*RS.molweights[1][-1]/total_weight, nHOH*RS.molweights[0][-1]/total_weight, nH3O*RS.molweights[-3][-1]/total_weight
                            RS.comps.append([C_W,C_TMA,C_H3O,C_CHAIN1,C_CHAIN2])
                            print([C_W,C_TMA,C_H3O,C_CHAIN1,C_CHAIN2])
                        print(f"NOW RUNNING RPA {key}")
                        RS.Initialize()
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
                                print(k,prev_ss,curr_ss)
                                if prev_ss != curr_ss:
                                    lb_idx, ub_idx = k-1, k
                                    transition = (res[lb_idx], res[ub_idx],[prev_ss,curr_ss],comp[3])
                                    if key not in phase_transitions.keys():
                                        phase_transitions[key] = []
                                    phase_transitions[key].append(transition)
                    else:
                        RS.beadtypes = [['HOH', 0.31, 0], ['GLUd', 0.75, -1],['GLUp', 0.75, 0], ['TMA', 0.31, 1]]
                        RS.moltypes = [['HOH',[0]], ['TMA',[3]], ['Chain1',chain_mol_beads1],['Chain2',chain_mol_beads2]]
                        RS.molbonds = [['HOH',[]], ['TMA',[]], ['Chain1',[[idx,idx+1] for idx in range(chain_len-1)]],['Chain2',[[idx,idx+1] for idx in range(chain_len-1)]]]
                        RS.molweights = [['HOH',18.01528],['TMA',74.14],['Chain1',aro_comp_fraction*chain_len*161.2004+num_deprot_monos1*128.1060+num_prot_monos1*129.114],['Chain2',aro_comp_fraction*chain_len*161.2004+num_deprot_monos2*128.1060+num_prot_monos2*129.114]] # chain is 5 ARO and 5 GLU
                        chain_vol = (1-aro_comp_fraction)*chain_len*0.130306509511330 + (aro_comp_fraction*chain_len)*0.22259864959526
                        RS.molvols = [['HOH',0.03001009271548933],['TMA',0.11820332057803033],['Chain1',chain_vol],['Chain2',chain_vol]]
                        RS.comps = []
                        for data_col in res:
                            n_chains1, n_chains2, nHOH, nTMA, nOH, nH3O = data_col
                            total_weight = n_chains2*RS.molweights[-1][-1] + n_chains1*RS.molweights[-2][-1] + nTMA*RS.molweights[1][-1] + nHOH*RS.molweights[0][-1]
                            C_CHAIN1,C_CHAIN2, C_OH, C_TMA, C_W = n_chains1*RS.molweights[-2][-1]/total_weight, n_chains2*RS.molweights[-1][-1]/total_weight, 0, nTMA*RS.molweights[1][-1]/total_weight, nHOH*RS.molweights[0][-1]/total_weight
                            RS.comps.append([C_W,C_TMA,C_CHAIN1,C_CHAIN2])
                            print([C_W,C_TMA,C_CHAIN1,C_CHAIN2])
                        print(f"NOW RUNNING RPA {key}")
                        RS.Initialize()
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
                                print(k,prev_ss,curr_ss)
                                if prev_ss != curr_ss:
                                    lb_idx, ub_idx = k-1, k
                                    transition = (res[lb_idx], res[ub_idx],[prev_ss,curr_ss],comp[3])
                                    if key not in phase_transitions.keys():
                                        phase_transitions[key] = []
                                    phase_transitions[key].append(transition)
        elif is_homopolymer_A:
            print("STILL NEED TO IMPLEMENT, THOUGH NOT PRESSING.")
            exit()
            chain_mol_beads = [1 for i in chain_sequence]
            
            res = [data_col for data_col in species_counts_data.values()] # [n_chain1, nHOH, nTMA, nOH]
            print("Running 100% deprotonated peptoid chains.")
            RPA_OBJECTS[key] = RPA.RPAsystem()
            RS = RPA_OBJECTS[key]
            RS.ff_file = './force_fields/current/cg_ff_GLUd_only.dat'
            RS.out = 'ARO_GLU'
            RS.bondtype = 'offset'
            RS.r_max = 10.
            RS.n_r = 10000
            RS.k_start = 0.001
            RS.k_end = 25.0
            RS.n_k = 10000
            n = 1000
            RS.beadtypes = [['HOH', 0.31, 0], ['GLUd', 0.75, -1], ['TMA', 0.31, 1], ['OH', 0.31, -1]]
            RS.moltypes = [['HOH',[0]], ['TMA',[2]], ['OH',[3]], ['Chain',[1 for i in range(len(chain_sequence))]]]
            RS.molbonds = [['HOH',[]], ['TMA',[]], ['OH',[]], ['Chain',[[idx,idx+1] for idx in range(chain_len-1)]]]
            RS.molweights = [['HOH',18.01528],['TMA',74.14],['OH',17.007],['Chain',aro_comp_fraction*chain_len*161.2004+(1-aro_comp_fraction)*chain_len*128.1060]] # chain is 5 ARO and 5 GLU
            chain_vol = (1-aro_comp_fraction)*chain_len*0.130306509511330 + (aro_comp_fraction*chain_len)*0.22259864959526
            RS.molvols = [['HOH',0.03001009271548933],['TMA',0.11820332057803033],['OH',0.03001009271548933],['Chain',chain_vol]]
            RS.comps = []
            for data_col in res:
                n_chains, nHOH, nTMA, nOH, nH3O = data_col
                total_weight = n_chains*RS.molweights[-1][-1] + nOH*RS.molweights[-2][-1] + nTMA*RS.molweights[-3][-1] + nHOH*RS.molweights[-4][-1]
                C_CHAIN, C_OH, C_TMA, C_W, C_H3O = n_chains*RS.molweights[-1][-1]/total_weight, nOH*RS.molweights[-2][-1]/total_weight, nTMA*RS.molweights[-3][-1]/total_weight, nHOH*RS.molweights[-4][-1]/total_weight, 0
                #if C_OH > 0:
                RS.comps.append([C_W,C_TMA,C_OH,C_CHAIN])
            RS.Initialize()
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
        else:
            print("NOT A HOMOPOLYMER.")
            exit()
        output_data = []
        with open("results/rpa/ARO_GLU_RPA.txt", "w") as f: # [C_W,C_TMA,C_OH,C_CHAIN,C_H3O]
            f.write("# weight fractions/chain concentrations : HOH   TMA   OH   Chain  H3O\n")
            f.write("# 0:stable, 1:macophase instability, 2:mesophase instability\n")
            for p in stability_data:
                for i in p[0:-1]:
                    f.write(f"{format(i, '.10f'):12.10}")
                f.write("{}\n".format(p[-1]))
            f.close()
        
        # LOCATE INITIAL SPINODAL
        stability_all = []
        polymer_concs = []
        with open("results/rpa/ARO_GLU_RPA.txt","r") as f:
            all_lines = f.readlines()
            for line in all_lines:
                if "#" in line:
                    continue
                cHOH, cTMA, cOH, cPoly, cH3O, stability = [float(i) for i in line.split()]
                stability_all.append(int(stability))
                polymer_concs.append(cPoly)
        l, r = 0, 1
        spinodal_idx = -1
        while r < len(stability_all):
            curr_stability, next_stability = stability_all[l], stability_all[r]
            if next_stability != curr_stability:
                spinodal_idx = r
                break
            l += 1
            r += 1
        spinodal_type, polymer_wt_spinodal = stability_all[spinodal_idx], polymer_concs[spinodal_idx]
        first_spindoals.append((len(chain_sequence),f"fA={aro_comp_fraction}",polymer_wt_spinodal,spinodal_type))
        output_data.append([len(chain_sequence),aro_comp_fraction,polymer_wt_spinodal,spinodal_type])
        print("FINISHED.")
    print(first_spindoals)
    output_data = np.array(output_data)
    print(output_data)
    np.savetxt("results/rpa/scratch.csv", output_data, delimiter=",")

main()
