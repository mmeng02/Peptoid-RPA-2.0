import numpy as np
from scipy.constants import Avogadro as avogadro
import SCFT_tools
from SCFT_tools import RPA_DZ as RPA
from matplotlib import pyplot as plt
from matplotlib import font_manager, rcParams

def calc_conc(total_volume,n_molecules):
    return (n_molecules/avogadro)/(total_volume*(1/10**21)*0.001) # in units of mol/L

def calc_mol_count(total_volume,concentration): # total_volume [=] nm3, concentration [=] mol/L
    return int(round(total_volume*(1/10**21)*0.001*concentration*avogadro)) # in units of # of molecules

def get_species_counts(chain_sequence,dw=1e-10,max_conc=0.99,tmah_conc=0.0238,precision=0.001):
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
    tot_w = chain_w/dw
    # — # deprotected monomers/ # total GLU monomers
    deprotection_ratios = list(i*(1/num_glu_per_chain) for i in range(num_glu_per_chain+1))
    stride = int(precision/dw)
    C0 = np.linspace(dw,max_conc,round(max_conc/(dw*stride)))
    DS = {}
    i = 0
    for j in range(0,len(C0)):
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
            if len(bounds) == 2:
                net_charge = -1*n_chains*chain_len*glu_comp_fraction + n_TMA - n_OH + n_H3O
                tot_vol = n_chains*chain_vol + n_HOH*0.03001009271548933 + n_TMA*0.11820332057803033 + n_OH*0.03001009271548933 + n_H3O*0.03001009271548933
                #print("Net Charge", net_charge, [n_chains,n_HOH,n_TMA,n_OH,n_H3O])
                if net_charge < 1e-10:
                    DS[bounds][j, tot_vol] = [n_chains,n_HOH,n_TMA,n_OH,n_H3O]
                else:
                    print("NOT ELECTRO-NEUTRAL")
                    exit()
            else:
                lb_deprot_frac, ub_deprot_frac = bounds[0], bounds[1]
                #print(DS)
                net_charge = n_TMA - n_OH + n_H3O - n_type1_chains*lb_deprot_frac*chain_len*glu_comp_fraction - n_type2_chains*ub_deprot_frac*chain_len*glu_comp_fraction 
                #print("Net Charge", net_charge, [n_type1_chains,n_type2_chains,n_HOH,n_TMA,n_OH,n_H3O])
                tot_vol = n_chains*chain_vol + n_HOH*0.03001009271548933 + n_TMA*0.11820332057803033 + n_OH*0.03001009271548933 + n_H3O*0.03001009271548933
                if net_charge < 1e-10:
                    #print(f"Added {n_type1_chains+n_type2_chains} chains.")
                    DS[bounds][j, tot_vol] = [n_type1_chains,n_type2_chains,n_HOH,n_TMA,n_OH,n_H3O]
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
        else:
            deprot_frac1, deprot_frac2 = key[0], key[1]
    return DS

def plot_deprotonation_fraction():
    plt.figure()
    plt.style.use("../styles.mplstyle")
    chain_sequences = ["A"*i+"G"*(20-i) for i in range(20)][::2]
    chain_sequences = ["A"*10+"G"*10]
    for chain_sequence in chain_sequences:
        dw = 0.001
        dw = 1e-10
        glu_frac = 0
        for char in chain_sequence:
            if char == "G":
                glu_frac += 1
        chain_len = len(chain_sequence)
        glu_frac /= chain_len
        data = []
        DS = get_species_counts(chain_sequence,dw=dw)
        for key in DS.keys():
            if len(key) == 3:
                deprot_frac1, deprot_frac2 = key[0], key[1]
                series = DS[key]
                for n_poly in series.keys():
                    weight = n_poly*dw
                    nChain1, nChain2 = series[n_poly][0], series[n_poly][1]
                    n_GLUd = (deprot_frac1*nChain1*chain_len + deprot_frac2*nChain2*chain_len)*glu_frac
                    tot_GLU = (nChain1+nChain2)*chain_len*glu_frac
                    data.append([weight, n_GLUd/tot_GLU])
            else:
                deprot_frac = key[0]
                series = DS[key]
                for n_poly in series.keys():
                    weight = n_poly*dw
                    nChain = series[n_poly][0]
                    n_GLUd = deprot_frac*nChain*chain_len*glu_frac
                    tot_GLU = nChain*chain_len*glu_frac
                    data.append([weight, n_GLUd/tot_GLU])
        data = np.array(data)
        plt.plot(data[:,0],data[:,1],label=f"{chain_sequence}")
    plt.legend(title=r"$N = 20,$ $f_A$")
    plt.xlabel(r"Polymer weight fraction, $w_{poly}$")
    plt.ylabel(r"$n_{GLUd}$/($n_{GLUd}$+$n_{GLUp}$)")
    plt.savefig("glutamic_deprotonation_ratio.png")
    plt.show()

def plot_deprotonation_fraction2():
    plt.figure()
    plt.style.use("../styles.mplstyle")
    chain_sequences = ["A"*i+"G"*(20-i) for i in range(20)][::2]

    for chain_sequence in chain_sequences:
        dw = 1e-3
        glu_frac = 0
        for char in chain_sequence:
            if char == "G":
                glu_frac += 1

        chain_len = len(chain_sequence)
        glu_frac /= chain_len
        data = []
        DS = get_species_counts(chain_sequence, dw=dw)

        for key in DS.keys():
            series = DS[key]

            if len(key) == 3:
                deprot_frac1, deprot_frac2 = key[0], key[1]
                for (j, tot_vol), values in series.items():
                    weight = j * dw
                    nChain1, nChain2 = values[0], values[1]
                    n_GLUd = (deprot_frac1 * nChain1 * chain_len + deprot_frac2 * nChain2 * chain_len) * glu_frac
                    tot_GLU = (nChain1 + nChain2) * chain_len * glu_frac
                    data.append([weight, n_GLUd / tot_GLU])

            else:
                deprot_frac = key[0]
                for (j, tot_vol), values in series.items():
                    weight = j * dw
                    nChain = values[0]
                    n_GLUd = deprot_frac * nChain * chain_len * glu_frac
                    tot_GLU = nChain * chain_len * glu_frac
                    data.append([weight, n_GLUd / tot_GLU])

        data = np.array(data)
        plt.plot(data[:, 0], data[:, 1], label=f"{round(1-glu_frac,2)}")

    plt.legend(title=r"$N = 20,$ $f_A$")
    plt.xlabel(r"Polymer weight fraction, $w_{poly}$")
    #plt.ylabel(r"$n_{GLUd}$/($n_{GLUd}$+$n_{GLUp}$)")
    plt.ylabel("Glutamic deprotonation fraction")
    plt.savefig("glutamic_deprotonation_ratio.png")
    plt.show()

def pH_calculation():
    # pH
    plt.figure()
    plt.style.use("../styles.mplstyle")
    chain_sequences = ["A"*10+"G"*10]
    for chain_sequence in chain_sequences:
        for tmah_conc in [0.01, 0.0238, 0.05, 0.1]:
            glu_frac = 0
            for char in chain_sequence:
                if char == "G":
                    glu_frac += 1
            chain_len = len(chain_sequence)
            glu_frac /= chain_len
            data = []
            # specify precision of speciation calculation
            dw = 1e-10 # infinite box limit
            precision = 0.001
            if precision < dw:
                print(f"dw must be smaller than the desired precision.")
                print("Exiting program.")
                exit()
            stride = round(precision/dw)
            DS = get_species_counts(chain_sequence,dw=dw,tmah_conc=tmah_conc,precision=precision)
            for key in DS.keys():
                if len(key) == 3:
                    deprot_frac1, deprot_frac2 = key[0], key[1]
                    series = DS[key]
                    for n_poly, tot_vol in series.keys():
                        key = (n_poly, tot_vol)
                        weight = n_poly*dw*stride
                        nChain1, nChain2, nHOH, nTMA, nOH, nH3O = series[key]
                        if nOH > 0.0:
                            OH_conc = calc_conc(tot_vol,nOH)
                            pOH = -np.log10(OH_conc)
                            pH = 14 - pOH
                            data.append([weight, pH])
                        elif nH3O > 0.0:
                            H3O_conc = calc_conc(tot_vol,nH3O)
                            pH = -np.log10(H3O_conc)
                            data.append([weight,pH])
                        else:
                            data.append([weight,7.0])
                else:
                    deprot_frac = key[0]
                    series = DS[key]
                    for n_poly, tot_vol in series.keys():
                        key = (n_poly, tot_vol)
                        weight = n_poly*dw*stride
                        nChain, nHOH, nTMA, nOH, nH3O = series[key]
                        if nOH > 0.0:
                            OH_conc = calc_conc(tot_vol,nOH)
                            pOH = -np.log10(OH_conc)
                            pH = 14 - pOH
                            data.append([weight, pH])
                        elif nH3O > 0.0:
                            H3O_conc = calc_conc(tot_vol,nH3O)
                            pH = -np.log10(H3O_conc)
                            data.append([weight,pH])
                        else:
                            data.append([weight,7.0])
            data = np.array(data)
            plt.plot(data[:,0],data[:,1],label=f"{round(tmah_conc*100,2)}%")
    plt.legend(title=r"$TMAH$ $wt$%")
    plt.title(r"$N = 20,$ $f_A = 0.5$")
    plt.xlabel(r"Polymer weight fraction, $w_{poly}$")
    plt.ylabel(r"$pH$")
    plt.savefig("pH_v_polywt.png")
    plt.show()

def main():
    font_manager.fontManager.addfont("/Users/MichaelMeng1/Documents/Graduate School/Spring26/cmu_serif_roman.ttf")
    prop = font_manager.FontProperties(fname="/Users/MichaelMeng1/Documents/Graduate School/Spring26/cmu_serif_roman.ttf")
    font_name = prop.get_name()
    #plt.style.use("styles.mplstyle")
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = [font_name]
    plot_deprotonation_fraction2()
    exit()
    plt.figure()
    plt.style.use("../styles.mplstyle")
    chain_sequences = ["A"*10+"G"*10]
    for chain_sequence in chain_sequences:
        for tmah_conc in [0.0238]:
            for dw in [1e-2,1e-4,1e-6]:
                glu_frac = 0
                for char in chain_sequence:
                    if char == "G":
                        glu_frac += 1
                chain_len = len(chain_sequence)
                glu_frac /= chain_len
                data = []
                # specify precision of speciation calculation
                if dw == 1e-2:
                    precision = 0.01
                else:
                    precision = 0.001
                if precision < dw:
                    print(f"dw must be smaller than the desired precision.")
                    print("Exiting program.")
                    exit()
                stride = round(precision/dw)
                DS = get_species_counts(chain_sequence,dw=dw,tmah_conc=tmah_conc,precision=precision)
                for key in DS.keys():
                    if len(key) == 3:
                        deprot_frac1, deprot_frac2 = key[0], key[1]
                        series = DS[key]
                        for n_poly, tot_vol in series.keys():
                            key = (n_poly, tot_vol)
                            weight = n_poly*dw*stride
                            nChain1, nChain2, nHOH, nTMA, nOH, nH3O = series[key]
                            if nOH > 0.0:
                                OH_conc = calc_conc(tot_vol,nOH)
                                pOH = -np.log10(OH_conc)
                                pH = 14 - pOH
                                data.append([weight, pH])
                            elif nH3O > 0.0:
                                H3O_conc = calc_conc(tot_vol,nH3O)
                                pH = -np.log10(H3O_conc)
                                data.append([weight,pH])
                            else:
                                data.append([weight,7.0])
                    else:
                        deprot_frac = key[0]
                        series = DS[key]
                        for n_poly, tot_vol in series.keys():
                            key = (n_poly, tot_vol)
                            weight = n_poly*dw*stride
                            nChain, nHOH, nTMA, nOH, nH3O = series[key]
                            if nOH > 0.0:
                                OH_conc = calc_conc(tot_vol,nOH)
                                pOH = -np.log10(OH_conc)
                                pH = 14 - pOH
                                data.append([weight, pH])
                            elif nH3O > 0.0:
                                H3O_conc = calc_conc(tot_vol,nH3O)
                                pH = -np.log10(H3O_conc)
                                data.append([weight,pH])
                            else:
                                data.append([weight,7.0])
                data = np.array(data)
                plt.plot(data[:,0],data[:,1],label=fr"{dw}")
    plt.legend(title=r"$dw$")
    plt.title(r"$N = 20,$ $f_A = 0.5$, $2.38$ $TMAH$ $wt$%")
    plt.xlabel(r"Polymer weight fraction, $w_{poly}$")
    plt.ylabel(r"$pH$")
    plt.savefig("pH_v_polywt_dw.png")
    plt.show()
main()

