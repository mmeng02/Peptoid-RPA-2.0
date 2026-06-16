import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
import pandas as pd
from matplotlib.lines import Line2D

# Sequence Features of Copolymers
# 17 November 2025
# A. Akkiraju et. al — Predicting Copolymer Critical Parameters with a Theory-Integrated Neutral Network

def scd(sequence): # sequence charge decoration: measures spatial distribution of G monomers
    r = len(sequence)
    scd = 0
    def scd_weight(r,i,j):
        w_ij = ((r-min(i,r-1-i))+(r-min(j,r-1-j)))/(2*r)
        return w_ij
    for i in range(r-1):
        for j in range(i+1,r):
            qi = 1 if sequence[i] == 'A' else -1
            qj = 1 if sequence[j] == 'A' else -1
            w_ij = scd_weight(r,i,j)
            scd += qi*qj*(j-i)*w_ij
    return scd

def norm_scd(sequence):
    r = len(sequence)
    scd_ = scd(sequence)
    # get min scd — hydrophilic homopolymer
    scd_min = scd("".join(["G" for i in range(r)]))

    # get max scd - diblock copolymer f = 0.5
    scd_max = scd("".join(["A" for i in range(round(r/2))]) + "".join(["G" for i in range(round(r/2))]))
    scd_norm = (scd_ - scd_min)/(scd_max - scd_min)
    return scd_norm

def avg_solvo_block_len(sequence): # average hydrophilic block length L_T
    N_blocks = len([seq for seq in sequence.split("A") if seq])
    n_solvo = 0
    for i in range(len(sequence)):
        if sequence[i] == "G":
            n_solvo += 1
    return n_solvo/N_blocks

def solvo_clusters(sequence): # N_adj, number of adjacent hydrophilic monomers
    N_adj = 0
    for i in range(len(sequence)-1):
        if sequence[i] == "G" and sequence[i+1] == "G":
            N_adj += 1
    return N_adj

def screening_score(sequence): # chi_S, how well 
    nT = 0
    r = len(sequence)
    for i in range(len(sequence)):
        if sequence[i] == "G":
            nT += 1
    N_adj = solvo_clusters(sequence)
    chi_s = max(0,1-0.5*(nT/r + N_adj/(nT - 1)))
    return chi_s

def local_autocorrelation(sequence): # correlations of neighboring bead identities: rho -> -1 (alternating, negative correlation), rho -> 1 (blocky, repeats more frequent)
    r = len(sequence)
    tot = 0
    for i in range(len(sequence)-1):
        qi = 1 if sequence[i] == "A" else -1
        qj = 1 if sequence[i+1] == "A" else -1
        tot += qi*qj
    return tot/(r-1)

def transition_density(sequence): # bead identity changes
    r = len(sequence)
    tot = 0
    for i in range(len(sequence)-1):
        si, sj = sequence[i], sequence[i+1]
        if si != sj:
            tot += 1
    tao = tot/(r-1)
    return tao

def all_possible_sequences(chain_length, fA):
    nA = round(chain_length*fA)
    nG = chain_length - nA
    sequences = []
    for A_positions in combinations(range(chain_length), nA):
        seq = ['G'] * chain_length
        for pos in A_positions:
            seq[pos] = 'A'
        sequences.append(''.join(seq))
    return sequences

def feature_plotter(sequences,solubilities,feature_name,fA,N=20):
    plt.style.use("styles.mplstyle")
    if feature_name == "transition density":
        x = np.array([transition_density(seq) for seq in sequences])/max([transition_density(seq) for seq in sequences])
        xlabel = r"$\tau$/$\tau_{max}$"
    elif feature_name == "local autocorrelation":
        x = [local_autocorrelation(seq) for seq in sequences]
        xlabel = r"$\rho$"
    elif feature_name == "screening score":
        x = [screening_score(seq) for seq in sequences]
        xlabel = r"$\chi$_{s}"
    elif feature_name == "clusters":
        x = [solvo_clusters(seq) for seq in sequences]
        xlabel = r"$N$_{adj}"
    elif feature_name == "average block length":
        x = [avg_solvo_block_len(seq) for seq in sequences]
        xlabel = r"$L$_T"
    plt.scatter(x,solubilities)
    plt.plot(x,solubilities,label=rf"{fA}")
    plt.xlabel(xlabel)
    plt.ylabel(r"$w^{*}_{poly}$")
    plt.legend(title=rf"$N$ = {N}, $f_A$")
    #plt.savefig("hello.png")

def transition_density_analysis():
    res = all_possible_sequences(20,0.6)
    transition_density_dict = {}
    for sequence in res:
        trans_dens = transition_density(sequence)
        if trans_dens not in transition_density_dict:
            transition_density_dict[trans_dens] = [sequence]
        else:
            transition_density_dict[trans_dens].append(sequence)
    seqs = []
    for key in sorted(list(transition_density_dict.keys())):
        idx = int(len(transition_density_dict[key])/2)-1
        print(key,transition_density_dict[key][idx])
        seqs.append(transition_density_dict[key][idx])
    print(seqs)
    #exit()
    ### Transition Densities
    # N = 20, fA = 0.2, transition densities
    sequences1 = ['GGGGGGGGGGGGGGGGAAAA', 'AAAGGAGGGGGGGGGGGGGG', 'AAGAGGAGGGGGGGGGGGGG', 'AGGGGAGGGGAGGGGAGGGG','GGAGGAGGGGAGGGGAGGGG']
    trans1 = [0.0517873, 0.0664837, 0.0744813, 0.1241601, 0.1241601]
    feature_plotter(sequences1,trans1,"transition density", 0.2)

    # N = 20, fA = 0.4, transition densities
    sequences2 = ['AAAAAAAAGGGGGGGGGGGG', 'GGAAAAAAAAGGGGGGGGGG', 'AGGGGGGGGGGGAAAAAAAG', 'GGAAAAAAAGGGGGGGAGGG', 'AGGGGGGGGGGAGAAAAAAG', 'GAGGAGAAAAAAGGGGGGGG', 'AGGGGGGGGGAGAGAAAAAG', 'GAGAGGGGAGGAAAAAGGGG', 'AGGGGGGGGAGAGAGAAAAG', 'GAGAGGGAGGGGAAAAGGAG', 'AGGGGGGGAGAGAGAGAAAG', 'GAGAGGGAGGGAAGAGAAGG', 'AGGGGGGAGAGAGAGAGAAG', 'GAGAGGGAGGAGAGAAGAGG', 'AGGGGGAGAGAGAGAGAGAG', 'GAGAGGGAGAGGAGAGAGAG'][::2]
    trans2 = [0.0182986, 0.0254972, 0.0314958, 0.0384937, 0.0334952, 0.0439917, 0.0391934, 0.1150626, 0.050689, 0.1421461, 0.0778793, 0.1421461, 0.1134634, 0.1421461, 0.1421461, 0.1421461][::2]
    feature_plotter(sequences2,trans2,"transition density", 0.4)

    # N = 20, fA = 0.5, transition densities
    sequences3 = ['GGGGGGGGGGAAAAAAAAAA', 'AAAAAAAAGGGGGGGGGGAA', 'AAAAAAAAAGGAGGGGGGGG', 'AAAAAAAAGGAGGGGGGGGA', 'AAAAAAAAGAGGAGGGGGGG', 'AAAAAAAGAGGAGGGGGGGA', 'AAAAAAAGAGAGGAGGGGGG', 'AAAAAAGAGAGGAGGGGGGA', 'AAAAAAGAGAGAGGAGGGGG', 'AAAAAGAGAGAGGAGGGGGA', 'AAAAAGAGAGAGAGGAGGGG', 'AAAAGAGAGAGAGGAGGGGA', 'AAAAGAGAGAGAGAGGAGGG', 'AAAGAGAGAGAGAGGAGGGA', 'AAAGAGAGAGAGAGAGGAGG', 'AAGAGAGAGAGAGAGGAGGA', 'AAGAGAGAGAGAGAGAGGAG', 'AGAGAGAGAGAGAGAGGAGA', 'GAGAGAGAGAGAGAGAGAGA'][::2]
    trans3 = [0.0104996, 0.017099, 0.0111996, 0.0144993, 0.0118995, 0.0156991, 0.0132994, 0.0180988, 0.0155992, 0.0222983, 0.0193987, 0.0304967, 0.0266975, 0.0510909, 0.0445931, 0.1100667, 0.1050693, 0.1356531, 0.135453][::2]
    feature_plotter(sequences3,trans3,"transition density", 0.5)

    # N = 20, fA = 0.6, transition densities
    sequences4 = ['AAAAAAAAAAAAGGGGGGGG', 'AAAGGGGGGGGAAAAAAAAA', 'AGGGGGGGAAAAAAAAAAAG', 'AAGGGGGGGAAAAAAAAGAA', 'AGGGGGGAGAAAAAAAAAAG', 'AGAAGAAGGGGGGAAAAAAA', 'AGGGGGAGAGAAAAAAAAAG', 'AGAGAAAAGAAAAGGGGGAA', 'AGGGGAGAGAGAAAAAAAAG', 'AGAGAAAGAAAAGGGAGGAA', 'AGGGAGAGAGAGAAAAAAAG', 'AGAGAAAGAAAGGAGAAGGA', 'AGGAGAGAGAGAGAAAAAAG', 'AGAGAAAGAAGAGAGGAAGA', 'AGAGAGAGAGAGAGAAAAAG', 'AGAGAAAAGAGAGAGAGAGA'][::2]
    trans4 = [0.0053999, 0.0065999, 0.0062999, 0.0065999, 0.0062999, 0.0065999, 0.0064999, 0.0065999, 0.0065999, 0.0065999, 0.0065999, 0.0065999, 0.0065999, 0.0065999, 0.0065999, 0.0065999][::2]
    feature_plotter(sequences4,trans4,"transition density", 0.6)

    plt.savefig("transition_density_norm.png")
    # scratch.csv containing Pandas df columns to be pasted into excel.
    #df = pd.DataFrame([transition_density(seq) for seq in sequences2])
    df = pd.DataFrame(trans4)
    df.to_csv('scratch.csv',index=False)

def hydrophobic_fraction():
    plt.style.use("styles.mplstyle")
    fA5 = [i*0.2 for i in range(0,6)]
    spinodals5 = [0.0938934] + [0.114675, 0.1319592, 0.0286978, 0.0056999, 0.0026879]
    fA10 = [i*0.10 for i in range(0,11)]
    spinodals10 = [0.0980928,0.1065732, 0.1174634, 0.0929734, 0.0615847, 0.029597, 0.0123996, 0.0048, 0.0029, 0.002,0.002]
    fA20 = [i*0.05 for i in range(0,21)]
    spinodals20 = [0.0978929,0.1035884, 0.0985767, 0.0759821, 0.0576858, 0.0452885, 0.0334943, 0.0256969, 0.0198983, 0.0152991, 0.0115995, 0.0084998, 0.0058999, 0.0039, 0.0026, 0.0019, 0.0015, 0.0012, 0.001, 0.0009,0.0009]
    fA30 = [i/30 for i in range(0,31)][::2]
    spinodals30 = [0.0979928, 0.1019898, 0.1061865, 0.110483, 0.0503885, 0.0416245, 0.0368919, 0.0275957, 0.0214975, 0.0172985, 0.014199, 0.0117994, 0.0098996, 0.0083997, 0.0070998, 0.0059999, 0.0049999, 0.0041999, 0.0034, 0.0027, 0.0021, 0.0016, 0.0013, 0.0011, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0006,0.0006][::2]
    spinodal_type5 = [1,1,1,2,1,1]
    fA5_micro = [fA for fA,m in zip(fA5,spinodal_type5) if m == 2]
    spinodals5_micro = [spinodal for spinodal,m in zip(spinodals5,spinodal_type5) if m == 2]
    fA5_macro = [fA for fA,m in zip(fA5,spinodal_type5) if m == 1]
    spinodals5_macro = [spinodal for spinodal,m in zip(spinodals5,spinodal_type5) if m == 1]
    plt.plot(fA5,spinodals5,label="5")
    plt.scatter(fA5_micro,spinodals5_micro,facecolors='white',edgecolors="C0",s=60,zorder=3)
    plt.scatter(fA5_macro,spinodals5_macro)
    plt.ylabel(r"Spinodal weight fraction, $w^*_{poly}$")
    plt.xlabel(r"Aromatic composition, $f_A$")
    plt.legend(title=r"$N$")
    plt.savefig("hydrophobic_fraction_N5_N10.png")
    exit()
    #plt.scatter(fA5,spinodals5)
    spinodal_type10 = [1,1,1,2,2,2,2,1,1,1,1]
    fA10_micro = [fA for fA,m in zip(fA10,spinodal_type10) if m == 2]
    spinodals10_micro = [spinodal for spinodal,m in zip(spinodals10,spinodal_type10) if m == 2]
    fA10_macro = [fA for fA,m in zip(fA10,spinodal_type10) if m == 1]
    spinodals10_macro = [spinodal for spinodal,m in zip(spinodals10,spinodal_type10) if m == 1]
    plt.plot(fA10,spinodals10,label="10")
    plt.scatter(fA10_micro,spinodals10_micro,facecolors='white',edgecolors="C1",s=60,zorder=3)
    plt.scatter(fA10_macro,spinodals10_macro)

    ####
    #plt.scatter(fA,spinodals)
    #plt.plot(fA,spinodals,label="20")
    spinodal_type20 = [1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1]
    fA20_micro = [fA for fA,m in zip(fA20,spinodal_type20) if m == 2]
    spinodals20_micro = [spinodal for spinodal,m in zip(spinodals20,spinodal_type20) if m == 2]
    fA20_macro = [fA for fA,m in zip(fA20,spinodal_type20) if m == 1]
    spinodals20_macro = [spinodal for spinodal,m in zip(spinodals20,spinodal_type20) if m == 1]
    plt.plot(fA20,spinodals20,label="20")
    plt.scatter(fA20_micro,spinodals20_micro,facecolors='white',edgecolors="C2",s=60,zorder=3)
    plt.scatter(fA20_macro,spinodals20_macro)
    ####
    #plt.scatter(fA30,spinodals30)
    #plt.plot(fA30,spinodals30,label="30")
    spinodal_type30 = [1,1,2,2,2,2,2,2,2,2,1,1,1,1,1,1]
    fA30_micro = [fA for fA,m in zip(fA30,spinodal_type30) if m == 2]
    spinodals30_micro = [spinodal for spinodal,m in zip(spinodals30,spinodal_type30) if m == 2]
    fA30_macro = [fA for fA,m in zip(fA30,spinodal_type30) if m == 1]
    spinodals30_macro = [spinodal for spinodal,m in zip(spinodals30,spinodal_type30) if m == 1]
    plt.plot(fA30,spinodals30,label="30")
    plt.scatter(fA30_micro,spinodals30_micro,facecolors='white',edgecolors="C3",s=60,zorder=3)
    plt.scatter(fA30_macro,spinodals30_macro)
    ####
    plt.ylabel(r"$w^*_{poly}$")
    plt.xlabel(r"$f_A$")
    plt.legend(title=r"$N$")
    plt.savefig("hydrophobic_fraction.png")
    df = pd.DataFrame(fA30)
    df.to_csv('scratch.csv',index=False)

def plot_local_autocorrelation():
    res = all_possible_sequences(20,0.5)
    local_autocorrelations_dict = {}
    for sequence in res:
        trans_dens = local_autocorrelation(sequence)
        if trans_dens not in local_autocorrelations_dict:
            local_autocorrelations_dict[trans_dens] = [sequence]
        else:
            local_autocorrelations_dict[trans_dens].append(sequence)
    seqs = []
    for key in sorted(list(local_autocorrelations_dict.keys())):
        idx = int(len(local_autocorrelations_dict[key])/2)-1
        print(key,local_autocorrelations_dict[key][idx])
        seqs.append(local_autocorrelations_dict[key][idx])
    seqs = seqs[::2]
    solubilities = [0.125862, 0.1307597, 0.1198644, 0.0995728, 0.0945746, 0.0461926, 0.0460926, 0.0283972, 0.0289971, 0.0210984, 0.0218983, 0.017199, 0.0182988, 0.0149992, 0.0163991, 0.0137993, 0.0156991, 0.0132994, 0.0104996][::2]
    feature_plotter(seqs,solubilities,"local autocorrelation",0.5,20)
    plt.xlim(-1.1, 1.1)
    plt.savefig("local_autocorrelation.png")
    df = pd.DataFrame([local_autocorrelation(seq) for seq in seqs])
    df.to_csv('scratch.csv',index=False)

def plot_solvo_block_len():
    res = all_possible_sequences(20,0.2)
    avg_solv_block_dict = {}
    for seq in res:
        metric = avg_solvo_block_len(seq)
        if metric not in avg_solv_block_dict:
            avg_solv_block_dict[metric] = [seq]
        else:
            avg_solv_block_dict[metric].append(seq)
    sorted_keys = sorted(list(avg_solv_block_dict.keys()))
    x = sorted_keys
    sequences = []
    for key in sorted_keys:
        sequences.append(avg_solv_block_dict[key][-1])
    print(sequences)
    exit()
    x = [avg_solvo_block_len(seq) for seq in sequences]
    y = [0.1229718, 0.1009765, 0.0719838, 0.0599871, 0.054987]
    res = all_possible_sequences(20,0.3)
    avg_solv_block_dict = {}
    for seq in res:
        metric = avg_solvo_block_len(seq)
        if metric not in avg_solv_block_dict:
            avg_solv_block_dict[metric] = [seq]
        else:
            avg_solv_block_dict[metric].append(seq)
    sorted_keys = sorted(list(avg_solv_block_dict.keys()))
    x0 = sorted_keys
    sequences = []
    for key in sorted_keys:
        sequences.append(avg_solv_block_dict[key][-1])
    x0 = [avg_solvo_block_len(seq) for seq in sequences]
    y0 = [0.1129717, 0.0759813, 0.0549866, 0.0469888, 0.0389919, 0.0349934, 0.0329944]
    res = all_possible_sequences(20,0.4)
    avg_solv_block_dict = {}
    for seq in res:
        metric = avg_solvo_block_len(seq)
        if metric not in avg_solv_block_dict:
            avg_solv_block_dict[metric] = [seq]
        else:
            avg_solv_block_dict[metric].append(seq)
    sorted_keys = sorted(list(avg_solv_block_dict.keys()))
    x1 = sorted_keys
    sequences = []
    for key in sorted_keys:
        sequences.append(avg_solv_block_dict[key][-1])
    x1 = [avg_solvo_block_len(seq) for seq in sequences]
    y1 = [0.121964, 0.0909758, 0.0629835, 0.0389939, 0.0299962, 0.0239979, 0.0209979, 0.0189986, 0.0189986]

    res = all_possible_sequences(20,0.5)
    avg_solv_block_dict = {}
    for seq in res:
        metric = avg_solvo_block_len(seq)
        if metric not in avg_solv_block_dict:
            avg_solv_block_dict[metric] = [seq]
        else:
            avg_solv_block_dict[metric].append(seq)
    sorted_keys = sorted(list(avg_solv_block_dict.keys()))
    x2 = sorted_keys
    sequences = []
    for key in sorted_keys:
        sequences.append(avg_solv_block_dict[key][-1])
    x2 = [avg_solvo_block_len(seq) for seq in sequences]
    y2 = [0.1276609, 0.0912757, 0.0407942, 0.0250978, 0.0184988, 0.0148992, 0.0127994, 0.0114996, 0.0107996, 0.0104996]


    res = all_possible_sequences(20,0.6)
    avg_solv_block_dict = {}
    for seq in res:
        metric = avg_solvo_block_len(seq)
        if metric not in avg_solv_block_dict:
            avg_solv_block_dict[metric] = [seq]
        else:
            avg_solv_block_dict[metric].append(seq)
    sorted_keys = sorted(list(avg_solv_block_dict.keys()))
    x3 = sorted_keys
    sequences = []
    for key in sorted_keys:
        sequences.append(avg_solv_block_dict[key][-1])
    x3 = [avg_solvo_block_len(seq) for seq in sequences]
    y3 = [0.0069998, 0.0069998, 0.0064999, 0.0064999, 0.0059999, 0.0059999, 0.0054999, 0.0054999]

    plt.style.use("styles.mplstyle")
    plt.scatter(x,y)
    plt.plot(x,y,label=r"$f_A$ = 0.2")
    plt.scatter(x0,y0)
    plt.plot(x0,y0,label=r"$f_A$ = 0.3")
    plt.scatter(x1,y1)
    plt.plot(x1,y1,label=r"$f_A$ = 0.4")
    plt.scatter(x2,y2)
    plt.plot(x2,y2,label=r"$f_A$ = 0.5")
    plt.scatter(x3,y3)
    plt.plot(x3,y3,label=r"$f_A$ = 0.6")
    plt.xlabel(r"$L_G$")
    plt.ylabel(r"$w^*_{poly}$")
    plt.legend(title=r"$N = $20")
    plt.savefig("avg_hydropholic_block.png")
    #df = pd.DataFrame(y)
    #df.to_csv('scratch.csv',index=False)

def plot_screening():
    res = all_possible_sequences(20,0.2)
    avg_solv_block_dict = {}
    for seq in res:
        metric = screening_score(seq)
        if metric not in avg_solv_block_dict:
            avg_solv_block_dict[metric] = [seq]
        else:
            avg_solv_block_dict[metric].append(seq)
    sorted_keys = sorted(list(avg_solv_block_dict.keys()))
    x = sorted_keys
    sequences = []
    for key in sorted_keys:
        n = len(avg_solv_block_dict[key])
        sequences.append(avg_solv_block_dict[key][n//2])
    sequences = ['AAAAGGGGGGGGGGGGGGGG','AAAGGGGAGGGGGGGGGGGG','AAGGGGGGAGGGGGGAGGGG','GGAGGGGAGGGGAGGGGAGG']
    x = [screening_score(seq) for seq in sequences]
    y = [0.054987, 0.0882807, 0.1229718, 0.1229718]
    
    res = all_possible_sequences(20,0.4)
    avg_solv_block_dict = {}
    for seq in res:
        metric = screening_score(seq)
        if metric not in avg_solv_block_dict:
            avg_solv_block_dict[metric] = [seq]
        else:
            avg_solv_block_dict[metric].append(seq)
    sorted_keys = sorted(list(avg_solv_block_dict.keys()))
    x1 = sorted_keys
    print(avg_solv_block_dict[0.5636363636363637])
    sequences = []
    for key in sorted_keys:
        n = len(avg_solv_block_dict[key])
        sequences.append(avg_solv_block_dict[key][n//2])
    sequences[0] = 'AAAAAAAAGGGGGGGGGGG'
    sequences = ["AAAAAAAAGGGGGGGGGGG","AAAAAAAGGGGAGGGGGGG","AAAAAAGGGGGAGGGAGGG","AAAAAGGGAGGGAGGGAGGG","AAAAGGGAGGGAGGAGAGGG","AGGAGGAGGAGGAGGAGAGA","GAGGAGAGAGAGAGAGGGAG"]
    x1 = [screening_score(seq) for seq in sequences]
    y1 = [0.0175987,0.0213981,0.0333954,0.0463908,0.0863771,0.1468521,0.1468521]


    plt.style.use("styles.mplstyle")
    plt.scatter(x,y)
    plt.plot(x,y,label=r"$f_A$ = 0.2")
    
    plt.scatter(x1,y1)
    plt.plot(x1,y1,label=r"$f_A$ = 0.4")
    
    plt.xlabel(r"$\chi_{s}$")
    plt.ylabel(r"$w^*_{poly}$")
    plt.legend(title=r"$N = $20")
    plt.savefig("screen_score_all.png")

def plot_autcorr():
    res = all_possible_sequences(20,0.2)
    avg_solv_block_dict = {}
    for seq in res:
        metric = local_autocorrelation(seq)
        if metric not in avg_solv_block_dict:
            avg_solv_block_dict[metric] = [seq]
        else:
            avg_solv_block_dict[metric].append(seq)
    sorted_keys = sorted(list(avg_solv_block_dict.keys()))
    x = sorted_keys
    sequences = []
    for key in sorted_keys:
        n = len(avg_solv_block_dict[key])
        sequences.append(avg_solv_block_dict[key][n//2])
    sequences = ['AAAAGGGGGGGGGGGGGGGG','AAAGGGGAGGGGGGGGGGGG','AAGGGGGGAGGGGGGAGGGG','GGAGGGGAGGGGAGGGGAGG']
    x = [local_autocorrelation(seq) for seq in sequences]
    y = [0.0569879, 0.0699853, 0.0999801, 0.1019781]

    
    res = all_possible_sequences(20,0.4)
    avg_solv_block_dict = {}
    for seq in res:
        metric = local_autocorrelation(seq)
        if metric not in avg_solv_block_dict:
            avg_solv_block_dict[metric] = [seq]
        else:
            avg_solv_block_dict[metric].append(seq)
    sorted_keys = sorted(list(avg_solv_block_dict.keys()))
    x1 = sorted_keys
    sequences = []
    for key in sorted_keys:
        n = len(avg_solv_block_dict[key])
        sequences.append(avg_solv_block_dict[key][n//2])
    sequences[0] = 'AAAAAAAAGGGGGGGGGGG'
    sequences = ["AAAAAAAAGGGGGGGGGGG","AAAAAAAGGGGAGGGGGGG","AAAAAAGGGGGAGGGAGGG","AAAAAGGGAGGGAGGGAGGG","AAAAGGGAGGGAGGAGAGGG","GGAAGGAGGAGGAGGAGAGA","AGGAGGAGGAGGAGGAGAGA","GAGGAGAGAGAGAGAGGGAG"]
    x1 = [local_autocorrelation(seq) for seq in sequences]
    y1 = [0.0199981, 0.0249975, 0.0439915, 0.0609849, 0.0789791, 0.0999738, 0.0999738, 0.0999738]
    print(sorted_keys)
    print(x1)
    print(local_autocorrelation("GGAAGGAAGGGGAGGAGAGA"))
    
    plt.style.use("styles.mplstyle")
    plt.scatter(x,y)
    plt.plot(x,y,label=r"$f_A$ = 0.2")
    
    plt.scatter(x1,y1)
    plt.plot(x1,y1,label=r"$f_A$ = 0.4")
    
    plt.xlabel(r"$\rho_{s}$")
    plt.ylabel(r"$w^*_{poly}$")
    plt.legend(title=r"$N = $20")
    plt.savefig("autocorr_all.png")

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

def main():
    hydrophobic_fraction()
    exit()
    res = all_possible_sequences(20,0.5)
    screening_score_dict = {}
    for seq in res:
        metric = screening_score(seq)
        if metric not in screening_score_dict:
            screening_score_dict[metric] = [seq]
        else:
            screening_score_dict[metric].append(seq)
    sorted_keys = sorted(list(screening_score_dict.keys()))
    sequences = ['GAGAGAGAGAGAGAGAGAGA', 'GGAGAGAGAGAGAGAGAGAA', 'GGGAGAGAGAGAGAGAGAAA', 'GGGGAGAGAGAGAGAGAAAA', 'GGGGGAGAGAGAGAGAAAAA', 'GGGGGGAGAGAGAGAAAAAA', 'GGGGGGGAGAGAGAAAAAAA', 'GGGGGGGGAGAGAAAAAAAA', 'GGGGGGGGGAGAAAAAAAAA', 'GGGGGGGGGGAAAAAAAAAA']
    #for key in sorted_keys:
    #    sequences.append(screening_score_dict[key][len(screening_score_dict[key])//2])
    #print(sorted_keys)
    print(sequences)
    x = [screening_score(seq) for seq in sequences][::-1]
    y = [0.1276609, 0.0912757, 0.0407942, 0.0250978, 0.0184988, 0.0148992, 0.0127994, 0.0114996, 0.0107996, 0.0104996][::-1]
    plt.style.use("styles.mplstyle")
    plt.scatter(x,y)
    plt.plot(x,y,label=r"$f_A$ = 0.5")
    plt.xlabel(r"$\chi_{s}$")
    plt.ylabel(r"$w^*_{poly}$")
    plt.legend(title=r"$N = $20")
    plt.savefig("screening_score_prev_ff.png")
main()

