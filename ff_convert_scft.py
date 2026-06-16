import numpy as np

def parse_cg_ff(filename,identity_dict):
    output_dict = {}
    with open(filename,"r") as f:
        lines = f.readlines()
        for line in lines:
            if '>' in line:
                species_pair = line.split()[-1].split('_')[1:]
                pair_id = tuple(sorted([identity_dict[species_pair[0]],identity_dict[species_pair[1]]]))
                continue
            else:
                output_dict[pair_id] = eval(line)
    return output_dict
    
# <B*(np.pi * kappa**(-1)) ** (1.5)>
def main():
    identity_dict = {'HOH':1,'ARO':2,'GLUd':3,'OH':4,'TMA':5}
    res_dict = parse_cg_ff('cg_ff.txt',identity_dict)
    keys = sorted(list(res_dict.keys()))
    for key in keys:
        b, kappa = res_dict[key]['B'],res_dict[key]['Kappa']
        print(f"BExclVolume{key[0]}{key[1]} = {b*(np.pi*kappa**(-1))**(1.5)}")
main()