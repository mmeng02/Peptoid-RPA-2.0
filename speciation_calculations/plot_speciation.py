import numpy
from matplotlib import pyplot as plt

def main():
    data = [0.9115789474, 0.7515789474]
    plt.style.use("../styles.mplstyle")
    plt.bar(['0.0','0.2'],data,width=0.5)
    plt.xlabel(r"Aromatic compostion, $f_A$")
    plt.ylabel(r"deprotonated monomer fraction, $n_{GLU_d}/n_{T}$")
    plt.title(r"$N$ = 5 glutamic speciation, at $w_{poly}$ = 0.095")
    plt.savefig("protonation_ratio.png")
    plt.show()

main()