import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.style.use("styles.mplstyle")
# Data
labels = [r'$f_A$ = 0.4 alt', r'$f_A$ = 0.2 alt', r'$f_A$ = 0.2 blocky']
rpa = [0.08207929, 12.74*0.01, 0.0751712]
experimental = [9.0*0.01, 12.74*0.01, 11.71*0.01]

x = np.arange(len(labels))
width = 0.35

# Create figure
plt.figure()

# RPA bars (plot individually to control hatch)
for i, value in enumerate(rpa):
    if i == 1:  # second bar
        plt.bar(x[i] - width/2, value, width,
                fill=True, linewidth=1, label='RPA' if i == 0 else "",
                hatch='/', edgecolor='black',color="C0")
    else:
        plt.bar(x[i] - width/2, value, width,
                fill=True, linewidth=1, label='RPA' if i == 0 else "", color="C0")


plt.bar(x + width/2, experimental, width,yerr=[0.438*0.01, 0.92*0.01, 0.55*0.01],capsize=5,
        fill=True, linewidth=2, label='Experimental',color="C3")



# Labels and legend
plt.ylabel(r'$w_{poly}^*$',fontsize=24)
plt.xticks(x, labels)
legend_elements = [
    # fA values (line styles)
    Line2D([0], [0], color='C0', lw=7, linestyle='-', label="RPA"),
    Line2D([0], [0], color='C3', lw=7, linestyle='-', label="Experimental"),
Patch(facecolor='white',
          edgecolor='black',
          hatch='/',
          label='Fully disordered (no spinodal)')
    
    ]

plt.legend(handles=legend_elements)
ax = plt.gca()
ax.set_ylim(0.0, 0.2)
plt.tight_layout()
plt.savefig("exp_validation.png")
plt.show()
