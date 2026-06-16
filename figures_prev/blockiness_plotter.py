from matplotlib import pyplot as plt
import numpy as np

# N = 20, fA = 0.6
blockiness1 = [0.0, 0.13333333333333333, 0.26666666666666666, 0.6666666666666666, 0.9333333333333333, 1.0]
spinodals1 = [0.0089997, 0.0089997, 0.0089997, 0.0089997, 0.0079998, 0.0064999]

# N = 20, fA = 0.2
blockiness2 = [0.0, 0.14285714285714285, 0.2857142857142857, 0.5714285714285714, 0.8571428571428571, 1.0]
spinodals2 = [0.1299573, 0.1299573, 0.1299573, 0.1299573, 0.1279581, 0.0764803]

# N = 20, fA = 0.4
blockiness3 = [0.0, 0.13333333333333333, 0.3333333333333333, 0.6, 0.8, 0.9333333333333333, 1.0]
spinodals3 = [0.1154619, 0.1149618, 0.1154619, 0.1149618, 0.1089665, 0.07648, 0.0224978]

# N = 20, fA = 0.1
blockiness4 = [0.0, 0.3333333333333333, 0.6666666666666666, 1.0]
spinodals4 = [0.374995, 0.374995,0.2278554, 0.1774224]

# degree of polymerization = GAAAAG
#dop = [1,2,3,4,5,6,7,8]
#spinodals = [0.0134997, 0.006, 0.0038, 0.0028, 0.0023, 0.0019, 0.0016, 0.0014]
#spinodals = [0.1216603, 0.1199611, 0.119861, 0.120211, 0.1207605, 0.1204109, 0.1199111, 0.1197112]
#spinodals = [0.1239572, 0.1209621, 0.1199611, 0.1184636, 0.1179619, 0.1179632, 0.1174627, 0.1184624]
plt.style.use("styles.mplstyle")
fA = [i*0.05 for i in range(1,20)]
#spinodals = [0.28325, 0.1779233, 0.11047, 0.0769804, 0.0579852, 0.0432905, 0.0303957, 0.0224979, 0.0167989, 0.0124095, 0.0089098, 0.0061099, 0.00394, 0.0026, 0.0019, 0.00148, 0.0012,0.000998,0.000849]
spinodals = [0.1177716, 0.0985767, 0.0759821, 0.0576858, 0.0452885, 0.0334943, 0.0256969, 0.0198983, 0.0152991, 0.0115995, 0.0084998, 0.0058999, 0.0039, 0.0026, 0.0019, 0.0015, 0.0012, 0.001, 0.0009]
plt.scatter(fA,spinodals)
plt.plot(fA,spinodals,label=r"$N$ = 20, diblock copolymer")
#blockiness = [i for i in range(17)][:9]
#trial1 = np.array([0.075581, 0.0864776, 0.0950742, 0.1036708, 0.107369, 0.1170641, 0.1203622, 0.1189629, 0.1211619, 0.1222614, 0.1193627, 0.1115668, 0.1296569, 0.1013718, 0.0931751, 0.0865777, 0.0777806][:9])
#trial2 = np.array([0.075581, 0.0822789, 0.0941747, 0.1025711, 0.1094681, 0.1170641, 0.120762, 0.1195629, 0.12466, 0.1233604, 0.1163644, 0.1117669, 0.1102675, 0.1008719, 0.0944746, 0.0862775, 0.0778803][:9])
#trial3 = np.array([0.0780804, 0.0865777, 0.0943745, 0.1011717, 0.111267, 0.1167643, 0.1178638, 0.1218615, 0.1232608])
#res = np.vstack((trial1,trial2,trial3))
#spinodals = np.mean(res,axis=0)
#spinodals = [0.0578859, 0.0626846, 0.0711823, 0.0775805, 0.0826788, 0.0875773, 0.0925754, 0.0939749, 0.0939749]
#blockiness = [i for i in range(1,5)]
#spinodals = [0.1018714, 0.1018714, 0.1009719, 0.0938749]
#plt.scatter(blockiness,spinodals)
#plt.plot(blockiness,spinodals,label=r"$N$ = 20, $f_A$ = 0.2")
'''
plt.scatter(blockiness2,spinodals2)
plt.plot(blockiness2,spinodals2,label=r"$N$ = 20, $f_A$ = 0.2")

plt.scatter(blockiness3,spinodals3)
plt.plot(blockiness3,spinodals3,label=r"$N$ = 20, $f_A$ = 0.4")

plt.scatter(blockiness1,spinodals1)
plt.plot(blockiness1,spinodals1,label=r"$N$ = 20, $f_A$ = 0.6")

plt.xlabel("Relative Blockiness")
plt.ylabel(r"$w^*_{poly}$")
'''

#plt.scatter(dop,spinodals)
#plt.plot(dop,spinodals,label=r"-(AGGG)-")
plt.legend()
plt.xlabel(r"$f_A$")
#plt.xlabel("Start index of A block")
plt.ylabel(r"$w^*_{poly}$")
plt.savefig("diblock_N20.png")
plt.show()