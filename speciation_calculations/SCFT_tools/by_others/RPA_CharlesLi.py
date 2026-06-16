from __future__ import division, print_function

import argparse
import time
import warnings

from scipy.constants import R
import networkx as nx
import numpy as np

from cgfts.forcefield.forcefield_v2 import ForceField


# parse input
parser = argparse.ArgumentParser()
parser.add_argument('-N', type=int, default=100)
parser.add_argument('-dN', type=int, default=1)
parser.add_argument('-N_ba_start', type=int, default=0)
parser.add_argument('-N_ba_end', type=int, default=100)
parser.add_argument('-sl', type=int, default=3)
parser.add_argument('-dphi', type=float, default=0.01)
parser.add_argument('-phi_start', type=float, default=0.0)
parser.add_argument('-phi_end', type=float, default=1.0)
parser.add_argument('-N_refine', '--num_refine', type=int, default=5)
parser.add_argument('-T', '--temperature', type=float, default=313.15)
parser.add_argument('-k_start', type=float, default=0.001)
parser.add_argument('-k_end', type=float, default=5.000)
parser.add_argument('-N_k', type=int, default=1000)
parser.add_argument('-fn', '--filename', type=str, default=None)
args = parser.parse_args()

# system information
smear_length = 0.5
bead_types = ["Bpba", "Bpla", "D4"]

# import sim ff file
kT = args.temperature * R / 1000
ff = ForceField.from_sim_ff_file("../../../forcefields/{}_ff.dat".format(args.temperature), kT=kT)
ff.reorder_bead_types(bead_types)

# bond lengths
#b_11 = b_12 = b_22 = 3.8718e-01
#b_33 = 4.5977e-01
b_11 = b_12 = b_22 = ff.get_pair_potential("Bonded", "Bpba", "Bpla").Dist0.value
b_33 = ff.get_pair_potential("Bonded", "D4", "D4").Dist0.value

# excluded volume matrix
# TODO: implement way to import sim ff file
u0 = np.empty((len(bead_types), len(bead_types)))
#u0[0, 0] = 1.36954889865
#u0[1, 1] = 0.734290131019
#u0[2, 2] = 2.84325396156
#u0[0, 1] = u0[1, 0] = 1.23522137149
#u0[0, 2] = u0[2, 0] = 2.10997435466
#u0[1, 2] = u0[2, 1] = 1.82522309477
u0[0, 0] = ff.get_pair_potential("Gaussian", "Bpba", "Bpba").excl_vol.value 
u0[1, 1] = ff.get_pair_potential("Gaussian", "Bpla", "Bpla").excl_vol.value
u0[2, 2] = ff.get_pair_potential("Gaussian", "D4", "D4").excl_vol.value
u0[0, 1] = u0[1, 0] = ff.get_pair_potential("Gaussian", "Bpba", "Bpla").excl_vol.value
u0[0, 2] = u0[2, 0] = ff.get_pair_potential("Gaussian", "Bpba", "D4").excl_vol.value
u0[1, 2] = u0[2, 1] = ff.get_pair_potential("Gaussian", "Bpla", "D4").excl_vol.value


def create_diblock_graph(N_ba, N_la):
    """Returns a graph representation of a
    butyl acrylate-block-lauryl acrylate diblock copolymer.

    Parameters
    ----------
    N_ba : int
        Number of butyl acrylate monomers.
    N_la : int
        Number of lauryl acrylate monomers.

    Returns
    -------
    G : networkx.Graph
        Graph representation of diblock copolymer.
    """
    # initialize graph
    G = nx.Graph()

    # add butyl acylate monomers
    for i in range(N_ba):
        G.add_node(i*2, name="Bpba")
        G.add_node(i*2 + 1, name="D4")
        G.add_edge(i*2, i*2 + 1, weight=b_33)
        if i > 0:
            G.add_edge(i*2, (i-1)*2, weight=b_11)

    # add lauryl acrylate monomers
    for i in range(N_la):
        G.add_node(N_ba*2 + i*4, name="Bpla")
        for j in range(3):
            G.add_node(N_ba*2 + i*4 + j + 1, name="D4")
            G.add_edge(N_ba*2 + i*4 + j, N_ba*2 + i*4 + j + 1, weight=b_33)
        if N_ba != 0:
            if i == 0:
                G.add_edge(N_ba*2 + i*4, (N_ba - 1)*2, weight=b_12)
            else:
                G.add_edge(N_ba*2 + i*4, N_ba*2 + (i - 1)*4, weight=b_22)

    return G


def create_dodecane_graph(num_beads=3):
    """Returns a graph representation of dodecane solvent.

    Returns
    -------
    G : networkx.Graph
        Graph representation of dodecane.
    """
    G = nx.Graph()
    for i in range(num_beads):
        G.add_node(i, name="D4")
        if i > 0:
            G.add_edge(i, i-1, weight=b_33)
    return G


def compute_form_factor(G, k):
    """
    Computes the intramolecular form factor for a chain.

    Parameters
    ----------
    G : networkx.Graph
        Graph representation of chain.
    k : array_like
        List of wavenumbers.

    Returns
    -------
    g : ndarray
        Array that has a form factor matrix for each wavemumber contained in k.
    """
    # initialize form factor array
    g = np.zeros((len(k), len(bead_types), len(bead_types)))

    # precompute square of k array
    k_squared = np.array(k) ** 2

    # use Dijkstra's algorithm to find path between all bead pairs
    for i, path_dict in nx.all_pairs_dijkstra_path(G):
        for j, path in path_dict.items():

            # get bead types from the nodes and determine type index
            bead_type_i = G.nodes[i]['name']
            bead_type_j = G.nodes[j]['name']
            bead_type_index_i = bead_types.index(bead_type_i)
            bead_type_index_j = bead_types.index(bead_type_j)

            # compute sum of square bond lengths along path
            sum_bond_lengths_squared = 0.0
            for m in range(len(path) - 1):
                bond_length = G.edges[path[m], path[m+1]]['weight']
                sum_bond_lengths_squared += bond_length * bond_length

            # add contribution to the form factor
            exponent = -k_squared * sum_bond_lengths_squared / 6.
            g[:, bead_type_index_i, bead_type_index_j] += np.exp(exponent)

    # nomalize by number of bead pairs
    g /= len(G) ** 2

    return g


def create_interaction_matrix(k):
    """
    Computes the interaction matrix for a given list of wavenumbers.

    Parameters
    ----------
    k : array_list

    Returns
    -------
    U : ndarray
        Array that has the interaction matrix computed for each wavenumber
        contained in k.
    """
    # initialize array
    U = np.empty((len(k), len(bead_types), len(bead_types)))

    # precompute square of k
    k_squared = np.array(k) ** 2

    # iterate through each pair of bead types
    for i in range(len(bead_types)):
        for j in range(len(bead_types)):
            U[:, i, j] = u0[i, j] * np.exp(-smear_length ** 2 * k_squared)

    return U


def compute_density_from_bead_volumes(T, N_ba, N_la, phi):
    """
    Computes overall system density from volumes of individual beads.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    N_ba : int
        Number of butyl acrylate monomers.
    N_la : int
        Number of lauryl acrylate monomers.
    phi : float
        Volume fraction of diblock copolymer in solution.

    Returns
    -------
    bead_density : float
        Overall bead density of diblock copolymer solution.
    """
    # bead volumes
    v_Bpba = {313.15: 0.0782090246379839,
              333.15: 0.07708724836107002,
              353.15: 0.07523024592003665,
              373.15: 0.0728}[T] 
    v_Bpla = v_Bpba
    #v_Bpla = 0.04511991500559438
    v_D4 = {313.15: 0.12685299840838823,
            333.15: 0.12955853799860306,
            353.15: 0.13224148529469396,
            373.15: 0.13499867126357457}[T]

    N_beads_diblock = N_ba * 2 + N_la * 4
    n_diblock = phi / N_beads_diblock
    n_dodecane = (1 - phi) / 3
    v_diblock = N_ba * (v_Bpba + v_D4) + N_la * (v_Bpla + 3 * v_D4)
    v_dodecane = 3 * v_D4
    return 1. / (n_diblock * v_diblock + n_dodecane * v_dodecane)


def determine_stability(form_factor_diblock, form_factor_dodecane, phi, c_chain_density):
    """
    Determines whether of not the DIS phase is stable. If DIS is unstable,
    this function also determines the type of instability.

    Parameters
    ----------
    form_factor_diblock : array_like
        Graph representation of diblock.
    form_factor_dodecane : array_like
        Graph representation of dodecane
    phi : float
        Diblock volume fraction
    c_chain_density : float
        Overall bead density.

    Returns
    -------
    out : int
        Indicator of the type of stability or instability of the DIS phase.
            0 = stable
            1 = macrophase instability
            2 = microphase instability
    """
    # compute ideal gas structure factor
    rho_diblock = c_chain_density * phi
    rho_dodecane = c_chain_density * (1 - phi)
    S0_diblock = rho_diblock * len(G_diblock) * form_factor_diblock
    S0_dodecane = rho_dodecane * len(G_dodecane) * form_factor_dodecane
    S0 = S0_diblock + S0_dodecane

    try:
        # compute S^-1
        Sinv = U + np.linalg.inv(S0)

        # compute det(S^-1)
        det_Sinv = np.linalg.det(Sinv)

        # check for instability of dis phase
        is_negative = det_Sinv <= 0.0
        if np.any(is_negative):
            if det_Sinv[0] <= 0.0:
                return 1
            else:
                return 2
        return 0

    except np.linalg.LinAlgError:
        warning_msg = "matrix is singular for N={}, N_ba={}, " \
                      "phi={}, density={}; assuming that DIS " \
                      "phase is stable".format(args.N, N_ba, phi,
                                               c_chain_density)
        warnings.warn(warning_msg)
        return 0


if __name__ == '__main__':

    start_time = time.time()

    # determine filename 
    if args.filename is None:
        filename = "rpa.txt"
    else:
        filename = args.filename

    # create list of wavenumbers
    k = np.linspace(args.k_start, args.k_end, args.N_k)

    # # check if 1 is integer multiple of phi increment
    # if not (1.0 / args.dphi).is_integer():
    #     err_msg = "1.0 is not an integer multiple of dphi={}".format(args.df)
    #     raise ValueError(err_msg)

    # initialize dictionary to store spinodal transitions
    spinodal_transitions = []

    # iterate through diblock composition
    for N_ba in range(args.N_ba_start, args.N_ba_end + args.dN, args.dN):

        # number of butyl acrylate monomers
        N_la = args.N - N_ba

        # create graphs for diblock and dodecane
        G_diblock = create_diblock_graph(N_ba, N_la)
        G_dodecane = create_dodecane_graph(num_beads=args.sl)

        # compute form factors for each chain
        form_factor_diblock = compute_form_factor(G_diblock, k)
        form_factor_dodecane = compute_form_factor(G_dodecane, k)

        # compute interaction matrix
        U = create_interaction_matrix(k)

        # remove components from form factors and U if we have a homopolymer
        if N_ba == 0 or N_la == 0:
            if N_ba == 0:
                delete_index = 0
            else:
                delete_index = 1
            form_factor_diblock = np.delete(form_factor_diblock, delete_index, axis=1)
            form_factor_diblock = np.delete(form_factor_diblock, delete_index, axis=2)
            form_factor_dodecane = np.delete(form_factor_dodecane, delete_index, axis=1)
            form_factor_dodecane = np.delete(form_factor_dodecane, delete_index, axis=2)
            U = np.delete(U, delete_index, axis=1)
            U = np.delete(U, delete_index, axis=2)

        # initialize variables for determining whether or not a transition occurs
        # 0 = stable, 1 = macrophase instability, 2 = microphase instability
        prev_dis_stability = 0
        curr_dis_stability = None
        prev_phi = 0.0
        curr_phi = None

        # iterate through phi
        for i_phi in range(int((args.phi_end - args.phi_start) / args.dphi) + 1):

            # calculated phi
            phi = args.phi_start + i_phi * args.dphi
            if phi == 0.0:
                continue

            # get bead density
            c_chain_density = compute_density_from_bead_volumes(args.temperature, N_ba, N_la, phi)

            # determine stability
            curr_dis_stability = determine_stability(form_factor_diblock, form_factor_dodecane, phi, c_chain_density)

            # check if spinodal transition occurs
            if curr_dis_stability != prev_dis_stability:

                # refine boundary
                phi_a = phi - args.dphi
                phi_b = phi
                for i_refine in range(args.num_refine):
                    dphi = args.dphi / (10 ** i_refine)
                    for i_phi_refine in range(int((phi_b - phi_a) / dphi) + 1):
                        phi = phi_a + i_phi_refine * dphi
                        c_chain_density = compute_density_from_bead_volumes(args.temperature, N_ba, N_la, phi)
                        stability = determine_stability(form_factor_diblock, form_factor_dodecane, phi, c_chain_density)
                        if stability == curr_dis_stability:
                            phi_a = phi - dphi
                            phi_b = phi
                            break

                # store spinodal transition
                spinodal_transitions.append((N_ba, phi_b, prev_dis_stability, curr_dis_stability))
                with open(filename, 'a') as f:
                    f.write("{} {} {} {}\n".format(*spinodal_transitions[-1]))
                print(N_ba, phi_b, prev_dis_stability, curr_dis_stability)
                prev_dis_stability = curr_dis_stability

#    # print results to file
#    with open(filename, 'w') as f:
#        for st in spinodal_transitions:
#            print("{} {} {} {}".format(*st), file=f)

    end_time = time.time()

    print("RPA calculation took {} s".format(end_time - start_time))
