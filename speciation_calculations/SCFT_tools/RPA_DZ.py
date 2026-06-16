from __future__ import division, print_function

import argparse
import time
import warnings
import sys

from scipy.constants import R
from scipy.special import gamma,gammaincc
import progressbar
from tqdm import tqdm
import networkx as nx
import numpy as np

from SCFT_tools import ParseSimFF
from matplotlib import pyplot as plt

class RPAsystem():


	def __init__(self):
		
		self.ff_file = None # Sim input file
		self.out = None # output file name
		self.bondtype = None # specify bondtype as either offset or DGC

		# wavenumber discretization
		self.k_start = 0.0001
		self.k_end = 10.0
		self.n_k = 10000

		# bond transition probability discretization 
		self.r_max = 1.0
		self.n_r = 1000

		# system chemistry
		self.beadtypes = []
		self.moltypes = []
		self.molbonds = []
		self.molweights = []
		self.molvols = []

		# system compositions
		self.comps = []
		self.rho_species = []

		# force field matrices
		self.k0_matrix = np.zeros((len(self.beadtypes),len(self.beadtypes)))
		self.r0_matrix = np.zeros((len(self.beadtypes),len(self.beadtypes)))
		self.b0_matrix = np.zeros((len(self.beadtypes),len(self.beadtypes)))
		self.u0_matrix = np.zeros((len(self.beadtypes),len(self.beadtypes)))
		self.lb_matrix = np.zeros((len(self.beadtypes),len(self.beadtypes)))
		
		# used internally
		self.molgraphs = {}
		self.formfactor = {}

		return
	

	def CreateMoleculeGraph(self, nodes, edges):
		
		"""
		Returns a graph representation of CG molecules.

		Parameters
		----------
		nodes : ndarray
			Topology of molecules in terms of defined bead types.
		edges : ndarray
			List of bond pairs for defined molecules.  
				
		Returns
		-------
		G : networkx.Graph
			Graph representation of molecules.
		"""
		
		G = nx.Graph()

		for i in range(len(nodes)):
			G.add_node(i, name=nodes[i])

		for _e in edges:

			bead_type_index_i = int(G.nodes[_e[0]]['name'])
			bead_type_index_j = int(G.nodes[_e[1]]['name'])
			
			bond_params = (self.r0_matrix[bead_type_index_i][bead_type_index_j], self.k0_matrix[bead_type_index_i][bead_type_index_j])
			G.add_edge(_e[0], _e[1], params=bond_params)

		return G


	def ComputeChainDensity(self, comp):
		
		mol_w = np.array([mw[1] for mw in self.molweights])
		mol_v = np.array([mv[1] for mv in self.molvols]) 
		mol_l = np.array([len(mg.nodes) for mg in self.molgraphs.values()])

		if False in [i==0 for i in mol_w]:	
			n_mols = comp / mol_w
			total_beads = np.sum(n_mols * mol_l)
			total_vol = np.sum(n_mols * mol_v)
			bead_fractions = (n_mols * mol_l) / total_beads
			c_chain_density = total_beads / total_vol

		else:
			n_mols = comp / mol_l
			total_vol = np.sum(n_mols * mol_v)
			bead_fractions = comp
			c_chain_density = 1. / total_vol

		rho_species = c_chain_density * bead_fractions / mol_l

		return c_chain_density, bead_fractions, rho_species


	def CreateInteractionMatrix(self):

		# initialize array
		Ugauss = np.zeros((len(self.k_vector), len(self.beadtypes), len(self.beadtypes)))
		Ue = np.zeros((len(self.k_vector), len(self.beadtypes), len(self.beadtypes)))

		# precompute square of k array
		k_squared = self.k_vector ** 2
		

		for i,bi in enumerate(self.beadtypes):
			for j,bj in enumerate(self.beadtypes):

				kappa = self.kappa_matrix[i][j]
				Ugauss[:,i,j] = self.u0_matrix[i][j] * np.exp(-(2 * kappa)**(-1) * k_squared / 2)
			
				zi = bi[2]
				zj = bj[2]
				Ue[:,i,j] += zi * zj * 4 * np.pi * self.lb_matrix[i][j] * (1 / k_squared) * np.exp(-(2 * kappa)**(-1) * k_squared / 2)         
		
		self.U = Ugauss + Ue
		return 


	def ComputeBondTransitionDGC(self, k0):

		k_squared = self.k_vector ** 2
		b0 = np.sqrt(1.5 / k0)
		exponent = -k_squared * b0 * b0 / 6.
		Phi = np.exp(exponent)

		return Phi


	def ComputeBondTransitionOffset(self, r0, k0, r_max=1., n_r=10000):

		k_squared = self.k_vector**2

		dr = r_max / n_r
		r_right = np.linspace(dr, r_max, n_r, endpoint=True)
		r_left = r_right - dr
		r_mid = r_right - 0.5 * dr

		# compute exp of bond potential
		U_bond = k0 * (r_mid - r0)**2 
		exp_U_bond = np.exp(-U_bond)

		Phi = np.zeros(len(self.k_vector))
		for i in range(n_r):
			w = np.cos(self.k_vector * r_left[i]) - np.cos(self.k_vector * r_right[i])
			Phi += r_mid[i] * exp_U_bond[i] * w

		# compute normalization
		U = k0 * (r_right - r0)**2
		exp_U = np.exp(-U)
		weights = np.ones(len(r_right))
		weights[0] = 55. / 24.
		weights[1] = -1. / 6. 
		weights[2] = 11. / 8. 
		weights[-3] = 23. / 24. 
		weights[-2] = 7. / 6. 
		weights[-1] = 3. / 8. 
		norm = dr * np.sum(r_right**2 * exp_U * weights) 
		Phi = Phi / (norm * k_squared)

		return Phi 


	def GetBondTransitions(self):
		
		k_squared = self.k_vector**2

		# initialize bond transition array 
		Phi_array = np.zeros((len(self.k_vector), len(self.beadtypes), len(self.beadtypes)))

		for i,bi in enumerate(self.beadtypes):
			for j,bj in enumerate(self.beadtypes):
				k0 = self.k0_matrix[i][j]
				r0 = self.r0_matrix[i][j]

				# not an existing bond pair
				if k0 == r0 == 0.0:
					Phi = 0.0

				# use DGC statistics
				elif self.bondtype == 'DGC':
					Phi = self.ComputeBondTransitionDGC(k0)
				
				# use offset harmonic bonds
				elif self.bondtype == 'offset':
					Phi = self.ComputeBondTransitionOffset(r0, k0, r_max=self.r_max, n_r=self.n_r)

				Phi_array[:, i, j] = Phi

		return Phi_array
	

	def ComputeFormFactor(self, graph):

		# initialize form factor array
		S = np.zeros((len(self.k_vector), len(self.beadtypes), len(self.beadtypes)))

		# precompute square of k array
		k_squared = self.k_vector ** 2
		curr = 0
		# use Dijkstra's algorithm to find path between all bead pairs
		for i, path_dict in nx.all_pairs_dijkstra_path(graph):
			for j, path in path_dict.items():
				# get bead types from the nodes and determine type index
				bead_type_index_i = int(graph.nodes[i]['name'])
				bead_type_index_j = int(graph.nodes[j]['name'])

				# compute propagator
				propagator = 1.
				#print(j, path)
				for m in range(len(path) - 1):
					index1 = int(graph.nodes[path[m]]['name'])
					index2 = int(graph.nodes[path[m+1]]['name'])
					propagator *= self.Phi_array[:,index1,index2]
					if np.sum(propagator) == 0.:
						raise Exception('Propagator is 0.0 at bond pair ({}, {})'.format(index1, index2))

				S[:, bead_type_index_i, bead_type_index_j] += propagator
		return S
			

	def GetSkInv(self, comp):

		c_chain_density, bead_fractions, rho_species = self.ComputeChainDensity(comp) # c_chain_density = #beads all/vol, bead_fractions = #i beads/#total beads, rho_species = # chains i/vol
		self.rho_species.append(rho_species)

		S0 = 0.0
		for j,(molname,graph) in enumerate(self.molgraphs.items()):
	
			S0 += c_chain_density * bead_fractions[j] * (1. / len(graph.nodes)) * self.formfactor[molname]
			
		try:
			# compute S^-1
			Sinv = self.U + np.linalg.inv(S0)

			# compute det(S^-1)
			det_Sinv = np.linalg.det(Sinv)
			# check for instability of dis phase
			is_negative = det_Sinv <= 0.0
			if np.any(is_negative):
				if det_Sinv[0] <= 0.0:
					stability_status = 1
				else:
					stability_status = 2
			else: 
				stability_status= 0

		except np.linalg.LinAlgError:
			warning_msg = 'matrix is singular for composition fractions {} and chain density {} \
								... omitting point in analysis\n'.format(comp, c_chain_density)
			warnings.warn(warning_msg)
			stability_status = -1
			return None, None, stability_status
		return det_Sinv, Sinv, stability_status


	def ComputeMatrixDeterminants(self):
		
		stability_status = []

		for i in tqdm(range(len(self.comps)), desc='Composition Points'):
			det_Sinv, Sinv, ss = self.GetSkInv(self.comps[i])
			stability_status.append(ss)
		return stability_status


	def Initialize(self):

		if self.beadtypes == [] or self.molvols == []:
			raise Exception('comps, beadtypes, and molvols all must be specified attributes')

		if self.ff_file is not None:
			self.ff = ParseSimFF.SimForceField(self.ff_file, self.beadtypes)
			self.ff.CreateFFDictionary()
			self.ff.CreateExcludedVolumeMatrix()
			self.ff.CreateBondMatrix()
			self.ff.CreateEwaldMatrix()

			self.kappa_matrix = self.ff.kappa_matrix
			self.u0_matrix = self.ff.excluded_volume_matrix
			self.r0_matrix = self.ff.Dist0_matrix
			self.k0_matrix = self.ff.FConst_matrix
			self.lb_matrix = self.ff.lb_matrix
			

		self.comps = np.asarray(self.comps)
		self.k_vector = np.linspace(self.k_start, self.k_end, self.n_k)

		self.CreateInteractionMatrix()


		if self.molgraphs == {}:
			print('\n========== Creating Molecular Graphs ==========\n')
			if self.moltypes == [] or self.molbonds == []:
				raise Exception('must specify moltypes and molbonds to create molecular graphs')

			for i in range(len(self.moltypes)):
				g = self.CreateMoleculeGraph(nodes=self.moltypes[i][1], edges=self.molbonds[i][1])
				self.molgraphs[self.moltypes[i][0]] = g
			#print("ye",self.molgraphs['Chain'])
			#nx.draw(self.molgraphs['Chain'])
			#plt.show()
		for molname,graph in self.molgraphs.items():
			s = '{} :\n'.format(molname)
			for i,n in graph.nodes.data('name'):
				s += ' {} '.format(int(n))
			s += '\n'
			for (u, v, w) in graph.edges.data('params'):       	
				s += ' {} - {} - {}\n'.format(int(u),w,int(v))
			print(s)
		if self.molweights == [] or self.molweights is None: 
			print('\n...assuming compositions are bead fractions...\n')
			self.molweights = [[key, 0.0] for key,value in self.molgraphs.items()]
		else:
			print('\n...assuming compositions are mass fractions...\n')
		
		# automatic detection of bond types
		if np.sum(self.r0_matrix == 0.0) and self.bondtype is None:
			self.bondtype = 'DGC'
		elif self.bondtype is None:
			self.bondtype = 'offset'

		self.Phi_array = self.GetBondTransitions()
		for molname,graph in self.molgraphs.items():
			self.formfactor[molname] = self.ComputeFormFactor(graph)
		return


	def MapStabilityStatus(self):

		self.Initialize()

		stability_data = []
		# stability_status 0:stable, 1:macophase instability, 2:mesophase instability
		
		print('\n========== Calculating Matrix Determinants ==========\n')
		stability_status = self.ComputeMatrixDeterminants()
		
		for i,ss in enumerate(stability_status):
			if ss != -1:
				comp = self.comps[i].tolist()
				rho = self.rho_species[i].tolist()
				stability_data.append(comp + rho + [ss])

		if self.out is None:
			prefix = 'rpa'
		else:
			prefix = '{}_rpa'.format(self.out)

		f = open('{}.txt'.format(prefix),'w')

		s = '# weight fractions/chain concentrations : '
		for key in self.molgraphs.keys():
			s += '{}   '.format(key)

		f.write('{}\n'.format(s))
		f.write('# 0:stable, 1:macophase instability, 2:mesophase instability\n')
		
		for p in stability_data:
			for i in p[0:-1]:
				f.write(f"{format(i, '.10f'):12.10}")
			f.write("{}\n".format(p[-1]))

		np.save('{}.npy'.format(prefix), np.asarray(stability_data, dtype=object))
		
		return
	
	
	def TernarySweep(self, edges=[0,1,2], x_min=0.001, x_max=0.999, nx=1000, dx_sweep=0.05, dx_tol = 0.001):

		self.Initialize()

		mol_w = np.array([mw[1] for mw in self.molweights])
		mol_l = np.array([len(mg.nodes) for mg in self.molgraphs.values()])
		x_vector = np.linspace(x_max, x_min, num=nx, endpoint=True)
		comps = np.zeros(len(self.Graphs))
		edge_comps = np.zeros(len(3))
		all_spinodal_points = []

		for ie in range(len(edges)):
		
			e = edges[ie]
			spinodal_points = []

			for ix in tqdm(range(len(x_vector)), desc='Sweeping across edge {}'.format(ie)):

				x = x_vector[ix]
				x_sweep = x_min
				dx_sweep = 0.1

				while True:
					
					edge_comps[e] = x
					edge_comps[e-1] = (1 - x) * x_sweep
					edge_comps[e-2] = (1 - x) * (1 - x_sweep)
				
					for iec,ec in enumerate(edge_comps):

						if isinstance(edges[iec], list):
							if False in [i==0 for i in mol_w]:
								for i in edges[iec]:
									comps[i] = ec * mol_w[i] / np.sum(mol_w[edges[iec]])
							else: 
								for i in edges[iec]:
									comps[i] = ec * mol_l[i] / np.sum(mol_l[edges[iec]])
						else:
							comps[e] = ec 

					det_Sinv, Sinv, ss = self.GetSkInv(comps)

					if ss == 0:
						x_sweep += dx_sweep
					else:
						if dx_sweep <= dx_tol:
							spinodal_points.append(edge_comps + [ss])
							break
						dx_sweep /= 2.
						x_sweep -= dx_sweep

			all_spinodal_points.append(spinodal_points)
		
		if self.out is None:
			prefix = 'rpa'
		else:
			prefix = '{}_rpa'.format(self.out)

		np.save('{}.npy'.format(prefix), np.asarray(all_spinodal_points, dtype=object))

		return



						






		
		
