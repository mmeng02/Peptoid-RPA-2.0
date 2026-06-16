import os
import sys
import re
import time
import shutil
import numpy as np
from subprocess import call

import SCFT_tools


class Gibbs_Sytem():

    def __init__(self):
    
        # example how to initialize class attributes for SDS/water system

        self.num_models = None

        self.num_species = None

        self.temp_files = ['temp_dis.in',
                           'temp_mic.in']
                           
        self.run_files = ['input_dis.in',
                          'input_mic.in']
        
        self.field_files = ['fields_k_dis.bin',
                            'fields_k_mic.bin']

        self.cell_scaling = [4, 4]

        self.NPW = [128, 1]

        self.species = ['W', 'Na', 'DS']

        self.neutral_species = [['Na'] + ['DS'],
                                ['W']]

        self.mollengths = [1, 1, 7]

        self.molweights = [18.01528, 22.98977, 265.39023]
    
        self.molvolumes = [0.0307, 0.1, 0.3042] 
    
        self.phi = [[0.95, 0.5*(7/8), 0.5*(1/8)],
                     [0.95, 0.5*(7/8), 0.5*(1/8)]]

        self.f = [0.5, 0.5]

        # Gibbs partition time stepping parameters.

        self.dtCtot = 0.01
        self.dtf = 0.001
        self.dtC = 0.1
        self.steps = 1000

        self.use_chain_density = None
        self.pressure_match = False
        self.volume_swap = True
        self.density_swap = True

        self.Ptarget = 283.951857
        self.pfts = 'PolyFTS.x'

        return


    def Set_Interal_Attributes(self):

        # attributes used internally

        self.P_error = 1.0
        self.miu_error = []
        self.P_tolerance = 1.e-3
        self.miu_tolerance = 1.e-3
        self.converged = False
    
        if self.num_models is None:
            self.num_models = len(self.phi)
        if self.num_species is None:
            self.num_species = len(self.species)
        if self.mollengths == []:
            self.mollengths = [1 for i in range(self.num_species)]
        if self.molweights == []:
            self.molweights = [0 for i in range(self.num_species)]
        if self.molvolumes == []:
            self.molvolumes = [1 for i in range(self.num_species)]

        self.labels = np.asarray(self.species[0:self.num_species])
        self.moll = np.asarray(self.mollengths[0:self.num_species])
        self.molw = np.asarray(self.molweights[0:self.num_species])
        self.molv = np.asarray(self.molvolumes[0:self.num_species])

        self.f = np.asarray(self.f)
        self.C = np.zeros((self.num_models))
        self.P = np.zeros((self.num_models))
        self.H = np.zeros((self.num_models))
        self.Ctot_species = np.zeros((len(self.species)))
        self.Ctot_neutral = np.zeros((len(self.neutral_species)))
        self.C_species = np.zeros((self.num_models, self.num_species))
        self.C_neutral = np.zeros((self.num_models, len(self.neutral_species)))
        self.miu = np.zeros((self.num_models, self.num_species))
        self.miu_neutral = np.zeros((self.num_models, len(self.neutral_species)))
        self.vol_fractions = np.zeros((self.num_models, self.num_species))

        return
    

    def Gibbs_Report(self, n):
        
        if n == 0:
            self.GibbsFile = open('Gibbs.dat','w')
            self.GibbsFile.write('# step C1 C2 f1 f2 P1 P2 mu1 mu2\n')  
            self.GibbsFile.flush()

            self.CFile = open('GibbsC.dat','w')
            self.CFile.write('# step C_species_tot C_species C_neutral\n')  
            self.CFile.flush()

            self.ErrorFile = open('GibbsErrors.dat', 'w')
            self.ErrorFile.write('# step P_error, miu_error\n')  
            self.ErrorFile.flush()

        s = ''
        s += f"{format(n):<6}"
        for i in self.C:
            s += f"{format(i, '.10f'):<12.10}"
        for i in self.f:
            s += f"{format(i, '.10f'):<12.10}"
        for i in self.P:
            s += f"{format(i, '.10f'):<12.10}"
        for i in self.miu:
            for j in i:
                s += f"{format(j, '.10f'):<12.10}"
        s += '\n'
        self.GibbsFile.write(s) 
        self.GibbsFile.flush()

        self.CFile.write('\nGibbs Step: {}\n'.format(n))
        self.CFile.write('Ctot_species :     {}\n'.format(' '.join(str(np.round(i,8)) for i in self.Ctot_species)))
        self.CFile.write('Ctot_neutral :     {}\n'.format(' '.join(str(np.round(i,8)) for i in self.Ctot_neutral)))
        self.CFile.write('Model1 C_species : {}\n'.format(' '.join(str(np.round(i,8)) for i in self.C_species[0])))
        self.CFile.write('Model2 C_species : {}\n'.format(' '.join(str(np.round(i,8)) for i in self.C_species[1])))
        self.CFile.write('Model1 C_neutral : {}\n'.format(' '.join(str(np.round(i,8)) for i in self.C_neutral[0])))
        self.CFile.write('Model2 C_neutral : {}\n'.format(' '.join(str(np.round(i,8)) for i in self.C_neutral[1])))
        self.CFile.flush()
        
        if n > 0 :
            s = ''
            s += f"{format(n):<6}"
            s += f"{format(self.P_error, '.10f'):<12.10}"
            for i in self.miu_error:
                s += f"{format(i, '.10f'):<12.10}"
            s += '\n'
            self.ErrorFile.write(s)
            self.ErrorFile.flush()

        if self.converged == True:
            self.GibbsFile.close()
            self.CFile.close()

        return


    def Set_Total_Densities(self):

        self.Ctot_species = np.sum(self.C_species.T * self.f, axis = 1)
        self.Ctot_neutral = np.sum(self.C_neutral.T * self.f, axis = 1)
        self.Ctot = np.sum(self.Ctot_species)

        return


    def Set_Initial_Chain_Density(self):
        
        self.phi = np.asarray(self.phi)

        if self.use_chain_density is not None:
            print('\n...assuming compositions are volume fractions...\n')
            for i in range(self.num_models):
                self.vol_fractions[i] = self.phi[i]
                self.C[i] = self.use_chain_density 
                self.C_species[i] = self.C[i] * self.phi[i] / self.moll

        elif False in [i==0 for i in self.molw]:	
            print('\n...assuming compositions are mass fractions...\n')
            for i in range(self.num_models):
                n_mols = self.phi[i] / self.molw
                total_beads = np.sum(n_mols * self.moll)
                total_vol = np.sum(n_mols * self.molv)
                self.vol_fractions[i] = (n_mols * self.moll) / total_beads
                self.C[i] = total_beads / total_vol
                self.C_species[i] = self.C[i] * self.vol_fractions[i] / self.moll

        else:
            print('\n...assuming compositions are bead fractions...\n')
            for i in range(self.num_models):
                n_mols = self.phi[i] / self.moll
                total_vol = np.sum(n_mols * self.molv)
                self.vol_fractions[i] = self.phi[i]
                self.C[i] = 1. / total_vol
                self.C_species[i] = self.C[i] * self.vol_fractions[i] / self.moll
        
        return


    def Write_Input(self):
        
        for i in range(self.num_models):
            with open(self.temp_files[i],'r') as fi:
                if os.path.exists('./' + self.field_files[i]):
                    RIF = 'True'
                else: 
                    RIF = 'False'
                fid = fi.read()
                fid = fid.replace('<CChainDensity>',str(np.round(self.C[i], 12)))
                fid = fid.replace('<CellScaling>',str(self.cell_scaling[i]))
                fid = fid.replace('<NPW>',str(self.NPW[i]))
                fid = fid.replace('<InputFieldsFile>',str(self.field_files[i]))
                fid = fid.replace('<ReadInputFields>', RIF)
                for _l, _vf in zip(self.labels, self.vol_fractions[i]):
                    fid = fid.replace('<{}>'.format(_l), str(np.round(_vf, 12)))

            fw = open(self.run_files[i],'w')
            fw.write(fid)
            fw.close()
            fi.close()
        
        return
        

    def Run_SCFT(self):
        
        for i,f in enumerate(self.run_files):
            call('/home/djzhao/Packages/PolyFTS/bin/Release/{} {} > z.log'.format(self.pfts, f), shell=True)

            status = np.loadtxt('STATUS')
            if status != 2:
                print('Error: SCFT simulation did not converge for model {}'.format(i))
                sys.exit()
        
            lines =  open('operators.dat').readlines()
            header = lines[0].strip('# ')
   
            col_labels = header.split(' ')
            miu_cols = []
            for j,l in enumerate(col_labels):
                if re.search('ChemicalPotential',l):
                    miu_cols.append(j)

            datafile = np.loadtxt('operators.dat')
            data = datafile[-1]
            
            self.H[i] = data[1]
            self.P[i] = data[2]
            self.miu[i] = data[np.asarray(miu_cols)]
         
            shutil.copy('fields_k.bin', '{}'.format(self.field_files[i]))

        return 


    def Set_Neutral_Species_Chemical_Potentials(self):

        if self.neutral_species is None or self.neutral_species == []:
            self.miu_neutral = self.miu
        
        else:
            for i in range(self.num_models):
                self.miu_dict = {}
                for j, _l in enumerate(self.labels):
                    self.miu_dict[_l] = self.miu[i][j]
    
                for j, _ns in enumerate(self.neutral_species):
                    miu_sum = np.sum([self.miu_dict[_n] for _n in _ns])
                    self.miu_neutral[i][j] = miu_sum
    
        return 


    def Set_Neutral_Species_Densities(self):

        if self.neutral_species is None or self.neutral_species == []:
            self.C_neutral = self.C_species

        else:      
            for i in range(self.num_models):
                C_dict = {}
                for j, _l in enumerate(self.labels):
                    C_dict[_l] = self.C_species[i][j]


                for j, _ns in enumerate(self.neutral_species):
                    
                    # count number of occurances of unique molecule types in each neutral species definition
                    unique_ns = []
                    ns_occur = []
                    for _n in _ns:
                        if _n not in unique_ns:
                            unique_ns.append(_n)
                    for _n in unique_ns:
                        ns_occur.append(_ns.count(_n))
                    
                    # find the concentration limiting species in the neutral pair
                    C_min = np.min([C_dict[_n] / _o  for _n, _o in zip(unique_ns, ns_occur)])
                    self.C_neutral[i][j] = C_min
                    
                    for _n in unique_ns:
                        C_dict[_n] -= C_min

        return


    def Update_Compositions(self):

        # update input file volume fractions based on particle swaps on neutral species

        if self.neutral_species is None or self.neutral_species == []:
            self.C_species = self.C_neutral

        else:
            self.C_species = np.zeros((self.num_models, self.num_species))
            for i in range(self.num_models):
                for j in range(len(self.vol_fractions)):
                    for k,_ns in enumerate(self.neutral_species):
                        count = _ns.count(self.labels[j])
                        if _ns.count(self.labels[j]) > 0:
                            self.C_species[i][j] += count * self.C_neutral[i][k]

                self.vol_fractions[i] = (self.C_species[i] * self.moll) / np.sum(self.C_species[i] * self.moll)
            
        for i in range(len(self.C)):
            self.C[i] = np.sum(self.C_species[i] * self.moll)
    
        return


    def Gibbs_Step(self):

        # currently only implemented to work for two model systems
       
        if self.pressure_match:
            Ctot_new = self.Ctot - self.dtCtot * (self.P[0] - self.Ptarget)
            self.C = self.C * Ctot_new / self.Ctot
            self.C_species = self.C_species * Ctot_new / self.Ctot
            self.C_neutral = self.C_neutral * Ctot_new / self.Ctot
            self.Set_Total_Densities()
        
        if self.volume_swap:
            self.f[0] = self.f[0] + self.dtf * (self.P[0] - self.P[1])
            self.f[1] = 1 - self.f[0]

        if self.density_swap:
            for i,(_miu1, _miu2, _Cn1, _Cn2) in enumerate(zip(self.miu_neutral[0], self.miu_neutral[1],
                                                        self.C_neutral[0], self.C_neutral[1])):
                dtC_i = self.dtC * np.minimum(_Cn1, _Cn2)
                _Cn1_new = _Cn1 - dtC_i * (_miu1 - _miu2)
                self.C_neutral[0][i] = _Cn1_new
                self.C_neutral[1][i] = (self.Ctot_species[i] - self.f[0] *  _Cn1_new) / self.f[1]
        
        if np.any(self.f <= 0.0):
            print('Error: volume fraction for one of the phases has shrunk to 0.0')
            sys.exit()

        if np.any(self.C_neutral < 0.0): 
            print('Error: density of one of the species in one phase is negative')
            sys.exit()

        return


    def Check_Convergence(self):

        # currently set up for only two sytems

        tests = []

        if self.volume_swap:
            self.P_error =  np.abs(self.P[0] - self.P[1])
            tests.append(self.P_error <= self.P_tolerance)

        if self.density_swap:
            self.miu_error.clear()
            for _miu1, _miu2 in zip(self.miu_neutral[0], self.miu_neutral[1]):
                self.miu_error.append(np.abs(_miu1 - _miu2))
                tests.append(self.miu_error[-1] <= self.miu_tolerance)

        if all(tests) == True:
            self.converged = True

        return


    def Run_Gibbs(self):

        # calculate intial species densities in each cell and total system density
        self.Set_Interal_Attributes()
        self.Set_Initial_Chain_Density()
        self.Set_Neutral_Species_Densities()
        self.Set_Total_Densities()
        self.Gibbs_Report(0)
        
        for n in range(self.steps):
           
            # write input files
            self.Write_Input()
        
            # run SCFT simulations for models and collect values
            self.Run_SCFT()

            # set the chemical potentials of the defined neutral pairs
            self.Set_Neutral_Species_Chemical_Potentials()

            # do volume and particle swaps
            self.Gibbs_Step()
   
            # update volume fractions and chain densities for PolyFTS input file 
            self.Update_Compositions()

            # check convergence
            self.Check_Convergence()

            # write Gibbs iteration to file
            self.Gibbs_Report(n+1)

            if self.converged == True:
                print('Gibbs Ensemble converged in {} steps'.format(n+1))
                break

        return