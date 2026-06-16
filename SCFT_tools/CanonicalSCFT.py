import os
import sys
import re
import copy
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
from scipy.interpolate import CubicSpline
from scipy.interpolate import BarycentricInterpolator


class CanonicalSystem():

    def __init__(self):

        self.dis_tempfile = 'temp_dis.in'
        self.mic_tempfile = 'temp_mic.in'

        self.dis_runfile = 'input_dis.in'
        self.mic_runfile = 'input_mic.in'

        self.dis_seedfile = 'fields_k_dis.bin'
        self.mic_seedfile = 'fields_k_mic.bin'

        self.dis_RIF = True
        self.mic_RIF = True

        self.dis_NPW = 1
        self.mic_NPW = 64

        self.boxL = None

        self.chainlabels = []
        self.CMC_species = []

        self.lowerbound = []
        self.upperbound = []

        self.chainlengths = []
        self.molweights = []
        self.molvolumes = []
        
        self.scale = 0.5
        self.abstol = 1.e-2
        self.converged = False

        self.nblocks = None
        self.pfts = 'PolyFTSGPU.x'

        return
    

    def Initialize(self):

        self.dH_lb = None
        self.dH_ub = None

        self.H_mic = []
        self.H_dis = []
        self.miu_species = []
        self.rho_mic = []
        self.rho_dis = []
        self.C_species = []
        self.volfractions = []
        self.massfractions = []

        self.chainlabels = np.asarray(self.chainlabels)
        self.CMC_species = np.asarray(self.CMC_species)
        self.chainlengths = np.asarray(self.chainlengths)
        self.molweights = np.asarray(self.molweights)
        self.molvolumes = np.asarray(self.molvolumes)

        self.scol = []
        for _s in self.CMC_species:
            _sc = np.where(self.chainlabels==_s)[0][0]
            self.scol.append(_sc)

        return
    

    def CMCReport(self, step):
        
        if step == 0:
            self.CMCFile = open('CMC.dat','w')
            self.CMCFile.write('# step MassFrac Miu Rho_mic Rho_dis H_mic H_dis H_diff\n')  
            self.CMCFile.flush()

        s = ''
        s += f"{format(step):<6}"

        x = np.sum(np.asarray(self.massfractions)[self.scol])
        m = np.sum(np.asarray(self.miu_species)[-1,self.scol])
        rm = np.sum(np.asarray(self.rho_mic)[-1,self.scol])
        rd = np.sum(np.asarray(self.rho_dis)[-1,self.scol])

        s += f"{format(x, '.10f'):<12.10}"
        s += f"{format(m, '.10f'):<12.10}"
        s += f"{format(rm, '.10f'):<12.10}"
        s += f"{format(rd, '.10f'):<12.10}"

        s += f"{format(self.H_mic[-1], '.12f'):<14.12}"
        s += f"{format(self.H_dis[-1], '.12f'):<14.12}"
        s += f"{format((self.H_mic[-1] - self.H_dis[-1]), '.10f'):<12.10}"
        s += '\n'
        self.CMCFile.write(s) 
        self.CMCFile.flush()

        if self.converged == True:
            self.CMCFile.close()

        return
    

    def CMCPlot(self):

        x_rho = np.zeros(len(self.rho_dis))
        x_miu = np.zeros(len(self.miu_species))            
        y = np.asarray(np.asarray(self.H_mic) - np.asarray(self.H_dis))
       
        for _sc in self.scol:
            x_rho += np.asarray(self.rho_dis)[:,_sc]
            x_miu += np.asarray(self.miu_species)[:,_sc]

        x_rho_sort = np.sort(x_rho)
        x_miu_sort = np.sort(x_miu)
        y_sort = np.sort(y)[::-1]

        rho_interp = CubicSpline(x_rho_sort, y_sort)
        x_rho_range = np.linspace(x_rho_sort[0], x_rho_sort[-1], 1000, endpoint=True)
        
        fig = plt.figure(dpi=1000)
        plt.xlabel('$ρ_{dis} \ [nm^{-3}]$')
        plt.ylabel('$βΩ_{mic} - βΩ_{dis}$')
        plt.plot(x_rho, y, linestyle='none', marker='o', label='simulated points')
        plt.plot(x_rho_range, rho_interp(x_rho_range), label='interpolation')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig('CMC_rho_Plot.png')

        miu_interp = CubicSpline(x_miu_sort, y_sort)
        x_miu_range = np.linspace(x_miu_sort[0], x_miu_sort[-1], 1000, endpoint=True)
        
        fig = plt.figure(dpi=1000)
        plt.xlabel('$μ \ [k_BT]$')
        plt.ylabel('$βΩ_{mic} - βΩ_{dis}$')
        plt.plot(x_miu, y, linestyle='none', marker='o', label='simulated points')
        plt.plot(x_miu_range, miu_interp(x_miu_range), label='interpolation')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig('CMC_miu_Plot.png')
        return
    

    def SetBounds(self, step, phi):

        dH = self.H_mic[-1] - self.H_dis[-1]

        if dH < 0:
            if self.dH_ub is None or dH > self.dH_ub: 
                self.upperbound = phi
            self.dH_ub = dH

        elif dH > 0:
            if self.dH_lb is None or dH < self.dH_lb:
                self.lowerbound = phi
            self.dH_lb = dH

        elif step >= 1 and self.dH_ub == self.dH_lb:
            raise RuntimeError('Failed to bound CMC, upper bound and lower bound are the same value')
        
        #if step == 1 and (self.dH_ub is None or self.dH_lb is None):
        #    raise RuntimeError('Failed to find either an upper bound or lower bound for CMC')
        
        return 
    

    def InterpolatePhi(self):
        
        x_lb = 0
        x_ub = 0
        for _sc in self.scol:
            x_lb += self.lowerbound[_sc]
            x_ub += self.upperbound[_sc]

        if self.dH_lb is None:
            x_new = x_ub * self.scale
            print('... Looking for lower bound of CMC')
        
        elif self.dH_ub is None:
            x_new = x_lb  / self.scale
            print('... Looking for upper bound of CMC')
        
        else:
            fit = np.polyfit([x_lb, x_ub], [self.dH_lb, self.dH_ub], 1)
            x_new = np.abs(-fit[1]/fit[0])
            #x_new = np.abs(np.interp(0.0, [x_lb, x_ub], [self.dH_lb, self.dH_ub]))

        phi = np.zeros(len(self.chainlabels))
        for _sc in self.scol:
            phi[_sc] = x_new*(self.molweights[_sc]/np.sum(self.molweights[self.scol]))
        phi[-1] = 1.0 - x_new 

        self.massfractions = phi

        return phi


    def SetChainDensity(self, phi):
     	
        '''
        print('\n...reading in bead volume fractions...')
        vol_fracs = phi
        n_mols = phi / self.chainlengths
        total_vol = np.sum(n_mols * self.molvolumes)
        C = np.round(1. / total_vol, 12)
        C_species = C * phi / self.chainlengths
        '''

        print('\n...assuming compositions are mass fractions...')
        n_mols = phi / self.molweights
        total_beads = np.sum(n_mols * self.chainlengths)
        total_vol = np.sum(n_mols * self.molvolumes)
        vol_fracs = np.round((n_mols * self.chainlengths) / total_beads, 12)
        C = np.round(total_beads / total_vol, 12)
        C_species = C * vol_fracs / self.chainlengths
      
        self.C = C
        self.volfractions = vol_fracs.tolist()
        self.C_species = C_species.tolist()
        
        return

    
    def CheckConvergence(self):

        if np.abs(self.H_mic[-1] - self.H_dis[-1]) <= self.abstol:
            self.converged = True 
            self.CMCPlot()
            print('\n... CMC calculation has converged')

        return


    def CalculateCMC(self):
        
        self.Initialize()
        timestart = time.time()
       
        lb_guess = copy.deepcopy(self.lowerbound)
        ub_guess = copy.deepcopy(self.upperbound)
        step = 0

        while self.converged == False:

            if step == 0:
                phi = ub_guess
            elif step == 1:
                phi = lb_guess
            else:
                self.CMCPlot()
                phi = self.InterpolatePhi()

            self.massfractions = phi
            self.SetChainDensity(phi)
            
            print('\n___________ step {} ___________'.format(step))
            print('\n... chain concentrations {}: {}'.format(self.chainlabels, self.C_species))
        
            with open(self.mic_tempfile,'r') as fi:
                fid = fi.read()
                for _cl, _vf in zip(self.chainlabels, self.volfractions):
                    fid = fid.replace('<{}>'.format(_cl), str(_vf))
                fid = fid.replace('<CChainDensity>', str(self.C))
                fid = fid.replace('<NumBlocks>', str(self.nblocks))
                fid = fid.replace('<CellScaling>', str(self.boxL))
                fid = fid.replace('<NPW>', str(self.mic_NPW))
                fid = fid.replace('<InputFieldsFile>', str(self.mic_seedfile))
                fid = fid.replace('<ReadInputFields>', str(self.mic_RIF))

            fw = open(self.mic_runfile,'w')
            fw.write(fid)
            fw.close()
            fi.close()

            call('/home/djzhao/Packages/PolyFTS/bin/Release/{} {} > z.log'.format(self.pfts, self.mic_runfile), shell=True)
            
            status = np.loadtxt('STATUS')
            if status != 2:
                raise RuntimeError('Error: SCFT simulation did not converge for {}'.format(self.mic_runfile))
    
            datafile = np.loadtxt('operators.dat')
            data = datafile[-1]   
            H = data[1]
            miu = data[-len(self.chainlabels):]
            miu_n = np.sum(miu*self.C_species)
         
            self.H_mic.append(H - miu_n)
            self.miu_species.append(miu)
            self.rho_mic.append(self.C_species)

            shutil.copy('fields_k.bin', '{}'.format(self.mic_seedfile))
            shutil.copy('DensityOperator.dat', 'DensityOperator_mic.dat')

            with open(self.dis_tempfile,'r') as fi:
                fid = fi.read()
                for _cl, _m in zip(self.chainlabels, miu):
                    fid = fid.replace('<{}>'.format(_cl), str(_m))
                fid = fid.replace('<NumBlocks>', str(self.nblocks))
                fid = fid.replace('<CellScaling>', str(self.boxL))
                fid = fid.replace('<NPW>', str(self.dis_NPW))
                fid = fid.replace('<InputFieldsFile>', str(self.dis_seedfile))
                fid = fid.replace('<ReadInputFields>', str(self.dis_RIF))

            fw = open(self.dis_runfile,'w')
            fw.write(fid)
            fw.close()
            fi.close()

            call('/home/djzhao/Packages/PolyFTS_GC/bin/Release/PolyFTS.x {} > z.log'.format(self.dis_runfile), shell=True)
            
            status = np.loadtxt('STATUS')
            if status != 2:
                raise RuntimeError('Error: SCFT simulation did not converge for {}'.format(self.dis_runfile))
    
            datafile = np.loadtxt('operators.dat')
            data = datafile[-1]   
            H = data[1]
            rho = data[-len(self.chainlabels):]
            self.H_dis.append(H)
            self.rho_dis.append(rho)

            shutil.copy('fields_k.bin', '{}'.format(self.dis_seedfile))

            self.SetBounds(step, phi)
            self.CMCReport(step)

            if step >= 2:
                self.CheckConvergence()

            step += 1
            print('\n... cumulative runtime: {}'.format(time.time()-timestart))
            
        return