import os
import sys
import re
import time
import shutil
import numpy as np
from subprocess import call


class GrandSystem():

    def __init__(self):

        self.dis_tempfile = 'temp_dis.in'
        self.mic_tempfile = 'temp_mic.in'

        self.dis_runfile = 'input_dis.in'
        self.mic_runfile = 'input_mic.in'

        self.dis_seedfile = 'dis_fields_k.bin'
        self.mic_seedfile = 'mic_fields_k.bin'

        self.dis_RIF = 'True'
        self.mic_RIF = 'True'

        self.boxL = None

        self.chainlabels = []

        self.miu_mic = []
        self.miu_dis = []
 
        self.dis_NPW = 1
        self.mic_NPW = 64
        
        self.pfts = 'PolyFTS.x'

        return
    
    def MiuWarmUp(self, miu1=None, miu2=None, NumBlocks=10, steps=10):
        
        miu_steps = np.linspace(miu1, miu2, steps)
        timestart = time.time()
        print('chemical potential steps: ', miu_steps)

        for i in range(steps):
            print('\n___________ step {} ___________'.format(i))
            print('... miu values {}'.format(miu_steps[i]))
            
            with open(self.mic_tempfile, 'r') as fi:
                fid = fi.read()
                for _c, _m in zip(self.chainlabels,miu_steps[i]):
                    fid = fid.replace('<{}>'.format(_c), str(_m))
                fid = fid.replace('<NumBlocks>', str(NumBlocks))
                fid = fid.replace('<CellScaling>', str(self.boxL))
                fid = fid.replace('<NPW>', str(self.mic_NPW))
                fid = fid.replace('<InputFieldsFile>', str(self.mic_seedfile))
                fid = fid.replace('<ReadInputFields>', str(self.mic_RIF))

            fw = open(self.mic_runfile,'w')
            fw.write(fid)
            fw.close()
            fi.close()

            call('/home/djzhao/Packages/PolyFTS_GC/bin/Release/{} {} > z.log'.format(self.pfts, self.mic_runfile), shell=True)
            shutil.copy('fields_k.bin', '{}'.format(self.mic_seedfile))

            status = np.loadtxt('STATUS')
            if status != 2 and status != 3:
                raise RuntimeError('Error: SCFT simulation did not converge')
            
            print('... cumulative runtime: {}'.format(time.time()-timestart))

        return
    
    def BoxLengthWarmUp(self, miu=None, boxL_initial=None, boxL_final=None, NumBlocks=10, steps=10):
        
        if boxL_initial is None:
            boxL_initial = self.boxL

        boxL_steps = np.linspace(boxL_initial, boxL_final, steps)
        timestart = time.time()
        print('cell scaling steps: ', boxL_steps)

        for i in range(steps):
            print('\n___________ step {} ___________'.format(i))
            print('... box length {}'.format(boxL_steps[i]))
            
            with open(self.mic_tempfile,'r') as fi:
                fid = fi.read()
                for _c, _m in zip(self.chainlabels, miu):
                    fid = fid.replace('<{}>'.format(_c), str(_m))
                fid = fid.replace('<NumBlocks>', str(NumBlocks))
                fid = fid.replace('<CellScaling>', str(boxL_steps[i]))
                fid = fid.replace('<NPW>', str(self.mic_NPW))
                fid = fid.replace('<InputFieldsFile>', str(self.mic_seedfile))
                fid = fid.replace('<ReadInputFields>', str(self.mic_RIF))

            fw = open(self.mic_runfile,'w')
            fw.write(fid)
            fw.close()
            fi.close()

            call('/home/djzhao/Packages/PolyFTS_GC/bin/Release/{} {} > z.log'.format(self.pfts, self.mic_runfile), shell=True)
            shutil.copy('fields_k.bin', '{}'.format(self.mic_seedfile))

            status = np.loadtxt('STATUS')
            if status != 2 and status != 3:
                raise RuntimeError('Error: SCFT simulation did not converge')
            
            print('... cumulative runtime: {}'.format(time.time()-timestart))

        return
