#!/usr/bin/env python

# This wrapper script implements a simple Gibbs ensemble CL sampling
# for two phases of different C. It is most useful for compressible systems
# undergoing macrophase separation.
#
# Kris T. Delaney, UCSB 2016.
# David Zhao, UCSB 2022.


import os
import sys
import re
import subprocess
from subprocess import call
import argparse as ap
import numpy as np
import math
#sys.path.append('/home/djzhao/Packages/PolyFTS/bin/Release/PolyFTS.x')
import time


def match_pressure(Ptarget, iters, tempfile, pfts):

	dtC=0.5
	tolerance=1e-6
	RIF = 'False'
	pfts = '/home/djzhao/Packages/PolyFTS/bin/Release/' + pfts
	timestart = time.time()

	for t in range(iters):

		if os.path.exists('./fields_k.bin'):
			RIF = 'True'
		if t == 0:
			Pfile = open('pmatch.dat','w')
			Pfile.write('# Chain Density, Pressure, Hamiltonian, Stress\n')
			with open('input.in', 'r') as fp:
				for line_no, line in enumerate(fp):
					if 'CChainDensity' in line:
						line = line.split()
						CD = float(line[2])
						break
		else: 
			if t < 3: # generate 3 inital points
				CD = CD *(1.0 + dtC*(math.sqrt(Ptarget/P1) - 1.0))
			elif t >= 3: # do linear extrapolation
				CD0 = np.loadtxt('pmatch.dat')[-1,0]
				x=np.loadtxt('pmatch.dat')[:,0]
				y=np.loadtxt('pmatch.dat')[:,1] - Ptarget
				fit = np.polyfit(x, y, 1)
				CD = -fit[1]/fit[0]
			if CD < 0.0: # manually fix overflows
				CD = abs(CD)
		
			fr = open(tempfile)		
			fid = fr.read().replace('<CChainDensity>',str(CD)).replace('<ReadInputFields>',RIF)
			fw = open('input.in','w')
			fw.write(fid)
			fw.close()
			fr.close()
		
		call('{} {} > z.log'.format(pfts,'input.in'), shell=True)
		print('... cumulative runtime: {}'.format(time.time()-timestart))

		LoadOps = np.loadtxt('operators.dat')
	
		if len(LoadOps) > 1:
			FinalOps = LoadOps[-1]
		else:
			FinalOps = LoadOps
	
		H1 = FinalOps[1]
		P1 = FinalOps[2]
		S1 = FinalOps[3]
		Pfile.write('{} {} {} {}\n'.format(str(CD),str(P1),str(H1),str(S1)))
		Pfile.flush()

		error = np.abs(P1-Ptarget)/np.abs(Ptarget)    
		if error <= tolerance:
			print('.......Pressure Converged.......')
			Pfile.close()
			break
		
	# last run
	#print(' === Final Run === ')
	#with open(tempfile,'r') as myfile:
	#	ini=myfile.read()
	#	ini=re.sub('<CChainDensity>',str(CD),ini)
	#	ini=re.sub('<ReadInputFields>','True',ini)
	#	runfile = open('input.in','w')
	#	runfile.write(ini)
	#	runfile.close()
	#call('{} {} > z.log'.format(pfts,'input.in'), shell=True)

	print('Final Chain Density: {}'.format(str(CD)))
	print('... cumulative runtime: {}'.format(time.time()-timestart))
	print(' === Final Run Done === ')

if __name__ == "__main__":
	parser = ap.ArgumentParser(description='SCFT pressure matching')
	parser.add_argument('-P', type=float, default=283.951857, help='pressure') 
	parser.add_argument('-template', type=str, default='temp.in', help='PolyFTS input file template')
	parser.add_argument('-iters', type=int, default=1, help='pressure matching interations')
	parser.add_argument('-pfts', type=str, default='PolyFTS.x', help='PolyFTS binary to run')
	args = parser.parse_args()

	match_pressure(args.P, args.iters, args.template, args.pfts)