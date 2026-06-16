import os
import glob
import subprocess
import argparse as ap
from subprocess import call
import numpy as np
import shutil
import time


def extractFE(args, outdirs):

	basedir = os.getcwd()
	FE = []
	weightpercent = []
	for outdir in outdirs:

		os.chdir(outdir)
		if os.path.exists('./pmatch.dat'):
			loadData = np.loadtxt('pmatch.dat')
			if loadData.ndim > 1:
				CD = loadData[-1,0]
				P = loadData[-1,1]
				H = loadData[-1,2]
			elif np.size(loadData) > 0:
				CD = loadData[0]
				P = loadData[1]
				H = loadData[2]
			else: 
				print('pmatch.dat file is empty for {}'.format(outdir))
				os.chdir(basedir)
				continue
		else:
			print('pmatch files not found for directory {}'.format(outdir))
			os.chdir(basedir)
			continue
		
		weightpercent.append(100*float(outdir[len(args.prefix):]))

		if args.FE == 'A':
			FE.append(H)
		elif args.FE == 'G':
			FE.append((H-P)/CD)
		os.chdir(basedir)	

	return weightpercent, FE

def extractDomainSpacing(args, outdirs):

	basedir = os.getcwd()
	d = []
	weightpercent = []
	for outdir in outdirs:

		os.chdir(outdir)
		if os.path.exists('./operators.dat'):
			loadData = np.loadtxt('operators.dat')
			if loadData.ndim > 1:
				L = loadData[-1,-1]			
			elif np.size(loadData) > 0:
				L = loadData[-1]
			else: 
				print('pmatch.dat file is empty for {}'.format(outdir))
				os.chdir(basedir)
				continue	
		else:
			print('pmatch files not found for directory {}'.format(outdir))
			os.chdir(basedir)
			continue

		weightpercent.append(100*float(outdir[len(args.prefix):]))
		d.append(L)
		os.chdir(basedir)
		
	return weightpercent, d


#########################################################################################################	

if __name__ == "__main__":

	parser = ap.ArgumentParser(description='SCFT composition sweep for a ternary system')
	parser.add_argument('-comps',type=str, default=None, help='numpy array of compositions')
	parser.add_argument('-prefix', type=str, default='WeightFrac', help='directory prefix')
	parser.add_argument('-FE', type=str, default=None, help='extract free energies to numpy array object')
	parser.add_argument('-d', action='store_true', help='extract lam domain spacing to numpy array object')
	args = parser.parse_args()

	if args.comps:	
		comps = np.round(np.load(args.comps),3)
		outdirs = [args.prefix+str(comp) for comp in comps[0]]
	else:
		outdirs = [p for p in glob.glob(args.prefix+'*') if os.path.isdir(p)]
		outdirs.sort()
		comps = [float(f[len(args.prefix):]) for f in outdirs]

	if args.FE:
		weightpercent, FE = extractFE(args, outdirs)	
		print(weightpercent, FE )
		np.save('FreeEnergy',[weightpercent,FE])
	if args.d:
		weightpercent, d = extractDomainSpacing(args, outdirs)
		np.save('DomainSpacing',[weightpercent,d])
	



