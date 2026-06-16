import os
import glob
import subprocess
from subprocess import call
import numpy as np
import argparse as ap
import shutil
import time
import sys

from SCFT_tools import ChainProps 
from SCFT_tools import Pmatch_module


def MolVol_dictionary(Chains):

	ChainMolVols = np.empty(len(Chains))
	for i in range(len(Chains)):
		ChainMolVols[i] = MolVolDictionary[Chains[i]]
	ChainMolVols = ChainMolVols.reshape((len(Chains),1))
	return ChainMolVols

def MolWeight_dictionary(Chains):

	ChainMolWeights = np.empty(len(Chains))
	for i in range(len(Chains)):
		ChainMolWeights[i] = MolWeightDictionary[Chains[i]]
	ChainMolWeights = ChainMolWeights.reshape((len(Chains),1))
	return ChainMolWeights

def ChainLength_dictionary(Chains):

	ChainLengths = np.empty(len(Chains))
	for i in range(len(Chains)):
		ChainLengths[i] = ChainLengthDictionary[Chains[i]]
	ChainLengths = ChainLengths.reshape((len(Chains),1))
	return ChainLengths
	
def estimate_chain_density(args, comps):	
	
	if args.lengths:
		ChainLengths = np.array(args.lengths).reshape((len(args.chains),1))
	else:
		ChainLengths = ChainLength_dictionary(args.chains)
	
	if args.molv:
		ChainMolVols = np.array(args.molv).reshape((len(args.chains),1))
	else:
		ChainMolVols = MolVol_dictionary(args.chains)

	if args.molw:
		ChainMolWeights = np.array(args.molw).reshape((len(args.chains),1))
	else:
		ChainMolWeights = MolWeight_dictionary(args.chains)
	ChainCharges = np.array(args.charges)

	ChainNums = comps/ChainMolWeights
	
	ChainNums = ChainNums.T
	for i in range(len(ChainNums)): # Check for electrostatic neutrality
		NetCharge = np.sum(ChainNums[i]*ChainCharges) 
		if NetCharge < 0.0:
			print('... adjusting bead numbers for net positive charge: {}'.format(NetCharge))
			ChainNums[i][ChainCharges < 0] -= np.abs(NetCharge*ChainCharges[ChainCharges < 0]/np.sum(ChainCharges[ChainCharges < 0]))
		elif NetCharge > 0.0:
			print('... adjusting bead numbers for net negative charge: {}'.format(NetCharge))
			ChainNums[i][ChainCharges > 0] -= np.abs(NetCharge*ChainCharges[ChainCharges > 0]/np.sum(ChainCharges[ChainCharges > 0]))
	ChainNums = ChainNums.T

	ChainBeadNums = ChainNums*ChainLengths

	ChainVols = ChainNums*ChainMolVols
	TotalBeads = np.sum(ChainBeadNums, axis=0)
	TotalVols = np.sum(ChainVols, axis=0)

	ChainDensityEst = np.round((TotalBeads/TotalVols),12)
	ChainBeadFracs = np.round(ChainBeadNums/np.sum(ChainBeadNums,axis=0),12)

	print('Checking chain bead fractions sum to 1.0  for all compositions: ',np.sum(ChainBeadFracs,axis=0))
	return ChainDensityEst, ChainBeadFracs
	
def write_input(args, outdirs):

	basedir = os.getcwd()
	comps = np.load(args.comps)
	scaling = np.load(args.scaling)
	npw = np.load(args.npw)
	lss = np.load(args.lss)
	ChainDensityEst, ChainBeadFracs = estimate_chain_density(args, comps)
	
	for i,outdir in enumerate(outdirs):

		os.chdir(outdir)
		fr = open(os.path.join(basedir,args.template),'r')
		fid = fr.read()
		for j in range(len(args.chains)):	
			fid = fid.replace('<'+str(args.chains[j])+'>', str(ChainBeadFracs[j][i]))	
		
		fid = fid.replace('<VariableCell>',str(args.vc))
		fid = fid.replace('<CellScaling>',str(scaling[i]))
		fid = fid.replace('<NPW>',str(int(npw[i])))
		fid = fid.replace('<LambdaStressScale>',str(lss[i]))

		if args.cont and os.path.exists('./pmatch.dat'):
			LoadChainDensity = np.loadtxt('pmatch.dat')
			if LoadChainDensity.ndim > 1:
				ChainDensityCont = LoadChainDensity[-1,0]
				fid0 = fid.replace('<CChainDensity>',str(ChainDensityCont))
			elif np.size(LoadChainDensity) > 0:
				ChainDensityCont = LoadChainDensity[0]
				fid0 = fid.replace('<CChainDensity>',str(ChainDensityCont))
			else:
				print('pmatch.dat file is empty for {}'.format(outdirs[i]))
				fid0 = fid.replace('<CChainDensity>',str(ChainDensityEst[i]))
		else:
			if args.cont:
				print('pmatch.dat file not found for {}'.format(outdirs[i]))
			fid0 = fid.replace('<CChainDensity>',str(ChainDensityEst[i]))

		if os.path.exists('./fields_k.bin'):
			fid0 = fid0.replace('<ReadInputFields>', 'True')
		elif args.seed and i>0:
			fid0 = fid0.replace('<ReadInputFields>', 'True')
		else:
			fid0 = fid0.replace('<ReadInputFields>','False')

		fw = open(args.template,'w')
		fw.write(fid)
		fw.close()
		fw0 = open('input.in','w')
		fw0.write(fid0)
		fw0.close()
		fr.close()
		os.chdir(basedir)	

	return

def write_cont(args, outdirs):

	basedir = os.getcwd()
	for outdir in outdirs:

		os.chdir(outdir)
		fr = open(args.template,'r')
		fid = fr.read()
		
		if os.path.exists('./pmatch.dat'):
			LoadChainDensity = np.loadtxt('pmatch.dat')
			if LoadChainDensity.ndim > 1:
				ChainDensityCont = LoadChainDensity[-1,0]
				fid = fid.replace('<CChainDensity>',str(ChainDensityCont))
			elif np.size(LoadChainDensity) > 0:
				ChainDensityCont = LoadChainDensity[0]
				fid = fid.replace('<CChainDensity>',str(ChainDensityCont))
			else:
				print('pmatch.dat file is empty for {}'.format(outdir))
				fr.close()
				os.chdir(basedir)
				continue
		else:
			print('pmatch.dat file not found for {}'.format(outdir))
			fr.close()		
			os.chdir(basedir)
			continue

		if os.path.exists('./fields_k.bin'):
			fid = fid.replace('<ReadInputFields>', 'True')
		else:
			fid = fid.replace('<ReadInputFields>','False')

		fw = open('input.in','w')
		fw.write(fid)
		fw.close()
		fr.close()
		os.chdir(basedir)

	return			

def write_directories(args, outdirs):

	comps = np.load(args.comps)
	for outdir in outdirs:
		try:
			os.mkdir(outdir)
		except:
			print('directory {} already exists'.format(outdir))

	if args.cp:
		copy_files(args, outdirs)

	if not args.chains or not args.comps or not args.scaling:
		print('ERROR: chains, compositions, and scaling must be specified to write new inputs')
		sys.exit()

	if not args.charges:
		args.charges = np.zeros(len(args.chains))
		
	write_input(args, outdirs)
	return

def copy_files(args, outdirs):

	basedir = os.getcwd()
	for outdir in outdirs:
		if os.path.exists('./'+outdir):
			os.chdir(outdir)
			for file in args.cp:
				shutil.copy(os.path.join(basedir,file),os.getcwd())
		else:
			print('ERROR: cannot cp to {} directory thats does not exists'.format(outdir))

		os.chdir(basedir)
	return

def run_SCFT(args, outdirs):

	cwd = os.getcwd()
	for i in range(len(outdirs)):

		os.chdir(outdirs[i])
		print('_______________ '+outdirs[i]+' _______________')
		if args.seed:
			if i > 0:
				shutil.copy(seedfile,os.getcwd())
			seedfile = os.path.abspath('./fields_k.bin')
			
		Pmatch_module.match_pressure(args.P, args.iters, args.template, args.pfts)
		os.chdir(cwd)

def submit_jobs(args, outdirs):
	
	cwd = os.getcwd()
	for i in range(len(outdirs)):

		os.chdir(outdirs[i])
		print('_______________ '+outdirs[i]+' _______________')
		if os.path.exists('./'+args.slurm):
			call('sbatch '+ args.slurm, shell=True)
		else:
			print('submit file not found for {}'.format(outdirs[i]))
		
		os.chdir(cwd)

def main(args):

	if args.comps:	
		comps = np.load(args.comps)
		comps_dict = {}
		for s,c in zip(args.chains,comps):
			comps_dict[s] = c
		comps_labels = np.zeros(len(comps[0]))

		if args.clabel == None: args.clabel = [args.chains[0]]

		for i in range(len(comps[0])):
			for s in args.clabel:
				comps_s = comps_dict[s]
				comps_labels[i] += comps_s[i]
				
		outdirs = [args.prefix+str(np.round(label,3)) for label in comps_labels]
		if args.llabel:
			scaling = np.load(args.scaling)
			outdirs = [_od + '_L{}nm'.format(_s) for _od,_s in zip(outdirs, scaling)]
	else:
		outdirs = [p for p in glob.glob(args.prefix+'*') if os.path.isdir(p)]
		comps = [float(p[len(args.prefix):]) for p in outdirs]

	if args.write:
		write_directories(args, outdirs)
	else:
		if args.cp:
			copy_files(args, outdirs)
		if args.cont:
			write_cont(args, outdirs)

	for outdir in outdirs:
		if not os.path.exists('./'+outdir):
			print('ERROR: {} directory does not exists'.format(outdir))
			sys.exit()

	if args.run:
		run_SCFT(args, outdirs)	
	
	if args.slurm:
		submit_jobs(args,outdirs)
				
		
#########################################################################################################	


if __name__ == "__main__":

	parser = ap.ArgumentParser(description='SCFT composition sweep for a ternary system')
	parser.add_argument('-comps',type=str, default=None, help='numpy array of compositions')
	parser.add_argument('-scaling', type=str, default=None, help='cell scaling') 
	parser.add_argument('-npw', type=str, default=None, help='cell plane waves') 
	parser.add_argument('-lss', type=str, default=None, help='lambda stress scale') 
	parser.add_argument('-chains', type=str, nargs = '+', default=None, help='species names') 
	parser.add_argument('-charges', type=int, nargs = '+', default=None, help='chain total charges') 
	parser.add_argument('-lengths', type=int, nargs = '+', default=None, help='chain length of each species')
	parser.add_argument('-molv', type=float, nargs = '+', default=None, help='molecular volume of each species') 
	parser.add_argument('-molw', type=float, nargs = '+', default=None, help='molecular weight of each species') 
	parser.add_argument('-vc', type=str, default='True', help='true or false for variable cell method')

	parser.add_argument('-cp', type=str, nargs='+', default=None, help='specify a list of files to copy into each created directory')
	parser.add_argument('-template', type=str, default='temp.in', help='PolyFTS input file template')
	parser.add_argument('-prefix', type=str, default='WeightFrac', help='directory prefix')
	parser.add_argument('-clabel', type=str, nargs = '+', default=None, help='species whose weight fractions will serve as directory labels') 
	parser.add_argument('-llabel', action='store_true', help='include box length in directory name')

	parser.add_argument('-P', type=float, default=283.951857, help='pressure') 
	parser.add_argument('-iters', type=int, default=1, help='pressure matching interations')
	parser.add_argument('-pfts', type=str, default='PolyFTS.x', help='PolyFTS binary to run')
	parser.add_argument('-write', action='store_true', help='write new directories and input files for all compositions')
	parser.add_argument('-seed', action='store_true', help='seed simulation with fields from previous sequential composition')
	parser.add_argument('-cont', action='store_true', help='restart pressure matching with chain density from last iteration')
	parser.add_argument('-run', action='store_true', help='run simulation for each composition sequentially')
	parser.add_argument('-slurm', type=str, default=None, help='specify the bash script to submit jobs for all compositions in parallel')
	args = parser.parse_args()

	# main function call
	main(args)