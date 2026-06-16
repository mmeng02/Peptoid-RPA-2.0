#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:23:31 2021

@author: nvthaomy

Read the density data file, DensityOperator.dat, from 3D FTS and plot the density profile and check total bead fraction
Note: 
    entries of DensityOperator.dat are bead concentration normalized by C, i.e. local bead fraction
    entries * C = local number density
    so sum_grid{total bead conc per grid * V_grid}/V_tot = 1.0
"""

import argparse as ap
import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import os, re
from scipy import optimize as opt
from scipy.optimize import curve_fit
from scipy.integrate import simps
try:
  os.environ["DISPLAY"] #Detects if display is available
except KeyError:
  showPlots = False
  matplotlib.use('Agg') #Need to set this so doesn't try (and fail) to open interactive graphics window

colors = ['#6495ED','#2E8B57','#6da81b','#483D8B','#FF8C00', 'r','#800080','#008B8B','#949c2d', '#a34a17','#c43b99','#949c2d','#1E90FF']
matplotlib.rc('font', size=7)
matplotlib.rc('axes', titlesize=7)

parser = ap.ArgumentParser(description="plot density profile from 3D FTS simulation and check the total bead fraction")
parser.add_argument('-f',type=str, default='DensityOperator.dat', help="density file path")
parser.add_argument('-s', type=int, nargs = '+', help="indices of species")
parser.add_argument('-ax1', type=str, nargs = 2, default=None, 
                    help="x,y or z. Axis to make a 1D and 2D slice and its value")
parser.add_argument('-ax2', type=str, nargs = 2, default=None, 
                    help="x,y or z. Axis to make a 1D slice and its value")
# additional arguments if enabling finding micelles and do integral
parser.add_argument('-m', action='store_true', help='enable finding micelle')
parser.add_argument('-stride', type=int, default=1, help='stride in grid when scanning for micelle')
parser.add_argument('-b', type=float, nargs='+', 
                    help='calculate concentration within bounds, integration bounds in order: x_lower x_upper y_lower y_upper etc.')
parser.add_argument('-slice', type=str, default=None, help='slice at different value of this axis and plot 2D density profile')
parser.add_argument('-c', type=float, nargs="+", help='list of charges for species specified in -s and plot charge profile')
parser.add_argument('-c2', type=float, nargs="+", help='list of charges for ALL species')
parser.add_argument('-cmap', type=str, default='coolwarm', help='color map, e.g. coolwarm, bwr, Spectral')
parser.add_argument('-count', action='store_true', help='enable counting density bin, and estimate fraction of species in bulk phase')
parser.add_argument('-g', action='store_true', help='show plots')
parser.add_argument('-r', type=float, nargs='+', help='range of value to cover on color map')
parser.add_argument('-rc', type=float, nargs='+', help='range of value to cover on color map for charge profile')
parser.add_argument('-circle', type=str, help='text file with coordinates and radius of circle')
parser.add_argument('-infl', type=float, default = 0.5, help='inflection value')
parser.add_argument('-names', type=str, nargs="+", default = None, help='list of name for ALL species')
args = parser.parse_args()


file = args.f
species = args.s
charges = args.c
if args.ax1:
    ax1 = args.ax1[0] # y z , slice in x/y/z direction
    ax1_val = float(args.ax1[1])
else:
    ax1 = None
if args.ax2:
    ax2 = args.ax2[0] # None to skip 1D plot
    ax2_val = float(args.ax2[1])
else:
    ax2 = None
if args.r:
    vmin = args.r[0]
    vmax = args.r[1]
else:
    vmin = None
    vmax = None
ax_slice = args.slice
cmap=matplotlib.cm.get_cmap(args.cmap)
#========================

lines = open(file,'r').readlines()
for line in lines:
    if line.startswith('#'):
        if 'nfields' in line:
            nbead = int(line.split()[-1])
        elif 'NDim' in line:
            dim = int(line.split()[-1])
        elif 'PW grid' in line:
            n_grid = line.split('=')[-1]
            n_grid = [int(x) for x in n_grid.split()]
            n_grid = {d: n_grid[d] for d in range(dim)}
        elif 'complex data' in line:
            CL = bool(int(line.split()[-1]))
    else:
        break

cols = {'x':0, 'y':1, 'z':2}
rhocols = {}
dat = np.loadtxt(file)
ncol = len(dat[0])
if CL:
    for i in range(nbead):
        rhocols.update({i: dim + i*2})
else:
    nbead = int(ncol-dim)
    for i in range(nbead):
        rhocols.update({i: dim + i})
print('==={} species detected==='.format(nbead))
print('CL: {}'.format(str(CL)))
if not species:
    species = list(range(nbead))    
coords = dat[:,0:dim]

# check bead fraction
grid_val = {}
d_grid = {}
for d in range(dim):
    grid_val.update({d: sorted(np.unique(coords[:,d]))})
    d_grid.update({d: max(grid_val[d])/(float(n_grid[d]-1))})
    if d == 0:
        V = (n_grid[d]-1) * d_grid[d]
    else:
        V *= (n_grid[d]-1) * d_grid[d]
dV = V / np.prod([n-1 for n in n_grid.values()])
# get total bead density
col_tmp=list(rhocols.values())
C = np.sum(np.sum(dat[:,col_tmp] * dV,axis=1))/V # total bead fraction (normalized by Ctot)
print('{} total grid points'.format(np.prod([n for n in n_grid.values()])))
print('box dimensions {}'.format([n_grid[d] * d_grid[d] for d in range(dim)]))
print('Check total bead fraction (should be 1.0): {:0.2f}'.format(C))
col_tmp = [rhocols[s] for s in species]
ni = np.sum(dat[:,col_tmp] * dV,axis=1)
filter = ni < (1.e-5 * np.max(ni))
Ci = np.sum(ni)/V 
Ci_out = np.sum(ni[filter])/V 
print('Total bead fraction of species {}: {:.4e}'.format(species,Ci))
print('Total bead fraction of species {} outside of primary phase: {:.4e}'.format(species,Ci_out))

if charges:
    col_tmp = [rhocols[s] for s in species]
    Ce = np.sum(np.sum(dat[:,col_tmp] * charges * dV,axis=1))/V # total charge normalized by Ctot
    print('Total charge normalized by Ctot of species {}: {:.3e}'.format(species,Ce))

col_tmp = [rhocols[s] for s in species]
rhos = dat[:,col_tmp]
if charges:
    rhos_e = rhos *  charges
    rhos_e = np.sum(rhos_e,axis=1)
rhos = np.sum(rhos,axis=1)

if dim == 1:
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=[3,3])
    if charges:
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=[3,3]) 
    data = [grid_val[0]] 
    for si, s in enumerate(species):
        rho = dat[:,rhocols[s]]
        data.append(rho)
        if charges:
            rho_e = rho *  charges[si]
            ax2.plot(grid_val[0], rho_e, label=s)
        ax1.plot(grid_val[0], rho, label=s)   
    ax1.set_xlabel('x')
    ax1.set_ylabel('$\\rho/Ctot$')
    ax1.legend(args.names,loc='upper right',prop={'size':5})
    title1='C'
    #fig1.title(title1, loc = 'center')
    fig1.savefig('_'.join(re.split(' |=|,',title1))+'.png',dpi=500,transparent=False,bbox_inches=
    "tight")
    if charges:
        ax2.set_xlabel('x')  
        ax2.set_ylabel('$\\rho_e/Ctot$')   
        ax2.legend(loc='upper right',prop={'size':5})    
        title2='Ce'   
        #fig2.title(title2, loc = 'center')
        fig2.savefig('_'.join(re.split(' |=|,',title2))+'.png',dpi=500,transparent=False,bbox_inches=
        "tight")
    data = np.stack((data),axis=1)
    np.savetxt('_'.join(re.split(' |=|,',title1))+'.txt',data,header='x rho/Ctot of species {}'.format(species))
    if args.g:
        plt.show()

elif dim == 2:
    # Make 2D array of rhos
    X = rhos.reshape([n_grid[0],n_grid[1]]).T
    
    plt.figure(figsize=[3,3])
    im=plt.imshow(X,cmap=cmap,alpha = 1,interpolation= 'none', vmin=vmin, vmax=vmax,
                extent = [min(coords[:,0]),max(coords[:,0]),min(coords[:,1]),max(coords[:,1])],
                origin='lower')
    if args.names: plt.colorbar(im,label='$\\rho_{}/Ctot$'.format(args.names))
    else: plt.colorbar(im,label='$\\rho_{}/Ctot$'.format(species))
    if args.names: title='C '+ (len(species)*'{} ').format(*args.names)
    else: args.names: title='C '+ (len(species)*'{} ').format(*species)
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.title(title,loc='center')
    axs = plt.gca()
    #axs.xaxis.set_major_locator(ticker.MultipleLocator(2.))
    #axs.yaxis.set_major_locator(ticker.MultipleLocator(2.))
    np.savetxt('_'.join(re.split(' |=|,',title))+'.txt', X, header='{} columns {} rows'.format(X.shape[1], X.shape[0]))
    plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches="tight")

    if charges:
        Y = rhos_e.reshape([n_grid[0],n_grid[1]]).T
        if args.rc:
            vmin = args.rc[0]
            vmax = args.rc[1]
        else:
            vmin = - max(np.abs(np.min(Y)),np.abs(np.max(Y)))
            vmax = -vmin
        plt.figure(figsize=[3,3])
        im=plt.imshow(Y,cmap=cmap,alpha = 1,interpolation= 'none', vmin=vmin, vmax=vmax,
                extent = [min(coords[:,0]),max(coords[:,0]),min(coords[:,1]),max(coords[:,1])],
                origin='lower')
        
        if args.names: plt.colorbar(im,label='$\\rho_e,{}/Ctot$'.format(args.names))
        else: plt.colorbar(im,label='$\\rho_e,{}/Ctot$'.format(species))
        if args.names: title='charge '+ (len(species)*'{} ').format(*args.names)
        else: title='charge '+ (len(species)*'{} ').format(*species)
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.title(title,loc='center')
        axs = plt.gca()
        #axs.xaxis.set_major_locator(ticker.MultipleLocator(2.))
        #axs.yaxis.set_major_locator(ticker.MultipleLocator(2.))
        plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches="tight")

    if args.g:
        plt.show()

elif dim == 3:     

    # 3D contour plot of density  
    print('\nCreating 3D contour plot of species {} '.format(species))
    col_tmp = [rhocols[s] for s in species]
    rhos = dat
    rhos = rhos[:,col_tmp]
    if charges:
        rhos_e = rhos *  charges
        rhos_e = np.sum(rhos_e,axis=1)
    rhos = np.sum(rhos,axis=1)
    #print(np.shape(rhos))
    Xmin, Xmax = np.min(dat[:,0]), np.max(dat[:,0])
    Ymin, Ymax = np.min(dat[:,1]), np.max(dat[:,1])
    Zmin, Zmax = np.min(dat[:,2]), np.max(dat[:,2])
    X, Y, Z = np.meshgrid(np.linspace(Xmin, Xmax, num=n_grid[0]), np.linspace(Ymin, Ymax, num=n_grid[1]), np.linspace(Zmin, Zmax, num=n_grid[2]))
    #print(np.shape(X))
    rho_mesh = np.zeros([n_grid[0], n_grid[1], n_grid[2]])
    idx = 0
    for ii in range(n_grid[0]):
        for jj in range(n_grid[1]):
            for kk in range(n_grid[2]):
                rho_mesh[ii][jj][kk] = rhos[idx]
                idx += 1
    
    kw = {
    'vmin': rho_mesh.min(),
    'vmax': rho_mesh.max(),
    'levels': np.linspace(rho_mesh.min(), rho_mesh.max(), 100)
    }

    # Create a figure with 3D ax
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Plot contour surfaces
    _ = ax.contourf(
        X[:, :, 0], Y[:, :, 0], rho_mesh[:, :, 0],
        zdir='z', offset=Z.max(), **kw
    )
    _ = ax.contourf(
        X[0, :, :], rho_mesh[0, :, :], Z[0, :, :],
        zdir='y', offset=0, **kw
    )
    C = ax.contourf(
        rho_mesh[:, 0, :], Y[:, 0, :], Z[:, 0, :],
        zdir='x', offset=X.max(), **kw
    )

    ax.set(xlim=[Xmin, Xmax], ylim=[Ymin, Ymax], zlim=[Zmin, Zmax])
    # Plot edges
    edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
    ax.plot([Xmax, Xmax], [Ymin, Ymax], Zmax, **edges_kw)
    ax.plot([Xmin, Xmax], [Ymin, Ymin], Zmax, **edges_kw)
    ax.plot([Xmax, Xmax], [Ymin, Ymin], [Zmin, Zmax], **edges_kw)
    ax.set(
        xlabel='X [nm]',
        ylabel='Y [nm]',
        zlabel='Z [nm]',
        xticks=[0, Xmax],
        yticks=[0, Ymax],
        zticks=[0, Zmax],
    )

    # Set zoom and angle view
    #ax.view_init(40, -30, 0)
    #ax.set_box_aspect(None, zoom=0.9)

    # Colorbar
    if args.names: title='C_contour '+ (len(species)*'{} ').format(*args.names)
    else: title='C_contour '+ (len(species)*'{} ').format(*species)
    if args.names: fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='ρ_{}/Ctot'.format(args.names))
    else: fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='ρ_{}/Ctot'.format(species))
    #plt.title(title,loc='center')
    plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches="tight")
    
    # 3D Heat map scatter plot
    print('\nCreating 3D heat map scatter plot of species {} '.format(species))
    xcoords = dat[:,0]
    ycoords = dat[:,1]
    zcoords = dat[:,2]

    rhos_filter = rhos > (1.e-2*np.max(rhos)) #remove scatter points where the density is essentially zero
    rhos = rhos[rhos_filter]
    xcoords = xcoords[rhos_filter]
    ycoords = ycoords[rhos_filter]
    zcoords = zcoords[rhos_filter]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    C = ax.scatter(xcoords, ycoords, zcoords, c=rhos, marker='.')
    ax.set(xlim=[Xmin, Xmax], ylim=[Ymin, Ymax], zlim=[Zmin, Zmax])
    # Plot edges
    edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
    ax.plot([Xmax, Xmax], [Ymin, Ymax], Zmax, **edges_kw)
    ax.plot([Xmin, Xmax], [Ymin, Ymin], Zmax, **edges_kw)
    ax.plot([Xmax, Xmax], [Ymin, Ymin], [Zmin, Zmax], **edges_kw)
    ax.set(
        xlabel='X [nm]',
        ylabel='Y [nm]',
        zlabel='Z [nm]',
        xticks=[0, Xmax],
        yticks=[0, Ymax],
        zticks=[0, Zmax],
    )

    if args.names: title='C_scatter '+ (len(species)*'{} ').format(*args.names)
    else: title='C_scatter '+ (len(species)*'{} ').format(*species)
    if args.names: fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='ρ_{}/Ctot'.format(args.names))
    else: fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='ρ_{}/Ctot'.format(species))
    plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches="tight")

    # 2D slice of density 
    if ax1:
        x_array = dat[:,cols[ax1]] # coordinates of slice direction
        mindx = np.argmin(abs(x_array-ax1_val))
        x = x_array[mindx] # get the x value closest to the requested slice location
        rows = np.where(abs(x_array-x)/abs(x) <= 1e-3)[0] # row indices of this slice
        
        yz_name = np.array([key for key ,val in cols.items() if key != ax1])
        yz_cols = np.array([val for key ,val in cols.items() if key != ax1])
        yz_array = coords[rows,:]
        yz_array = yz_array[:,yz_cols]
        y_vals = sorted(np.unique(yz_array[:,0]))
        z_vals = sorted(np.unique(yz_array[:,1]))
        col_tmp = [rhocols[s] for s in species]
        rhos = dat[rows,:]
        rhos = rhos[:,col_tmp]
        if charges:
            rhos_e = rhos *  charges
            rhos_e = np.sum(rhos_e,axis=1)
        rhos = np.sum(rhos,axis=1)
        
        print('\n...Get density of species {} on 2D plane of {}={:.3f}...'.format(species,ax1,x))
        
        # Make 2D array of rhos
        X=np.zeros([len(y_vals),len(z_vals)])
        for i, y_tmp in enumerate(y_vals):
            ii = np.where(abs(yz_array[:,0] - y_tmp) < 1e-3)[0]
            for j, z_tmp in enumerate(z_vals):
                jj = np.where(abs(yz_array[:,1] - z_tmp) < 1e-3)[0]
                idx = np.intersect1d(ii,jj)
                X[j,i] = rhos[idx]
        plt.figure(figsize=[3,3])       
        im=plt.imshow(X,cmap=cmap,alpha = 1,interpolation= 'none', vmin=vmin, vmax=vmax,
                    extent = [min(yz_array[:,0]),max(yz_array[:,0]),min(yz_array[:,1]),max(yz_array[:,1])],
                    origin='lower')
        if args.names: plt.colorbar(im,label='$\\rho_{}/Ctot$'.format(args.names))
        else: plt.colorbar(im,label='$\\rho_{}/Ctot$'.format(species))
        if args.names: title='C '+ (len(species)*'{} ').format(*args.names)+'{} {:.2f}'.format(ax1, x)
        else: title='C '+ (len(species)*'{} ').format(*species)+'{} {:.2f}'.format(ax1, x)
        plt.xlabel('{}'.format(yz_name[0]))
        plt.ylabel('{}'.format(yz_name[1]))
        #plt.title(title,loc='center')
        axs = plt.gca()
        #axs.xaxis.set_major_locator(ticker.MultipleLocator(2.))
        #axs.yaxis.set_major_locator(ticker.MultipleLocator(2.))
        plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches="tight")
        
        if charges:
            Y=np.zeros([len(y_vals),len(z_vals)])
            for i, y_tmp in enumerate(y_vals):
                ii = np.where(abs(yz_array[:,0] - y_tmp) < 1e-3)[0]
                for j, z_tmp in enumerate(z_vals):
                    jj = np.where(abs(yz_array[:,1] - z_tmp) < 1e-3)[0]
                    idx = np.intersect1d(ii,jj)
                    Y[j,i] = rhos_e[idx]
            plt.figure(figsize=[3,3])  
            if args.rc:
                vmin = args.rc[0]
                vmax = args.rc[1]
            else:
                vmin = - max(np.abs(np.min(Y)),np.abs(np.max(Y)))
                vmax = -vmin
            im=plt.imshow(Y,cmap=cmap,alpha = 1,interpolation= 'none',
                        extent = [min(yz_array[:,0]),max(yz_array[:,0]),min(yz_array[:,1]),max(yz_array[:,1])],
                        vmin=vmin, vmax=vmax,origin='lower')
            plt.colorbar(im,label='$\\rho_e,{}/Ctot$'.format(species))
            title='charge '+ (len(species)*'{} ').format(*species)+'{} {:.2f}'.format(ax1, x)
            plt.xlabel('{}'.format(yz_name[0]))
            plt.ylabel('{}'.format(yz_name[1]))
            plt.title(title,loc='center')
            axs = plt.gca()
            #axs.xaxis.set_major_locator(ticker.MultipleLocator(2.))
            #axs.yaxis.set_major_locator(ticker.MultipleLocator(2.))
            plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches="tight")    
    if ax2:
        if ax1 == ax2:
            raise Exception('2 axes must be different')
        y_array = dat[:,cols[ax2]] # coordinates of slice direction
        mindy = np.argmin(abs(y_array-ax2_val))
        y = y_array[mindy] 
        x_rows = np.where(abs(x_array-x)/abs(x) <= 1e-3)[0] 
        y_rows = np.where(abs(y_array-y)/abs(y) <= 1e-3)[0] 
        rows = np.intersect1d(x_rows,y_rows)
        print('\n...Get density of species {} on 1D plane of {}={:.3f} {}={:.3f}...'.format(species, ax1,x,ax2,y))
        z_name = np.array([key for key ,val in cols.items() if not key in [ax1,ax2]])
        z_col = np.array([val for key ,val in cols.items() if not key in [ax1,ax2]])
        z_array = coords[rows,:]
        z_array = z_array[:,z_col]
        rhos = dat[rows,:]
        rhos = rhos[:,col_tmp]
        if charges:
            rhos_e = rhos *  charges
            rhos_e = np.sum(rhos_e,axis=1)
        rhos = np.sum(rhos,axis=1)
        
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[3,3])
        plt.plot(z_array, rhos)
        plt.xlabel('{}'.format(z_name[0]))
        plt.ylabel('$\\rho_{}/Ctot$'.format(species))
        title='C '+ (len(species)*'{} ').format(*species)+'{} {:.2f} {} {:.2f}'.format(ax1, x, ax2, y)
        plt.title(title, loc = 'center')
        plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches=
        "tight")
        data = np.stack((z_array.flatten(),rhos),axis=1)
        np.savetxt('_'.join(re.split(' |=|,',title))+'.txt',data,header='{} rho/Ctot'.format(z_name))

        if charges:
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[3,3])
            plt.plot(z_array, rhos_e)
            plt.xlabel('{}'.format(z_name[0]))
            plt.ylabel('$\\rho_e,{}/Ctot$'.format(species))
            title='charge '+ (len(species)*'{} ').format(*species)+'{} {:.2f} {} {:.2f}'.format(ax1, x, ax2, y)
            plt.title(title, loc = 'center')
            plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches=
            "tight")
            data = np.stack((z_array.flatten(),rhos),axis=1)
            np.savetxt('_'.join(re.split(' |=|,',title))+'.txt',data,header='{} rho_e/Ctot'.format(z_name))

    '''
    print('\n...do 1D histogram along each axis...')
    col_tmp = [rhocols[s] for s in species]
    # x
    x_array = dat[:,0] # coordinates of slice direction
    rho_array = []
    for i,x in enumerate(grid_val[0]):
        rows = np.where(abs(x_array-x) <= 1e-3)[0] # row indices of this slice
        rhos = dat[rows,:]
        rhos = np.sum(rhos[:,col_tmp],axis=1)
        rho_array.append(np.mean(rhos))

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[3,2])
    plt.plot(grid_val[0], rho_array, color='k', marker='o',ms=4)
    plt.xlabel('x')
    plt.ylabel('$\\rho_{}/Ctot$'.format(species))
    title='xhist '+ (len(species)*'{} ').format(*species)
    plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches=
        "tight")
    # y       
    x_array = dat[:,1] # coordinates of slice direction
    rho_array = []
    for i,x in enumerate(grid_val[1]):
        rows = np.where(abs(x_array-x) <= 1e-3)[0] # row indices of this slice
        rhos = dat[rows,:]
        rhos = np.sum(rhos[:,col_tmp],axis=1)
        rho_array.append(np.mean(rhos))
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[3,2])
    plt.plot(grid_val[1], rho_array, color='k', marker='o',ms=4)
    plt.xlabel('y')
    plt.ylabel('$\\rho_{}/Ctot$'.format(species))
    title='yhist '+ (len(species)*'{} ').format(*species)
    plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches=
        "tight")
    '''
    
    '''
    # z
    x_array = dat[:,2] # coordinates of slice direction
    rho_array = []
    for i,x in enumerate(grid_val[2]):
        rows = np.where(abs(x_array-x)<= 1e-3)[0] # row indices of this slice
        rhos = dat[rows,:]
        rhos = np.sum(rhos[:,col_tmp],axis=1)
        rho_array.append(np.mean(rhos))

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[3,2])
    plt.plot(grid_val[2], rho_array, color='k', marker='o',ms=4)
    plt.xlabel('z')
    plt.ylabel('$\\rho_{}/Ctot$'.format(species))
    title='zhist '+ (len(species)*'{} ').format(*species)
    plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches=
        "tight")
    if args.g:
        plt.show()
    # else:
    #     print('\n=== Finding micelles consisting of species {} ==='.format(species))

    if ax_slice:
        print('\n... Slice along {} and plot 2D profile...'.format(ax_slice))
        # slice in x:
        r_vals = sorted(np.unique(coords[:,cols[ax_slice]]))
        for k, x in enumerate(r_vals[::args.stride]):
            x_array = coords[:,cols[ax_slice]] # coordinates of slice in the x direction
            rows = np.where(abs(x_array-x) <= 1e-3)[0] # row indices of this slice
            
            yz_name = np.array([key for key ,val in cols.items() if key != ax_slice])
            yz_cols = np.array([val for key ,val in cols.items() if key != ax_slice])
            yz_array = coords[rows,:]
            yz_array = yz_array[:,yz_cols]
            y_vals = sorted(np.unique(yz_array[:,0]))
            z_vals = sorted(np.unique(yz_array[:,1]))
            col_tmp = [rhocols[s] for s in species]
            rhos = dat[rows,:]
            rhos = rhos[:,col_tmp]
            if charges:
                rhos_e = rhos * charges
                rhos_e = np.sum(rhos_e,axis=1)
            rhos = np.sum(rhos,axis=1)

            # plot 2D cut of rhos in yz plane
            X=np.zeros([len(z_vals),len(y_vals)])
            print(yz_array.shape)
            for i, y_tmp in enumerate(y_vals):
                ii = np.where(abs(yz_array[:,0] - y_tmp) < 1e-3)[0]
                for j, z_tmp in enumerate(z_vals):
                    jj = np.where(abs(yz_array[:,1] - z_tmp) < 1e-3)[0]
                    idx = np.intersect1d(ii,jj)
                    X[j,i] = rhos[idx]
            plt.figure(figsize=[3,3]) ; plt.clf()      
            im=plt.imshow(X,cmap=cmap,alpha = 1,interpolation= 'none', vmin=vmin, vmax=vmax,
                        extent = [min(yz_array[:,0]),max(yz_array[:,0]),min(yz_array[:,1]),max(yz_array[:,1])],
                        origin='lower')
            plt.colorbar(im,label='$\\rho_{}/Ctot$'.format(species))
            title='C '+ (len(species)*'{} ').format(*species)+'{} {:.2f}'.format(ax_slice, x)
            plt.xlabel(yz_name[0])
            plt.ylabel(yz_name[1])
            plt.title(title,loc='center')
            axs = plt.gca()
            #axs.xaxis.set_major_locator(ticker.MultipleLocator(2.))
            #axs.yaxis.set_major_locator(ticker.MultipleLocator(2.))

            if charges:
                Y=np.zeros([len(y_vals),len(z_vals)])
                for i, y_tmp in enumerate(y_vals):
                    ii = np.where(abs(yz_array[:,0] - y_tmp) < 1e-3)[0]
                    for j, z_tmp in enumerate(z_vals):
                        jj = np.where(abs(yz_array[:,1] - z_tmp) < 1e-3)[0]
                        idx = np.intersect1d(ii,jj)
                        Y[j,i] = rhos_e[idx]
                plt.figure(figsize=[3,3]) ; plt.clf()    
                if args.rc:
                    vmin = args.rc[0]
                    vmax = args.rc[1]
                else:
                    vmin = - max(np.abs(np.min(Y)),np.abs(np.max(Y)))
                    vmax = -vmin
                im=plt.imshow(Y,cmap=cmap,alpha = 1,interpolation= 'none', 
                            extent = [min(yz_array[:,0]),max(yz_array[:,0]),min(yz_array[:,1]),max(yz_array[:,1])],
                            vmin=vmin, vmax=vmax,origin='lower')
                plt.colorbar(im,label='$\\rho_e,{}/Ctot$'.format(species))
                title='charge '+ (len(species)*'{} ').format(*species)+'{} {:.2f}'.format(ax_slice, x)
                plt.xlabel(yz_name[0])
                plt.ylabel(yz_name[1])
                plt.title(title,loc='center')
                axs = plt.gca()
                #axs.xaxis.set_major_locator(ticker.MultipleLocator(2.))
                #axs.yaxis.set_major_locator(ticker.MultipleLocator(2.))            
         
            plt.show() 
            plt.pause(3)
    '''

if args.count:
    col_tmp = [rhocols[s] for s in species]
    rhos = np.ravel(np.sum(dat[:,col_tmp], axis = 1))
    hist,bins = np.histogram(rhos, bins=1000, density=True) 
    binmid = 0.5*(bins[1:]+bins[0:-1])

    def gauss(x,mu,sigma,A):
        return A*1/(sigma*np.sqrt(2.*np.pi))*np.exp(-(x-mu)**2/2/sigma**2)

    def lognormal(x,mu,sigma,A):
        return A * np.exp(-0.5 * (np.log(x) - mu)**2./sigma**2)
    # fit peak with a Gaussian
    showFit = False
    try:
        p0 = [binmid[np.argmax(hist)], (max(binmid)-min(binmid))/20. ,max(hist)]
        (mu,sigma,A),cov =curve_fit(gauss, binmid, hist, p0)
        x_fit = np.linspace(min(binmid),max(binmid),num=1000)
        fit = gauss(x_fit,mu,sigma,A)
        showFit = True
        print('\nSuggested bulk C: {:.3e} +/- {:.3e}\n'.format(mu,2*sigma))
    except:
        cutoff = 0.7
        CDF = 0
        maxCDF = simps(hist,binmid)
        print(maxCDF)
        if np.argmax(hist) <= 0.5 * len(hist):
            cnt = 0
            while CDF/maxCDF < cutoff and cnt < len(hist):
                cnt += 1
                CDF = simps(hist[:cnt+1], binmid[:cnt+1])
        else:
            cnt = len(hist) - 1
            while CDF/maxCDF < cutoff and cnt > 0:
                cnt -= 1
                CDF = simps(hist[cnt:], binmid[cnt:])
        mu = binmid[np.argmax(hist)]
        sigma = np.abs(binmid[cnt] - mu)
        print('\nSuggested bulk C: {:.3e} -> {:.3e}'.format(binmid[np.argmax(hist)],binmid[cnt]))

    # estimate fraction of species in the  bulk phase based on density criteria
    if showFit:
        cut_off = [mu-2*sigma, mu+2*sigma]
    else:
        cut_off = [mu-sigma, mu+sigma]
    bulk_rhos = rhos[(rhos >=  cut_off[0]) & (rhos <= cut_off[1])]
    V_bulk = dV * len(bulk_rhos)
    n_bulk = np.sum(bulk_rhos * dV)
    n_tot = np.sum(rhos * dV)
    f_bulk = n_bulk/n_tot
    print('bead fraction of {} in bulk phase: {:.3e}'.format(species,n_bulk))
    print('number fraction of {} in bulk phase vs overall composition: {:.3e}\n'.format(species, f_bulk))

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[3,2])
    plt.plot(binmid, hist, color='k')
    if showFit:
        plt.plot(x_fit, fit, ':r',label='Gaussian fit, $\mu$ = {:.3e}, $\sigma$ = {:.3e}'.format(mu,sigma))
    else:
        plt.vlines(binmid[np.argmax(hist)], min(hist), max(hist), color = 'r', linestyle =':', label = 'peak {:.3e}'.format(binmid[np.argmax(hist)]))
        if np.argmax(hist) <= 0.5 * len(hist):
            plt.fill_between(binmid[:cnt+1], hist[:cnt+1], alpha = 0.5)
        else:
            plt.fill_between(binmid[cnt:], hist[cnt:], alpha = 0.5)
    plt.vlines(cut_off[0], min(hist), max(hist), color = 'b', linestyle =':', label = 'cut off')
    plt.vlines(cut_off[1], min(hist), max(hist), color = 'b', linestyle =':')
    plt.xlim(max(min(binmid),mu - 50*sigma),mu + 50*sigma)
    plt.legend(args.names, loc='best',prop={'size':6})
    plt.xlabel('Ci/Ctot $(nm^-3)$')
    plt.ylabel('count')
    title='count C_'+ (len(species)*'{} ').format(*species)
    plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches=
        "tight")
if args.g:
    plt.show()
