#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:41:59 2024

@author: mfeldmann@giub.local
"""


import sys
sys.path.append('/home/mfeldmann/Research/code/mesocyclone_climate/')
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from skimage.morphology import disk, dilation
import argparse
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm
from matplotlib import gridspec, colormaps
import glob
import math
import utils as fmap
import matplotlib
matplotlib.rcParams.update({'font.size': 24})
from matplotlib import colors
#from sklearn.neighbors import KernelDensity


c1='#648fff' #lightblue
c2='#785ef0' #indigo
c3='#dc267f' #magenta
c4='#fe6100' #orange
c5='#ffb000' #gold
c6='#000000' #black
c7='#f3322f' #red

#%% INIT

climate = "future"
start_day = "0401"
end_day = "1130"
if climate == "current": years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
if climate == "future": years = ['2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095']
method = "model_tracks"
path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
capepath = "/media/mfeldmann@giub.local/Elements/supercell_climate/CAPE/"
rainpath = "/media/mfeldmann@giub.local/Elements/supercell_climate/CT2/"+climate+"_climate/"
figpath = "/home/mfeldmann/Research/figs/mesocyclone_climate/"

subdomains = xr.open_dataset('/home/mfeldmann/Research/data/mesocyclone_climate/domain/subdomains_lonlat.nc')

#%%
months = np.arange(4,12)
ctrls=[]
pgws=[]
deltas=[]
for month in months:
    month = str(month).zfill(2)
    ctrl = xr.open_dataset('/media/mfeldmann@giub.local/Elements/supercell_climate/CAPE/CAPE_MU_mean_ctrl_'+month,engine='netcdf4').CAPE_MU.squeeze()
    pgw = xr.open_dataset('/media/mfeldmann@giub.local/Elements/supercell_climate/CAPE/CAPE_MU_mean_pgw_'+month,engine='netcdf4').CAPE_MU.squeeze()
    delta = pgw - ctrl
    ctrls.append(ctrl)
    pgws.append(pgw)
    deltas.append(delta)
lons = ctrl.lon.values
lats = ctrl.lat.values
#%%
ddeltas = xr.concat(deltas,dim='months')

ddeltas_mean = ddeltas.mean(dim='months')

#%%

ncolors=256
cmap='seismic'
extend='both'
cmap = colormaps[cmap]
cmap.set_bad('grey')
bounds=[-300,-200,-100,-50,-10,10,50,100,200,300]
#bounds=[-1000,-500,-300,-200,-100,-50,-10,10,50,100,200,300,500,1000]
norm = BoundaryNorm(boundaries=bounds, ncolors=ncolors, extend=extend)
# determine the area of influence based on the footprint
# aoi = np.count_nonzero(disk(r_disk))*4.84
# aoi = round(aoi)
    
# load geographic features
resol = '10m'  # use data at this scale
bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol,
                                    facecolor='none', edgecolor='k', linestyle='-', alpha=1, linewidth=0.7)
coastline = cfeature.NaturalEarthFeature(category='physical', name='coastline', scale=resol, facecolor='none',
                                         edgecolor='k', linestyle='-', alpha=1, linewidth=0.7)
lakes = cfeature.NaturalEarthFeature(category='physical', name='lakes', scale=resol, facecolor='none',
                                     edgecolor='blue', linestyle='-', alpha=1, linewidth=0.8)
rp = ccrs.RotatedPole(pole_longitude = -170, pole_latitude = 43)


fig = plt.figure(figsize=(10,8))

ax = plt.axes(projection=rp)
ax.add_feature(bodr)
ax.add_feature(coastline)
zl, zr, zb, zt = 180, 25, 195, 25 #smart cut for entire domain
cbar_orientation = 'vertical'

cont = ax.pcolormesh(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], ddeltas_mean[zb:-zt,zl:-zr], 
                     norm=norm, cmap=cmap, transform=ccrs.PlateCarree())
# if hatch:
#     # ax.add_patch(hatching,hatch='x',edgecolor='k',transform=ccrs.PlateCarree())
#     ax.contourf(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], hatching[zb:-zt,zl:-zr],
#                 levels=[-0.1,0.9],colors='grey',alpha=0.3,hatches='..', 
#                 transform=ccrs.PlateCarree())
# if CH:
#     ax.set_extent([-3.88815558,  0.99421144, -1.86351067,  1.50951803], crs=rp)


cbar = plt.colorbar(cont, orientation=cbar_orientation, label='$\Delta$ CAPE [J kg$^{-1}$]', boundaries=bounds)
cbar.set_ticks(bounds)
# cbar.set_ticklabels(bounds_str)
    
plt.title('c) Mean $\Delta$ daily maximum CAPE')
plt.tight_layout()


fig.savefig(figpath+'delta_CAPEmean.png', dpi=300)#, format="svg")
plt.show()
plt.close()