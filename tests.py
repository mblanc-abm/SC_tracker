import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#import cartopy
#import json
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.ticker import MultipleLocator
from skimage.segmentation import expand_labels

from Scell_tracker import label_above_thresholds, find_vortex_rain_overlaps

import sys
sys.path.append("../first_visu")
from CaseStudies import zeta_plev

#================================================================================================================================
# FUNCTIONS

def plot_vortex_masks(lons, lats, labeled, cmap, levels, ticks, typ, dt):
    
    resol = '10m'
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    rp = ccrs.RotatedPole(pole_longitude = -170, pole_latitude = 43)

    plt.figure()#figsize=(6,12))
    ax = plt.axes(projection=rp)
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    cont = ax.contourf(lons, lats, labeled, cmap=cmap, levels=levels, transform=ccrs.PlateCarree())
    plt.colorbar(cont, ticks=ticks, orientation='horizontal', label = typ + " vorticies")
    plt.title(dt.strftime("%d/%m/%Y %H:%M:%S"))

#================================================================================================================================
#inputs
dtstr = "20210620150000" # to be filled: full date and time of timeshot
dt = pd.to_datetime(dtstr, format="%Y%m%d%H%M%S")
aura = 1
zeta_th = 4e-3
w_th = 6
min_area = 3

#================================================================================================================================
## test label_above_thresholds

fp = "/scratch/snx3000/mblanc/SDT/infiles/CaseStudies/1h_3D_plev/cut_lffd" + dtstr + "p.nc"
fs = "/scratch/snx3000/mblanc/SDT/infiles/CaseStudies/1h_2D/cut_PSlffd" + dtstr + ".nc"

zeta_400 = zeta_plev(fp, fs, 2)
zeta_500 = zeta_plev(fp, fs, 3)
zeta_600 = zeta_plev(fp, fs, 4)
zeta_700 = zeta_plev(fp, fs, 5)
zeta_3lev = np.stack([zeta_400, zeta_500, zeta_600, zeta_700])

with xr.open_dataset(fp) as dset:
    lons = dset["lon"].values
    lats = dset["lat"].values
    w_400 = dset['W'][0][2]
    w_500 = dset['W'][0][3]
    w_600 = dset['W'][0][4]
    w_700 = dset['W'][0][5]
w_3lev = np.stack([w_400, w_500, w_600, w_700])

labeled_pos = label_above_thresholds(zeta_3lev, w_3lev, zeta_th, w_th, min_area, aura, "positive")
labeled_neg = label_above_thresholds(zeta_3lev, w_3lev, zeta_th, w_th, min_area, aura, "negative")
labeled_pos = np.where(labeled_pos == 0, np.nan, labeled_pos)
labeled_neg = np.where(labeled_neg == 0, np.nan, labeled_neg)

## plot positive vorticies

if not np.isnan(np.nanmax(labeled_pos)):
    vmin = 0.5
    ncells = int(np.nanmax(labeled_pos))
    vmax = ncells + 0.5
    ticks = np.arange(0, ncells+1)
    levels = np.linspace(vmin, vmax, ncells+1)
    cmap = plt.cm.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(10*int(np.ceil(ncells/10)))]
    cmap = ListedColormap(colors)
    plot_vortex_masks(lons, lats, labeled_pos, cmap, levels, ticks, "positive", dt)

if not np.isnan(np.nanmax(labeled_neg)):
    vmin = 0.5
    ncells = int(np.nanmax(labeled_neg))
    vmax = ncells + 0.5
    ticks = np.arange(0, ncells+1)
    levels = np.linspace(vmin, vmax, ncells+1)
    cmap = plt.cm.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(10*int(np.ceil(ncells/10)))]
    cmap = ListedColormap(colors)
    plot_vortex_masks(lons, lats, labeled_neg, cmap, levels, ticks, "negative", dt)

# ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
# ax1.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
# cont_iuh = ax1.pcolormesh(lons, lats, iuh, cmap="RdBu_r", norm=norm_iuh, transform=ccrs.PlateCarree())
# plt.colorbar(cont_iuh, ticks=ticks_iuh, extend='both', orientation='horizontal', label=r"IUH ($m^2/s^2$)")
# plt.title(dt.strftime("%d/%m/%Y %H:%M:%S"))

# ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
# cont = ax2.contourf(lons, lats, labeled_disp, cmap=cmap_lab, levels=levels_lab, transform=ccrs.PlateCarree())
# ax2.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
# cbar = plt.colorbar(cont, ticks=ticks_lab, orientation='horizontal', label="supercells labels, aura="+str(aura))
# cbar.locator = MultipleLocator(base=2)

# ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
# cont_mask = ax3.contourf(lons, lats, mask, levels=levels_mask, cmap=cmap_mask, transform=ccrs.PlateCarree())
# ax3.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
# cbar = plt.colorbar(cont_mask, ticks=ticks_mask, orientation='horizontal', label="Cell mask")
# cbar.locator = MultipleLocator(base=5)

# fig.savefig(dtstr+".png", dpi=300)


#================================================================================================================================
## test find_overlaps
# maskfile = "/scratch/snx3000/mblanc/CaseStudies/outfiles_CT1/cell_masks_" + dt.strftime("%Y%m%d") + ".nc"
# with xr.open_dataset(maskfile) as dset:
#     masks = dset['cell_mask'] # 3D matrix
#     times = pd.to_datetime(dset['time'].values)

# # select hourly masks and times -> decrease temporal resolution to match the IUH one
# time_slice = times == dt
# mask = masks[time_slice][0]

# vmin = -0.5
# ncells = int(np.max(mask)+1)
# vmax = (ncells - 1) + 0.5
# ticks_mask = np.arange(0, ncells)
# levels_mask = np.linspace(vmin, vmax, ncells+1)
# cmap_mask = plt.cm.get_cmap('tab20')
# colors = [cmap_mask(i % 20) for i in range(20*int(np.ceil(ncells/20)))]
# cmap_mask = ListedColormap(colors)

# overlaps = find_overlaps(iuh, labeled, mask, aura, printout=True)

# #================================================================================================================================
# ## test meso masks

# with xr.open_dataset("/scratch/snx3000/mblanc/SDT_output/seasons/2020/meso_masks_20200401.nc") as dset:
#     meso_masks = dset['meso_mask']
#     lats = dset['lat']
#     lons = dset['lon']
#     times = pd.to_datetime(dset['time'])

# # load geographic features
# resol = '10m'  # use data at this scale
# bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
# ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])

# # plot
# for i, meso_mask in enumerate(meso_masks):
#     plt.figure()
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
#     ax.add_feature(ocean, linewidth=0.2)
#     cont = plt.pcolormesh(lons, lats, meso_mask, cmap="Blues", transform=ccrs.PlateCarree())
#     plt.colorbar(cont)
#     plt.suptitle(times[i].strftime("%d/%m/%Y %H:%M:%S"))

