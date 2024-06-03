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

from Scell_tracker import label_above_thresholds, find_vortex_rain_overlaps, merge_dictionaries

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


def plot_vortex_rain_masks(lons, lats, labeled_vx, cmap_vx, levels_vx, ticks_vx, typ_vx, dt, cmap_rm, levels_rm, ticks_rm, rain_mask):
    
    resol = '10m'
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    rp = ccrs.RotatedPole(pole_longitude = -170, pole_latitude = 43)

    fig = plt.figure(figsize=(6,9))
    
    ax1 = fig.add_subplot(2, 1, 1, projection=rp)
    ax1.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    cont_vx = ax1.contourf(lons, lats, labeled_vx, cmap=cmap_vx, levels=levels_vx, transform=ccrs.PlateCarree())
    plt.colorbar(cont_vx, ticks=ticks_vx, orientation='horizontal', label = typ_vx + " vorticies masks")
    plt.title(dt.strftime("%d/%m/%Y %H:%M:%S"))
    
    ax2 = fig.add_subplot(2, 1, 2, projection=rp)
    cont_rm = ax2.contourf(lons, lats, rain_mask, levels=levels_rm, cmap=cmap_rm, transform=ccrs.PlateCarree())
    ax2.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    cbar = plt.colorbar(cont_rm, ticks=ticks_rm, orientation='horizontal', label="Rain masks")
    cbar.locator = MultipleLocator(base=5)

#================================================================================================================================
#inputs
dtstr = "20170801180000" # to be filled: full date and time of timeshot
dt = pd.to_datetime(dtstr, format="%Y%m%d%H%M%S")
aura = 1
zeta_th = 4e-3
zeta_pk = 6e-3
w_th = 6
min_area = 3

#================================================================================================================================
## test label_above_thresholds and find_vortex_rain_overlaps ##

# load zeta and w data

fp = "/scratch/snx3000/mblanc/SDT/infiles/CaseStudies/1h_3D_plev/cut_lffd" + dtstr + "p.nc"
fs = "/scratch/snx3000/mblanc/SDT/infiles/CaseStudies/1h_2D/cut_PSlffd" + dtstr + ".nc"

zeta_400 = zeta_plev(fp, fs, 2)
zeta_500 = zeta_plev(fp, fs, 3)
zeta_600 = zeta_plev(fp, fs, 4)
zeta_700 = zeta_plev(fp, fs, 5)
zeta_4lev = np.stack([zeta_400, zeta_500, zeta_600, zeta_700])

with xr.open_dataset(fp) as dset:
    lons = dset["lon"].values
    lats = dset["lat"].values
    w_400 = dset['W'][0][2]
    w_500 = dset['W'][0][3]
    w_600 = dset['W'][0][4]
    w_700 = dset['W'][0][5]
w_4lev = np.stack([w_400, w_500, w_600, w_700])

# load rain mask
maskfile = "/scratch/snx3000/mblanc/CT_CSs/outfiles_CT2PT/cell_masks_" + dt.strftime("%Y%m%d") + ".nc"
with xr.open_dataset(maskfile) as dset:
    masks = dset['cell_mask'] # 3D matrix
    times = pd.to_datetime(dset['time'].values)
    
# select hourly masks and times -> decrease temporal resolution to match the IUH one
time_slice = times == dt
mask = masks[time_slice][0]

# prepare mask metadata
vmin = -0.5
ncells = int(np.max(mask)+1)
vmax = (ncells - 1) + 0.5
ticks_mask = np.arange(0, ncells)
levels_mask = np.linspace(vmin, vmax, ncells+1)
cmap_mask = plt.cm.get_cmap('tab20')
colors = [cmap_mask(i % 20) for i in range(20*int(np.ceil(ncells/20)))]
cmap_mask = ListedColormap(colors)

# search and label vortices, and find overlaps
labeled_pos = label_above_thresholds(zeta_4lev, w_4lev, zeta_th, zeta_pk, w_th, min_area, aura, 1)
labeled_neg = label_above_thresholds(zeta_4lev, w_4lev, zeta_th, zeta_pk, w_th, min_area, aura, -1)
overlaps_pos, no_overlaps_pos = find_vortex_rain_overlaps(zeta_4lev, w_4lev, labeled_pos, mask, zeta_th, w_th, 1)
overlaps_neg, no_overlaps_neg = find_vortex_rain_overlaps(zeta_4lev, w_4lev, labeled_neg, mask, zeta_th, w_th, -1)
labeled_pos = np.where(labeled_pos == 0, np.nan, labeled_pos)
labeled_neg = np.where(labeled_neg == 0, np.nan, labeled_neg)
overlaps = merge_dictionaries(overlaps_pos, overlaps_neg)
no_overlaps = merge_dictionaries(no_overlaps_pos, no_overlaps_neg)

print(overlaps)
print(no_overlaps)

## plot positive and negative vorticies, if any, together with the rain mask

if not np.isnan(np.nanmax(labeled_pos)):
    vmin = 0.5
    ncells = int(np.nanmax(labeled_pos))
    vmax = ncells + 0.5
    ticks_vx = np.arange(0, ncells+1)
    levels_vx = np.linspace(vmin, vmax, ncells+1)
    cmap_vx = plt.cm.get_cmap('tab10')
    colors = [cmap_vx(i % 10) for i in range(10*int(np.ceil(ncells/10)))]
    cmap_vx = ListedColormap(colors)
    #plot_vortex_masks(lons, lats, labeled_pos, cmap_vx, levels_vx, ticks_vx, "positive", dt)
    plot_vortex_rain_masks(lons, lats, labeled_pos, cmap_vx, levels_vx, ticks_vx, "positive", dt, cmap_mask, levels_mask, ticks_mask, mask)

if not np.isnan(np.nanmax(labeled_neg)):
    vmin = 0.5
    ncells = int(np.nanmax(labeled_neg))
    vmax = ncells + 0.5
    ticks_vx = np.arange(0, ncells+1)
    levels_vx = np.linspace(vmin, vmax, ncells+1)
    cmap_vx = plt.cm.get_cmap('tab10')
    colors = [cmap_vx(i % 10) for i in range(10*int(np.ceil(ncells/10)))]
    cmap_vx = ListedColormap(colors)
    #plot_vortex_masks(lons, lats, labeled_neg, cmap_vx, levels_vx, ticks_vx, "negative", dt)
    plot_vortex_rain_masks(lons, lats, labeled_neg, cmap_vx, levels_vx, ticks_vx, "negative", dt, cmap_mask, levels_mask, ticks_mask, mask)

#==============================================================================================================================
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

