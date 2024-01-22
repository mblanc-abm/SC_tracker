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

from Scell_tracker import label_above_threshold, find_overlaps

import sys
sys.path.append("../first_visu")
from CaseStudies import IUH

#================================================================================================================================
#inputs
dtstr = "20190611130000" # to be filled: full date and time of timeshot
cut = "largecut"
dt = pd.to_datetime(dtstr, format="%Y%m%d%H%M%S")
aura = 2
threshold = 75
sub_threshold = 65
min_area = 3

#================================================================================================================================
## test label_above_threshold

fp = "/scratch/snx3000/mblanc/UHfiles/" + cut + "_lffd" + dtstr + "p.nc"
fs = "/scratch/snx3000/mblanc/UHfiles/" + cut + "_PSlffd" + dtstr + ".nc"
iuh = np.array(IUH(fp, fs))
iuh[abs(iuh)<50] = np.nan
iuh_max = 150 # set here the maximum (or minimum in absolute value) IUH that you want to display
norm_iuh = TwoSlopeNorm(vmin=-iuh_max, vcenter=0, vmax=iuh_max)
ticks_iuh = np.arange(-iuh_max, iuh_max+1, 25)

labeled = label_above_threshold(iuh, threshold, sub_threshold, min_area)
labeled_disp = expand_labels(labeled, distance=aura)
labeled_disp = np.where(labeled_disp == 0, np.nan, labeled_disp)

vmin = 0.5
ncells = int(np.max(labeled))
vmax = ncells + 0.5
ticks_lab = np.arange(0, ncells+1)
levels_lab = np.linspace(vmin, vmax, ncells+1)
cmap_lab = plt.cm.get_cmap('tab10')
colors = [cmap_lab(i % 10) for i in range(10*int(np.ceil(ncells/10)))]
cmap_lab = ListedColormap(colors)

with xr.open_dataset(fs) as dset:
    lons = dset["lon"].values
    lats = dset["lat"].values

#================================================================================================================================
## test find_overlaps
maskfile = "/scratch/snx3000/mblanc/cell_tracker/CaseStudies/outfiles/cell_masks_" + dt.strftime("%Y%m%d") + ".nc"
with xr.open_dataset(maskfile) as dset:
    masks = dset['cell_mask'] # 3D matrix
    times = pd.to_datetime(dset['time'].values)

# select hourly masks and times -> decrease temporal resolution to match the IUH one
time_slice = times == dt
mask = masks[time_slice][0]

vmin = -0.5
ncells = int(np.max(mask)+1)
vmax = (ncells - 1) + 0.5
ticks_mask = np.arange(0, ncells)
levels_mask = np.linspace(vmin, vmax, ncells+1)
cmap_mask = plt.cm.get_cmap('tab20')
colors = [cmap_mask(i % 20) for i in range(20*int(np.ceil(ncells/20)))]
cmap_mask = ListedColormap(colors)

overlaps = find_overlaps(iuh, labeled, mask, aura, printout=True)

#================================================================================================================================
## plot

# load geographic features
resol = '10m'  # use data at this scale
bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])

fig = plt.figure(figsize=(6,12))

ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
ax1.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
ax1.add_feature(ocean, linewidth=0.2)
cont_iuh = ax1.pcolormesh(lons, lats, iuh, cmap="RdBu_r", norm=norm_iuh, transform=ccrs.PlateCarree())
plt.colorbar(cont_iuh, ticks=ticks_iuh, orientation='horizontal', label=r"IUH ($m^2/s^2$)")
plt.title(dt.strftime("%d/%m/%Y %H:%M:%S"))

ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
cont = ax2.contourf(lons, lats, labeled_disp, cmap=cmap_lab, levels=levels_lab, transform=ccrs.PlateCarree())
ax2.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
ax2.add_feature(ocean, linewidth=0.2)
plt.colorbar(cont, ticks=ticks_lab, orientation='horizontal', label="supercells labels, aura="+str(aura))

ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
cont_mask = ax3.contourf(lons, lats, mask, levels=levels_mask, cmap=cmap_mask, transform=ccrs.PlateCarree())
ax3.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
ax3.add_feature(ocean, linewidth=0.2)
cbar = plt.colorbar(cont_mask, ticks=ticks_mask, orientation='horizontal', label="Cell mask")
cbar.locator = MultipleLocator(base=5)
