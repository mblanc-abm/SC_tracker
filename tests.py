import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import json
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.ticker import MultipleLocator
from Scell_tracker import label_local_maxima, label_above_threshold, find_overlaps

import sys
sys.path.append("../first_visu")
from CaseStudies import IUH

# ================================================================================================================================
#inputs
dtstr = "20210713110000" # to be filled: full date and time of timeshot
cut = "swisscut"
dt = pd.to_datetime(dtstr, format="%Y%m%d%H%M%S")
aura = 1


# load geographic features
resol = '10m'  # use data at this scale
bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])

# ================================================================================================================================
## test label_above_threshold

fp = "/scratch/snx3000/mblanc/UHfiles/" + cut + "_lffd" + dtstr + "p.nc"
fs = "/scratch/snx3000/mblanc/UHfiles/" + cut + "_PSlffd" + dtstr + ".nc"
iuh = np.array(IUH(fp, fs))
iuh[abs(iuh)<50] = np.nan
iuh_max = 150 # set here the maximum (or minimum in absolute value) IUH that you want to display
norm_iuh = TwoSlopeNorm(vmin=-iuh_max, vcenter=0, vmax=iuh_max)
ticks_iuh = np.arange(-iuh_max, iuh_max+1, 25)

labeled = label_above_threshold(iuh, 75, 60, 3)
labeled_disp = np.where(labeled == 0, np.nan, labeled)

with xr.open_dataset(fs) as dset:
    lons = dset["lon"].values
    lats = dset["lat"].values

# plot
fig = plt.figure(figsize=(6,8))

ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax1.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
ax1.add_feature(ocean, linewidth=0.2)
cont_iuh = ax1.pcolormesh(lons, lats, iuh, cmap="RdBu_r", norm=norm_iuh, transform=ccrs.PlateCarree())
plt.colorbar(cont_iuh, ticks=ticks_iuh, orientation='horizontal', label=r"IUH ($m^2/s^2$)")
plt.title(dt.strftime("%d/%m/%Y %H:%M:%S"))

ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
cont = ax2.pcolormesh(lons, lats, labeled_disp, cmap="tab20", transform=ccrs.PlateCarree())
ax2.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
ax2.add_feature(ocean, linewidth=0.2)
plt.colorbar(cont, orientation='horizontal', label="supercells labels, aura=" + str(aura))

# ================================================================================================================================
## test find_overlaps
maskfile = "/scratch/snx3000/mblanc/cell_tracker/outfiles/cell_masks_" + dt.strftime("%Y%m%d") + ".nc"
with xr.open_dataset(maskfile) as dset:
    lats = dset.variables['lat']
    lons = dset.variables['lon']
    masks = dset['cell_mask'] # 3D matrix
    times = pd.to_datetime(dset['time'].values)
vmin = -0.5
ncells = int(np.max(masks)+1)
vmax = (ncells - 1) + 0.5
ticks_mask = np.arange(0, ncells)
levels_mask = np.linspace(vmin, vmax, ncells+1)
cmap_mask = plt.cm.get_cmap('tab20')
colors = [cmap_mask(i % 20) for i in range(20*int(np.ceil(ncells/20)))]
cmap_mask = ListedColormap(colors)

# select hourly masks and times -> decrease temporal resolution to match the IUH one
time_slice = times.strftime("%H%M%S") == dt.strftime('%H%M%S')
mask = masks[time_slice][0]
times = times[time_slice]

overlaps = find_overlaps(iuh, labeled, mask, aura, printout=True)

