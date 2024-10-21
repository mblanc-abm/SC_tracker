#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:40:57 2024

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
start_hours = np.arange(24)
end_hours = np.arange(24)+2
#years = ["2019", "2020", "2021"]
method = "model_tracks"
##iuh_thresh = args.iuh_thresh#
path = "/scratch/snx3000/mblanc/SDT/SDT2_output/" + climate + "_climate/domain/XPT_1MD_zetath5_wth5/"
path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
rainpath = "/media/mfeldmann@giub.local/Elements/supercell_climate/CT2/"+climate+"_climate/"
figpath = "/home/mfeldmann/Research/figs/mesocyclone_climate/"
r_disk = 25

subdomains = xr.open_dataset('/home/mfeldmann/Research/data/mesocyclone_climate/domain/subdomains_lonlat.nc')

#%% DIURNAL 
climate = "current"
path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
for i in range(len(start_hours))[11:]:
        start_hr = start_hours[i]
        end_hr = start_hours[i]
        lons, lats, counts_SC = fmap.diurnal_supercell_tracks_model_fmap(start_hr, end_hr, path, r_disk)
        filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate" + method + '_hourly_' + str(start_hr).zfill(2) + "h_disk" + str(r_disk) + ".nc"
        fmap.write_to_netcdf(lons, lats, counts_SC, filename_SC)
climate = "future"
path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
for i in range(len(start_hours))[:]:
        start_hr = start_hours[i]
        end_hr = start_hours[i]
        lons, lats, counts_SC = fmap.diurnal_supercell_tracks_model_fmap(start_hr, end_hr, path, r_disk)
        filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate" + method + '_hourly_' + str(start_hr).zfill(2) + "h_disk" + str(r_disk) + ".nc"
        fmap.write_to_netcdf(lons, lats, counts_SC, filename_SC)

#%%  SEASONAL 
climate = "current"
if climate == "current": years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
if climate == "future": years = ['2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095']
path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
# r_disk = 50

startdays = ['0401','0501','0601','0701','0801','0901','1001','1101']
enddays = ['0430','0531','0630','0731','0831','0930','1031','1130']
snames = ['A','AM','MJ','JJ','JA','AS','SO','ON','N']
snames1 = ['April','May','June','July','August','September','October','November']
snames = ['APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV']
months = np.arange(4,12)

for i in range(len(months))[:]:
        lons, lats, counts_SC = fmap.seasonal_supercell_tracks_model_fmap(months[i], path, r_disk)
        filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/" +climate + "_climate" + method + "SC_season" + '_' + snames[i] + "_disk" + str(r_disk) + ".nc"
        fmap.write_to_netcdf(lons, lats, counts_SC, filename_SC)


climate = "future"
path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
# r_disk = 50

startdays = ['0401','0501','0601','0701','0801','0901','1001','1101']
enddays = ['0430','0531','0630','0731','0831','0930','1031','1130']
snames = ['A','AM','MJ','JJ','JA','AS','SO','ON','N']
snames1 = ['April','May','June','July','August','September','October','November']
snames = ['APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV']
months = np.arange(4,12)

for i in range(len(months))[:]:
        lons, lats, counts_SC = fmap.seasonal_supercell_tracks_model_fmap(months[i], path, r_disk)
        filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/" +climate + "_climate" + method + "SC_season" + '_' + snames[i] + "_disk" + str(r_disk) + ".nc"
        fmap.write_to_netcdf(lons, lats, counts_SC, filename_SC)
        
climate = "future"
path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
r_disk = 50

startdays = ['0401','0501','0601','0701','0801','0901','1001','1101']
enddays = ['0430','0531','0630','0731','0831','0930','1031','1130']
snames = ['A','AM','MJ','JJ','JA','AS','SO','ON','N']
snames1 = ['April','May','June','July','August','September','October','November']
snames = ['APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV']
months = np.arange(4,12)

for i in range(len(months))[:]:
        lons, lats, counts_SC = fmap.seasonal_supercell_tracks_model_fmap(months[i], path, r_disk)
        filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/" +climate + "_climate" + method + "SC_season" + '_' + snames[i] + "_disk" + str(r_disk) + ".nc"
        fmap.write_to_netcdf(lons, lats, counts_SC, filename_SC)
        
#%% radar_obs

start_day = "0401"
end_day = "1130"
years = ["2016", "2017", "2018", "2019", "2020", "2021"]
path1 = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/current_climate/domain/XPT_1MD_zetath5_wth5"
path2 = "/media/mfeldmann@giub.local/Elements/supercell_climate/"

figpath = "/home/mfeldmann/Research/figs/mesocyclone_climate/"
r_disk = 5

for season in years:
    day_i = season + start_day
    day_f = season + end_day
    lons, lats, counts_SC = fmap.seasonal_supercell_tracks_obs_fmap(day_i, day_f, path1, path2, r_disk)
    filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/radar_climate_" + str(season)+"h_disk" + str(r_disk) + "hfilt.nc"
    fmap.write_to_netcdf(lons, lats, counts_SC, filename_SC)


#%% model

climate = "future"
start_day = "0401"
end_day = "1130"
if climate == "current": years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
if climate == "future": years = ['2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095']
path1 = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/current_climate/domain/XPT_1MD_zetath5_wth5"
path2 = "/media/mfeldmann@giub.local/Elements/supercell_climate/"
path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"

figpath = "/home/mfeldmann/Research/figs/mesocyclone_climate/"
r_disk = 5

for season in years:
    print(season)
    day_i = season + start_day
    day_f = season + end_day
    lons, lats, counts_SC = fmap.yearly_supercell_tracks_model_fmap(day_i, day_f, path, r_disk, twoMD=False, skipped_days=None)
    filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/model_"+climate+"_climate_" + str(season)+"h_disk" + str(r_disk) + ".nc"
    fmap.write_to_netcdf(lons, lats, counts_SC, filename_SC)
#%% RIGHT AND LEFT MOVERS
# climate = "future"
# start_day = "20900401"
# end_day = "20901130"
# if climate == "current": years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
# if climate == "future": years = ['2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095']
# path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
# rainpath = "/media/mfeldmann@giub.local/Elements/supercell_climate/CT2/"+climate+"_climate/"
# figpath = "/home/mfeldmann/Research/figs/mesocyclone_climate/"
# r_disk=50


# lons, lats, counts_r, counts_l = fmap.mover_supercell_tracks_model_fmap(start_day, end_day, path, r_disk, twoMD=False, skipped_days=None)

# filename_SC_r = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate_rightmover_h_disk" + str(r_disk) + ".nc"
# filename_SC_l = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate_leftmover_h_disk" + str(r_disk) + ".nc"

# fmap.write_to_netcdf(lons, lats, counts_r, filename_SC_r)
# fmap.write_to_netcdf(lons, lats, counts_l, filename_SC_l)


# climate = "current"
# start_day = "20200401"
# end_day = "20201130"
# if climate == "current": years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
# if climate == "future": years = ['2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095']
# path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
# rainpath = "/media/mfeldmann@giub.local/Elements/supercell_climate/CT2/"+climate+"_climate/"
# figpath = "/home/mfeldmann/Research/figs/mesocyclone_climate/"
# r_disk=50


# lons, lats, counts_r, counts_l = fmap.mover_supercell_tracks_model_fmap(start_day, end_day, path, r_disk, twoMD=False, skipped_days=None)

# filename_SC_r = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate_rightmover_h_disk" + str(r_disk) + ".nc"
# filename_SC_l = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate_leftmover_h_disk" + str(r_disk) + ".nc"

# fmap.write_to_netcdf(lons, lats, counts_r, filename_SC_r)
# fmap.write_to_netcdf(lons, lats, counts_l, filename_SC_l)