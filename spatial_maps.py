#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:38:53 2024

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
import scipy as sc


c1='#648fff' #lightblue
c2='#785ef0' #indigo
c3='#dc267f' #magenta
c4='#fe6100' #orange
c5='#ffb000' #gold
c6='#000000' #black
c7='#f3322f' #red


#%% INIT

climate = "current"
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
r_disk = 50

subdomains = xr.open_dataset('/home/mfeldmann/Research/data/mesocyclone_climate/domain/subdomains_lonlat.nc')

startdays = ['0401','0501','0601','0701','0801','0901','1001','1101']
enddays = ['0430','0531','0630','0731','0831','0930','1031','1130']
snames = ['A','AM','MJ','JJ','JA','AS','SO','ON','N']
snames1 = ['April','May','June','July','August','September','October','November']
snames = ['APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
lnames = ['APR-1','MAY-1','JUN-1','JUL-1','AUG-1','SEP-1','OCT-1','NOV-1','DEC-1']
months = np.arange(4,12)

#%% All frequency
climate = "current"
path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
rainpath = "/media/mfeldmann@giub.local/Elements/supercell_climate/CT2/"+climate+"_climate/"
figpath = "/home/mfeldmann/Research/figs/mesocyclone_climate/"
snames = ['APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
lnames = ['APR-1','MAY-1','JUN-1','JUL-1','AUG-1','SEP-1','OCT-1','NOV-1','DEC-1']

r_disk = 5
counts=np.zeros([9,1542, 1542])
for i in range(len(snames)-1)[:]:
    filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate" + method + "SC_season" + '_' + snames[i] + "_test_disk" + str(r_disk) + ".nc"
    dataset = xr.open_dataset(filename_SC)
    lons = dataset.lon.values
    lats = dataset.lat.values
    counts_SC = dataset.frequency_map.values
    #plt.imshow(counts_SC); plt.show()
    counts[i,:,:]=counts_SC
countsum1 = np.nansum(counts, axis=0)/11
fmap.plot_fmap_simplified(lons, lats, countsum1, 'b) supercell frequency map', 'annual frequency within '+ str(r_disk)+' gridpoints', 
                          [0,0.05,0.5,0.7,1,1.5,2,2.5,3,3.5,4,4.5,5], [0,0.05,0.5,0.7,1,1.5,2,2.5,3,3.5,4,4.5,5], 
                          'CMRmap_r', figpath+climate+'frequency_h_disk' + str(r_disk)+'.png')

climate = "future"

counts=np.zeros([9,1542, 1542])
for i in range(len(snames)-1)[:]:
    filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate" + method + "SC_season" + '_' + snames[i] + "_test_disk" + str(r_disk) + ".nc"
    dataset = xr.open_dataset(filename_SC)
    lons = dataset.lon.values
    lats = dataset.lat.values
    counts_SC = dataset.frequency_map.values
    #plt.imshow(counts_SC); plt.show()
    counts[i,:,:]=counts_SC
countsum2 = np.nansum(counts, axis=0)/11
fmap.plot_fmap_simplified(lons, lats, countsum2, 'b) supercell frequency map', 'annual frequency within '+ str(r_disk)+' gridpoints', 
                          [0,0.05,0.5,0.7,1,1.5,2,2.5,3,3.5,4,4.5,5], [0,0.05,0.5,0.7,1,1.5,2,2.5,3,3.5,4,4.5,5], 
                          'CMRmap_r', figpath+climate+'frequency_h_disk' + str(r_disk)+'.png')
#%% delta frequency
delta = countsum2 - countsum1
binary = (countsum1> 3/11) * (countsum2> 3/11)
bounds=[-2,-1.7,-1.5,-1.2,-1,-0.7,-0.5,-0.2,-0.05,0.05,0.2,0.5,0.7,1,1.2,1.5,1.7,2]
fmap.plot_fmap_simplified(lons, lats, delta, 'a) $\Delta$ supercell frequency', 
                          '$\Delta$ annual frequency within '+ str(r_disk)+' gridpoints', bounds, bounds, 
                          'seismic', figpath+'frequency_delta_h_disk' + str(r_disk)+'.png', hatch=False, 
                          hatching=binary, delta=True)
#%%
r_disk=25
## diurnal eval ##
climate = "current"

counts=np.zeros([24,1542, 1542])
for i in range(len(start_hours)):
    start_hr = start_hours[i]
    filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate" + method + '_hourly_' + str(start_hr).zfill(2) + "h_disk" + str(r_disk) + ".nc"
    print(i,start_hr,filename_SC)
    dataset = xr.open_dataset(filename_SC)
    lons = dataset.lon.values
    lats = dataset.lat.values
    counts_SC = dataset.frequency_map.values
    #plt.imshow(counts_SC); plt.show()
    counts[i,:,:]=counts_SC

    
#     fmap.plot_fmap_simplified(lons, lats, counts_SC*11, str(start_hr).zfill(2)+':00h UTC', 'number of supercells', np.arange(0,11,1), np.arange(0,11,1), 'CMRmap_r', figpath+climate+str(start_hr).zfill(2)+'.png')
if r_disk==50:
    bounds = np.arange(0,3000,100); filt=100
elif r_disk==25:
    bounds = np.arange(0,1500,100); filt=10
else:
    bounds = np.arange(0,16,1); filt=3/11
fmap.plot_fmap_simplified(lons, lats, np.nansum(counts[:6],axis=0)*11, 'night', 
                          'number of supercells', bounds, bounds, 
                          'CMRmap_r', figpath+climate+'night_h_disk' + str(r_disk)+'.png')
fmap.plot_fmap_simplified(lons, lats, np.nansum(counts[6:12],axis=0)*11, 'morning', 
                          'number of supercells', bounds, bounds, 
                          'CMRmap_r', figpath+climate+'morning_h_disk' + str(r_disk)+'.png')
fmap.plot_fmap_simplified(lons, lats, np.nansum(counts[12:18],axis=0)*11, 'afternoon', 
                          'number of supercells', bounds, bounds, 
                          'CMRmap_r', figpath+climate+'afternoon_h_disk' + str(r_disk)+'.png')
fmap.plot_fmap_simplified(lons, lats, np.nansum(counts[18:],axis=0)*11, 'evening', 
                          'number of supercells', bounds, bounds, 
                          'CMRmap_r', figpath+climate+'evening_h_disk' + str(r_disk)+'.png')
weights = np.ones([3,1,1])
counts_c = sc.ndimage.convolve(counts,weights,mode='wrap')   
d1 = np.argmax(counts_c,axis=0) * 1.0
countsum = np.nansum(counts, axis=0)
d1[countsum<(0.00001)]=np.nan
binary1=countsum>filt#(3/11)

fmap.plot_fmap_simplified(lons, lats, d1, 'd) daily peak supercell occurrence', 'time of day [UTC]', 
                          np.arange(0,25,3), np.arange(0,25,3), 'twilight_shifted', 
                          figpath+climate+'diurnal_h_disk' + str(r_disk)+'.png', 
                          hatch=True, hatching=binary1, cycle=True, boundaries=np.arange(-3,28,3))


#%%
climate = "future"


counts=np.zeros([25,1542, 1542])
for i in range(len(start_hours)):
    start_hr = start_hours[i]
    filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate" + method + '_hourly_' + str(start_hr).zfill(2) + "h_disk" + str(r_disk) + ".nc"
    print(i,start_hr,filename_SC)
    dataset = xr.open_dataset(filename_SC)
    lons = dataset.lon.values
    lats = dataset.lat.values
    counts_SC = dataset.frequency_map.values
    #plt.imshow(counts_SC); plt.show()
    counts[i,:,:]=counts_SC
    
#     fmap.plot_fmap_simplified(lons, lats, counts_SC*11, str(start_hr).zfill(2)+':00h UTC', 
#                               'number of supercells', np.arange(0,11,1), np.arange(0,11,1), 
#                               'CMRmap_r', figpath+climate+str(start_hr).zfill(2)+'.png')

if r_disk==50:
    bounds = np.arange(0,3000,100); filt=100
elif r_disk==25:
    bounds = np.arange(0,1500,100); filt=10
else:
    bounds = np.arange(0,16,1); filt=3/11
fmap.plot_fmap_simplified(lons, lats, np.nansum(counts[:6],axis=0)*11, 'night', 
                          'number of supercells', bounds, bounds, 
                          'CMRmap_r', figpath+climate+'night_h_disk' + str(r_disk)+'.png')
fmap.plot_fmap_simplified(lons, lats, np.nansum(counts[6:12],axis=0)*11, 'morning', 
                          'number of supercells', bounds, bounds, 
                          'CMRmap_r', figpath+climate+'morning_h_disk' + str(r_disk)+'.png')
fmap.plot_fmap_simplified(lons, lats, np.nansum(counts[12:18],axis=0)*11, 'afternoon', 
                          'number of supercells', bounds, bounds, 
                          'CMRmap_r', figpath+climate+'afternoon_h_disk' + str(r_disk)+'.png')
fmap.plot_fmap_simplified(lons, lats, np.nansum(counts[18:],axis=0)*11, 'evening', 
                          'number of supercells', bounds, bounds, 
                          'CMRmap_r', figpath+climate+'evening_h_disk' + str(r_disk)+'.png')


weights = np.ones([3,1,1])
counts_c = sc.ndimage.convolve(counts,weights,mode='wrap')       
d2 = np.argmax(counts_c,axis=0) * 1.0
countsum = np.nansum(counts, axis=0)
d2[countsum<(0.00001)]=np.nan
binary2=countsum>filt#(3/11)

fmap.plot_fmap_simplified(lons, lats, d2, 'd) daily peak supercell occurrence', 'time of day [UTC]', 
                          np.arange(0,25,3), np.arange(0,25,3), 'twilight_shifted', 
                          figpath+climate+'diurnal_h_disk' + str(r_disk)+'.png', hatch=True, 
                          hatching=binary2, cycle=True, boundaries=np.arange(-3,28,3))


#%% DELTA PLOT DIURNAL
binary = binary1 * binary2
delta = d2 - d1
delta[delta>12]-=24
delta[delta<-12]+=24
fmap.plot_fmap_simplified(lons, lats, delta, '$\Delta$ daily peak supercell occurrence', 
                          '$\Delta$ time of day [h]', np.arange(-12.5,13.5), np.arange(-12.5,13.5), 
                          'seismic', figpath+'diurnal_delta_h_disk' + str(r_disk)+'.png', hatch=True, 
                          hatching=binary, delta=True)



        

    
#%% CURRENT SEASONAL
climate = "current"
path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
rainpath = "/media/mfeldmann@giub.local/Elements/supercell_climate/CT2/"+climate+"_climate/"
figpath = "/home/mfeldmann/Research/figs/mesocyclone_climate/"
snames = ['APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
lnames = ['APR-1','MAY-1','JUN-1','JUL-1','AUG-1','SEP-1','OCT-1','NOV-1','DEC-1']
if r_disk==50:
    bounds = np.arange(0,3000,100); filt=100
elif r_disk==25:
    bounds = np.arange(0,1000,50); filt=10
else:
    bounds = np.arange(0,16,1); filt=3/11
    
counts=np.zeros([9,1542, 1542])
for i in range(len(snames)-1)[:]:

    #filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate" + method + "SC_season" + '_' + snames[i] + "_test_disk" + str(r_disk) + ".nc"
    filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate" + method + "SC_season" + '_' + snames[i] + "_disk" + str(r_disk) + ".nc"
    dataset = xr.open_dataset(filename_SC)
    lons = dataset.lon.values
    lats = dataset.lat.values
    counts_SC = dataset.frequency_map.values
    counts[i,:,:]=counts_SC
    
    fmap.plot_fmap_simplified(lons, lats, counts_SC*11, snames1[i], 'number of supercells', 
                              bounds, bounds, 'CMRmap_r', 
                              figpath+climate+snames[i]+'_h_disk' + str(r_disk)+'.png')
    
d1 = np.argmax(counts,axis=0) * 1.0
countsum = np.nansum(counts, axis=0)
d1[countsum<(0.0001)]=np.nan
binary1=countsum>filt#(3/11)

fmap.plot_fmap_simplified(lons, lats, d1, 'c) annual peak supercell occurrence', 'month', 
                          np.arange(len(months)+1), lnames, 'twilight_shifted', 
                          figpath+climate+'seasonal_h_disk' + str(r_disk)+'.png', hatch=True, 
                          hatching=binary1, cycle=True, boundaries=np.arange(-1,len(months)+2,1))
#%% FUTURE SEASONAL
climate = "future"
path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
rainpath = "/media/mfeldmann@giub.local/Elements/supercell_climate/CT2/"+climate+"_climate/"
figpath = "/home/mfeldmann/Research/figs/mesocyclone_climate/"

if r_disk==50:
    bounds = np.arange(0,3000,100); filt=100
elif r_disk==25:
    bounds = np.arange(0,1000,50); filt=10
else:
    bounds = np.arange(0,16,1); filt=3/11
counts=np.zeros([9,1542, 1542])
for i in range(len(snames)-1)[:]:
    #filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate" + method + "SC_season" + '_' + snames[i] + "_test_disk" + str(r_disk) + ".nc"
    filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate" + method + "SC_season" + '_' + snames[i] + "_disk" + str(r_disk) + ".nc"
    
    dataset = xr.open_dataset(filename_SC)
    lons = dataset.lon.values
    lats = dataset.lat.values
    counts_SC = dataset.frequency_map.values
    counts[i,:,:]=counts_SC
    fmap.plot_fmap_simplified(lons, lats, counts_SC*11, snames1[i], 'number of supercells', 
                              bounds, bounds, 'CMRmap_r', 
                              figpath+climate+snames[i]+'.png')
    
    
d2 = np.argmax(counts,axis=0) * 1.0
countsum = np.nansum(counts, axis=0)
d2[countsum<(0.0001)]=np.nan
binary2=countsum>filt#(3/11)

fmap.plot_fmap_simplified(lons, lats, d2, 'c) annual peak supercell occurrence', 'month', 
                          np.arange(len(months)+1), lnames, 'twilight_shifted', 
                          figpath+climate+'seasonal_h_disk' + str(r_disk)+'.png', hatch=True, 
                          hatching=binary1, cycle=True, boundaries=np.arange(-1,len(months)+2,1))

#%% DELTA SEASONAL
binary = binary1 * binary2
delta = d2 - d1
fmap.plot_fmap_simplified(lons, lats, delta, '$\Delta$ annual peak supercell occurrence', 
                          '$\Delta$ time of year [months]', np.arange(-7.5,8.5), np.arange(-7.5,8.5), 
                          'seismic', figpath+'seasonal_delta_h_disk' + str(r_disk)+'.png', hatch=True, 
                          hatching=binary, delta=True)

#%% MOVERS
r_disk = 50
climate = "current"
filename_SC_r = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate_rightmover_h_disk" + str(r_disk) + ".nc"
filename_SC_l = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate_leftmover_h_disk" + str(r_disk) + ".nc"
counts_r = xr.open_dataset(filename_SC_r).frequency_map.values
counts_l = xr.open_dataset(filename_SC_l).frequency_map.values
ratio_cc = counts_r / (counts_r+counts_l)
counts = counts_r+counts_l

binary1 = counts>50

fmap.plot_fmap_simplified(lons, lats, ratio_cc, 'percentage of right movers', '[%]', 
                          np.arange(-0.05,1.05,0.1), [0,5,15,25,35,45,55,65,75,85,95], 'seismic', 
                          figpath+climate+'mover_h_disk' + str(r_disk)+'.png', hatch=True, hatching=binary1)

climate = "future"
filename_SC_r = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate_rightmover_h_disk" + str(r_disk) + ".nc"
filename_SC_l = "/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate_leftmover_h_disk" + str(r_disk) + ".nc"
counts_r = xr.open_dataset(filename_SC_r).frequency_map.values
counts_l = xr.open_dataset(filename_SC_l).frequency_map.values
ratio_fc = counts_r / (counts_r+counts_l)
counts = counts_r+counts_l

binary2 = counts>50

fmap.plot_fmap_simplified(lons, lats, ratio_fc, 'percentage of right movers', '[%]', 
                          np.arange(-0.05,1.05,0.1), [0,5,15,25,35,45,55,65,75,85,95], 'seismic', 
                          figpath+climate+'mover_h_disk' + str(r_disk)+'.png', hatch=True, hatching=binary2)

#%% DELTA MOVERS

binary = binary1 * binary2
delta = ratio_fc - ratio_cc

fmap.plot_fmap_simplified(lons, lats, delta, '$\Delta$ percentage of right movers', 
                          '[%]', np.arange(-0.55,0.59,0.05), np.arange(-55,59,5),#[-55,-45,-35,-25,-15,-5,5,15,25,35,45,55], 
                          'seismic', figpath+'mover_delta_h_disk' + str(r_disk)+'.png', hatch=True, 
                          hatching=binary, delta=True)


#%% obs vs mod

r_disk = 5
climate = "current"
years = ["2016", "2017", "2018", "2019", "2020", "2021"]

counts=np.zeros([len(years),1542, 1542])
for i in range(len(years)):
    season = years[i]
    start_hr = start_hours[i]
    filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/model_"+climate+"_climate_" + str(season)+"h_disk" + str(r_disk) + ".nc"
    dataset = xr.open_dataset(filename_SC)
    lons = dataset.lon.values
    lats = dataset.lat.values
    counts_SC = dataset.frequency_map.values
    #plt.imshow(counts_SC); plt.show()
    counts[i,:,:]=counts_SC
dataset.close()
countsum1 = np.nansum(counts, axis=0)/len(years)
fmap.plot_fmap_simplified(lons, lats, countsum1, ' ', 'annual frequency within '+ str(r_disk)+' gridpoints', 
                          [0,0.1,0.5,0.7,1,1.5,2,2.5,3,3.5,4,4.5,5,10], [0,0.1,0.5,0.7,1,1.5,2,2.5,3,3.5,4,4.5,5,10], 
                          'CMRmap_r', figpath+climate+'model_h_disk' + str(r_disk)+'.png',CH=True,cbar='horizontal')


counts=np.zeros([24,1542, 1542])
for i in range(len(years)):
    season = years[i]
    start_hr = start_hours[i]
    filename_SC = "/home/mfeldmann/Research/data/mesocyclone_climate/radar_climate_" + str(season)+"h_disk" + str(r_disk) + "hfilt.nc"
    dataset = xr.open_dataset(filename_SC)
    lons = dataset.lon.values
    lats = dataset.lat.values
    counts_SC = dataset.frequency_map.values
    #plt.imshow(counts_SC); plt.show()
    counts[i,:,:]=counts_SC
dataset.close()
countsum2 = np.nansum(counts, axis=0)/len(years)
fmap.plot_fmap_simplified(lons, lats, countsum2, ' ', 'annual frequency within '+ str(r_disk)+' gridpoints', 
                          [0,0.1,0.5,1,1.5,2,2.5,3,5,7.5,10,15], [0,0.1,0.5,1,1.5,2,2.5,3,5,7.5,10,15], 
                          'CMRmap_r', figpath+climate+'radar_h_disk' + str(r_disk)+'hfilt.png',CH=True,cbar='horizontal')



#%%
# Abbrev;Name;Y;X;Z
# A;Albis;681201;237604;938
# D;La Dole;497057;142408;1682
# L;Monte Lema;707957;99762;1626
# P;Plaine Morte;603687;135476;2937
# W;Weissfluhgipfel;779700;189790;2850

# albis - 8.511999921 - 47.284333348
# dole - 6.099415777 - 46.425102252
# lema - 8.833216354 - 46.040754913
# plaine morte- 7.486550845 - 46.370650060
# weissfluh - 9.794466983 - 46.834974607
