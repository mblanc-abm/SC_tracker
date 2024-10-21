#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:01:04 2024

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
matplotlib.rcParams.update({'font.size': 18})
import seaborn as sns

c1='#648fff' #lightblue
c2='#785ef0' #indigo
c3='#dc267f' #magenta
c4='#fe6100' #orange
c5='#ffb000' #gold
c6='#000000' #black
c7='#f3322f' #red

#%%
def assign_regions(lats,lons,subdomains):
    
    rp = ccrs.RotatedPole(pole_longitude = -170, pole_latitude = 43)
    regular_coords = ccrs.PlateCarree()
    transformed_coords = rp.transform_points(regular_coords, lons, lats)
    rlons = transformed_coords[:, 0]
    rlats = transformed_coords[:, 1]
    
    rvec = pd.DataFrame(data = None, index = np.arange(len(lats)), columns = list(subdomains.keys())[2:])

    for n in range(len(lats)):
        tlon = rlons[n]
        tlat = rlats[n]
        for dom in list(subdomains.keys())[2:]:
            # AL which is union of NAL and SAL is treated differently, also MD which is union of MDS and MDL
            if dom == 'AL' or dom == 'MD':
                continue
            test = (
                subdomains[dom]
                .sel(
                    rlon=tlon,
                    rlat=tlat,
                    method="nearest",
                )
                .values
            )
            if test == 1.0:
                IN_SUBDOMAIN = True
                rvec.iloc[n][dom] = 1
                # if dom is NAL or SAL, then also put cookie in AL directory
                if dom == 'NAL' or dom == 'SAL':
                    rvec.iloc[n]['AL'] = 1
                # if dom is MDS or MDL, then also put cookie in MD directory
                if dom == 'MDS' or dom == 'MDL':
                    rvec.iloc[n]['MD'] = 1
                break
    return rvec

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
r_disk = 5

subdomains = xr.open_dataset('/home/mfeldmann/Research/data/mesocyclone_climate/domain/subdomains_lonlat.nc')


#%% lifecycle properties
climate = "future"
path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
rainpath = "/media/mfeldmann@giub.local/Elements/supercell_climate/CT2/"+climate+"_climate/"

LC_prop_fc = fmap.supercell_lifecycle(path, rainpath, twoMD=False, skipped_days=None)
LC_prop_fc.to_csv("/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate" + method + "a.csv")
#%%
climate = "current"
path = "/media/mfeldmann@giub.local/Elements/supercell_climate/SDT2_output/"+climate+"_climate/domain/XPT_1MD_zetath5_wth5"
rainpath = "/media/mfeldmann@giub.local/Elements/supercell_climate/CT2/"+climate+"_climate/"

LC_prop_cc = fmap.supercell_lifecycle(path, rainpath, twoMD=False, skipped_days=None)
LC_prop_cc.to_csv("/home/mfeldmann/Research/data/mesocyclone_climate/" + climate + "_climate" + method + "a.csv")

#%%
LC_prop_cc = pd.read_csv("/home/mfeldmann/Research/data/mesocyclone_climate/current_climate" + method + "a.csv")
LC_prop_cc = LC_prop_cc.drop('Unnamed: 0', axis=1)
LC_prop_fc = pd.read_csv("/home/mfeldmann/Research/data/mesocyclone_climate/future_climate" + method + "a.csv")
LC_prop_fc = LC_prop_fc.drop('Unnamed: 0', axis=1)
headers = LC_prop_cc.columns

LC_prop_cc.SC_duration = LC_prop_cc.SC_duration.astype('timedelta64[m]')
LC_prop_fc.SC_duration = LC_prop_fc.SC_duration.astype('timedelta64[m]')
LC_prop_cc['year'] = pd.DatetimeIndex(LC_prop_cc.dropna()['SC_date']).year
LC_prop_fc['year'] = pd.DatetimeIndex(LC_prop_fc.dropna()['SC_date']).year
LC_prop_cc.SC_date = pd.DatetimeIndex(LC_prop_cc.dropna()['SC_date']).dayofyear
LC_prop_fc.SC_date = pd.DatetimeIndex(LC_prop_fc.dropna()['SC_date']).dayofyear
LC_prop_cc.SC_rarea_av = LC_prop_cc.SC_rarea_av * (2.2**2)
LC_prop_fc.SC_rarea_av = LC_prop_fc.SC_rarea_av * (2.2**2)
LC_prop_cc.SC_r_max = LC_prop_cc.SC_r_max
LC_prop_fc.SC_r_max = LC_prop_fc.SC_r_max
LC_prop_cc.SC_r_av = LC_prop_cc.SC_r_av
LC_prop_fc.SC_r_av = LC_prop_fc.SC_r_av
LC_prop_cc.SC_rarea_sum = LC_prop_cc.SC_rarea_sum * (2.2**2)
LC_prop_fc.SC_rarea_sum = LC_prop_fc.SC_rarea_sum * (2.2**2)
#%%
cc='mediumblue'
fc='crimson'
headers = LC_prop_cc.columns
ranges_hi = [332,3000,1000,800,100,50,24,60,30,180,65,45,1.3,60,50,330,250,150000,2500,0.015,0.015,40,35,7]
ranges_lo = [100,0,0,30,10,10,0,30,-10,-180,0,0,-1.25,8,8,0,0,10000,100,0.005,0.005,5,5,1]
ranges_int = [15,100,50,30,5,2,1,1,1,10,2,2,0.5,1.5,1.5,15,10,10000,100,0.0005,0.0005,2,2,1]
fac=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1000,1000,1,1,1]
a=0
titles=['Day of year','Storm ID','a) track length [km]','b) track duration [min]',
        'total mesocyclone area [gridpoints]','average mesocyclone area [gridpoints]','Initiation time [h]',
        'Initiation latitude','Initiation longitude','Track direction [°]','e) maximum hail size [mm]','Average hail size [cm]',
        'Left vs right movers','f) maximum windgust [m s$^{-1}$]','average windgust [m s$^{-1}$]',
        'd) maximum 5min rain [mm h$^{-1}$]','d) average 5min rain [mm]','total storm area [km$^{2}$]','c) average storm area [km$^{2}$]',
        'h) maximum vorticity [10$^{-3}$ s$^{-1}$]','average vorticity [10$^{-3}$ s$^{-1}$]','g) maximum updraft [m s$^{-1}$]','average updraft [m s$^{-1}$]',
        'number of mesocyclone detections']


for header in headers[:]:
    r_hi = ranges_hi[a]
    r_lo = ranges_lo[a]
    r_int = ranges_int[a]
    FC = LC_prop_fc.dropna()[header]
    CC = LC_prop_cc.dropna()[header]
    
    # plt.hist(FC,alpha=0.6,bins=np.arange(r_lo,r_hi,r_int),label='future climate',color=fc,density=True)
    # plt.hist(CC,alpha=0.6,bins=np.arange(r_lo,r_hi,r_int),label='current climate',color=cc,density=True)
    
    p=sns.distplot(FC,bins=np.arange(r_lo,r_hi,r_int),color=fc)
    sns.distplot(CC,bins=np.arange(r_lo,r_hi,r_int),color=cc,ax=p)
    plt.xticks(np.arange(r_lo,r_hi,2*r_int),labels=(np.arange(r_lo,r_hi,2*r_int) *fac[a]).astype(int),rotation=90)
    
    plt.axvline(x = np.nanmedian(FC), color = fc)
    plt.axvline(x = np.nanmedian(CC), color = cc)
    
    plt.axvline(x = np.nanpercentile(FC,25), color = fc, linestyle='dashed')
    plt.axvline(x = np.nanpercentile(CC,25), color = cc, linestyle='dashed')
    
    plt.axvline(x = np.nanpercentile(FC,75), color = fc, linestyle='dashed')
    plt.axvline(x = np.nanpercentile(CC,75), color = cc, linestyle='dashed')
    
    #if a==2: p.legend()
    p.set(xlabel=None)
    p.set(xlim=[r_lo,r_hi])
    plt.title(titles[a])
    plt.tight_layout()
    plt.savefig(figpath + '2hist_' + header + '.png')
    plt.show()
    a+=1
    
#%%
import scipy
headers = LC_prop_cc.columns
a=0

for header in headers[:]:
    r_hi = ranges_hi[a]
    r_lo = ranges_lo[a]
    r_int = ranges_int[a]
    FC = LC_prop_fc[header].dropna()
    CC = LC_prop_cc[header].dropna()
    print(scipy.stats.mannwhitneyu(CC,FC)[1]<=0.05,header)
    # plt.hist(FC,alpha=0.6,bins=np.arange(r_lo,r_hi,r_int),label='future climate',color=c4,density=True)
    # plt.hist(CC,alpha=0.6,bins=np.arange(r_lo,r_hi,r_int),label='current climate',color=c1,density=True)
    
    # plt.xticks(np.arange(r_lo,r_hi,r_int),rotation=90)
    
    # plt.axvline(x = np.nanmedian(FC), color = c4)
    # plt.axvline(x = np.nanmedian(CC), color = c1)
    
    # plt.axvline(x = np.nanpercentile(FC,25), color = c4, linestyle='dashed')
    # plt.axvline(x = np.nanpercentile(CC,25), color = c1, linestyle='dashed')
    
    # plt.axvline(x = np.nanpercentile(FC,75), color = c4, linestyle='dashed')
    # plt.axvline(x = np.nanpercentile(CC,75), color = c1, linestyle='dashed')
    
    # plt.legend()
    # plt.title(titles[a])
    # plt.savefig(figpath + header + '_dens_hist.png')
    # plt.show()
    a+=1

#%%
LC_prop_cc = pd.read_csv("/home/mfeldmann/Research/data/mesocyclone_climate/current_climate" + method + "a.csv")
LC_prop_cc = LC_prop_cc.drop('Unnamed: 0', axis=1)
LC_prop_fc = pd.read_csv("/home/mfeldmann/Research/data/mesocyclone_climate/future_climate" + method + "a.csv")
LC_prop_fc = LC_prop_fc.drop('Unnamed: 0', axis=1)
headers = LC_prop_cc.columns

# LC_prop_cc.SC_duration = LC_prop_cc.SC_duration.astype('timedelta64[m]')
# LC_prop_fc.SC_duration = LC_prop_fc.SC_duration.astype('timedelta64[m]')
LC_prop_cc['year'] = pd.DatetimeIndex(LC_prop_cc.dropna()['SC_date']).year
LC_prop_fc['year'] = pd.DatetimeIndex(LC_prop_fc.dropna()['SC_date']).year
LC_prop_cc.SC_date = pd.DatetimeIndex(LC_prop_cc.dropna()['SC_date']).dayofyear
LC_prop_fc.SC_date = pd.DatetimeIndex(LC_prop_fc.dropna()['SC_date']).dayofyear

#%%
rvec_cc = assign_regions(LC_prop_cc.SC_init_lat,LC_prop_cc.SC_init_lon,subdomains)
rvec_fc = assign_regions(LC_prop_fc.SC_init_lat,LC_prop_fc.SC_init_lon,subdomains)

LC_prop_cc = pd.concat([LC_prop_cc,rvec_cc],axis=1)
LC_prop_fc = pd.concat([LC_prop_fc,rvec_fc],axis=1)


    #%%
cc='mediumblue'
fc='crimson'
headers = LC_prop_cc.columns
ranges_hi = [332,3000,1000,800,100,50,24,60,30,180,65,45,1.3,60,50,330,250,150000,2500,0.015,0.015,40,35,7]
ranges_lo = [100,0,0,30,10,10,0,30,-10,-180,0,0,-1.25,8,8,0,0,10000,100,0.005,0.005,5,5,1]
ranges_int = [15,100,50,30,5,2,1,1,1,10,2,2,0.5,1.5,1.5,15,10,10000,100,0.0005,0.0005,2,2,1]
fac=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1000,1000,1,1,1]
a=0
titles=['Day of year','Storm ID','track length [km]','track duration [min]',
        'total mesocyclone area [gridpoints]','average mesocyclone area [gridpoints]','Initiation time [h]',
        'Initiation latitude','Initiation longitude','Track direction [°]','maximum hail size [mm]','Average hail size [cm]',
        'Left vs right movers','maximum windgust [m s$^{-1}$]','average windgust [m s$^{-1}$]',
        'maximum 5min rain [mm h$^{-1}$]','average 5min rain [mm]','total storm area [km$^{2}$]','average storm area [km$^{2}$]',
        'maximum vorticity [10$^{-3}$ s$^{-1}$]','average vorticity [10$^{-3}$ s$^{-1}$]','maximum updraft [m s$^{-1}$]','average updraft [m s$^{-1}$]',
        'number of mesocyclone detections']


#%% PDF per region
a=0
import scipy
for header in headers[:]:
    #fig,axs=plt.subplots(3,5,figsize=(20,12))
    b=0
    for reg in list(subdomains.keys())[2:]:
        r_hi = ranges_hi[a]
        r_lo = ranges_lo[a]
        r_int = ranges_int[a]
        FC = (LC_prop_fc[header] * LC_prop_fc[reg]).dropna()
        CC = (LC_prop_cc[header] * LC_prop_cc[reg]).dropna()
        test = scipy.stats.mannwhitneyu(np.array(CC).astype(float),np.array(FC).astype(float))[1]
        title = titles[a]
        if test <= 0.05: title += ' *'
        
        plt.hist(FC,alpha=0.6,bins=np.arange(r_lo,r_hi,r_int),label='future climate',color=fc,density=True)
        plt.hist(CC,alpha=0.6,bins=np.arange(r_lo,r_hi,r_int),label='current climate',color=cc,density=True)
        
        plt.xticks(np.arange(r_lo,r_hi,2*r_int),labels=(np.arange(r_lo,r_hi,2*r_int) *fac[a]).astype(int),rotation=90)
        
        plt.axvline(x = np.nanmedian(FC), color = fc)
        plt.axvline(x = np.nanmedian(CC), color = cc)
        
        plt.axvline(x = np.nanpercentile(FC,25), color = fc, linestyle='dashed')
        plt.axvline(x = np.nanpercentile(CC,25), color = cc, linestyle='dashed')
        
        plt.axvline(x = np.nanpercentile(FC,75), color = fc, linestyle='dashed')
        plt.axvline(x = np.nanpercentile(CC,75), color = cc, linestyle='dashed')
        
        if a==2: plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figpath + reg + '_pdf_' + header + '.png')
        plt.show()
    a+=1
#%% combined PDF plot
import scipy
titles=['Day of year','Storm ID','track length [km]','track duration [min]',
        'total mesocyclone area [gridpoints]','average mesocyclone area [gridpoints]','Initiation time [h]',
        'Initiation latitude','Initiation longitude','Track direction [°]','maximum hail size [mm]','Average hail size [cm]',
        'Left vs right movers','maximum windgust [m s$^{-1}$]','average windgust [m s$^{-1}$]',
        'maximum 5min rain [mm h$^{-1}$]','average 5min rain [mm]','total storm area [km$^{2}$]','average storm area [km$^{2}$]',
        'maximum vorticity [10$^{-3}$ s$^{-1}$]','average vorticity [10$^{-3}$ s$^{-1}$]','maximum updraft [m s$^{-1}$]','average updraft [m s$^{-1}$]',
        'number of mesocyclone detections']

a=0
regions=np.array(list(subdomains.keys())[2:])
regions = regions[[1,2,3,4,5,8,10,11,12]]
for header in headers[:]:
    fig,axs=plt.subplots(3,3,figsize=(12,12))
    b=0
    for reg in regions:
        n1=int(b%3)
        n2=int(b/3)
        r_hi = ranges_hi[a]
        r_lo = ranges_lo[a]
        r_int = ranges_int[a]
        FC = (LC_prop_fc[header] * LC_prop_fc[reg]).dropna()
        CC = (LC_prop_cc[header] * LC_prop_cc[reg]).dropna()
        test = scipy.stats.mannwhitneyu(np.array(CC).astype(float),np.array(FC).astype(float))[1]
        title = reg
        if test <= 0.05: title += ' *'
        axs[n2,n1].hist(FC,alpha=0.6,bins=np.arange(r_lo,r_hi,r_int),label='future climate',color=fc,density=True)
        axs[n2,n1].hist(CC,alpha=0.6,bins=np.arange(r_lo,r_hi,r_int),label='current climate',color=cc,density=True)
        
        if a==12:
            axs[n2,n1].set_xticks(np.arange(r_lo,r_hi,2*r_int),(np.arange(r_lo,r_hi,2*r_int)).astype(int),rotation=90)
        
        else: 
            axs[n2,n1].set_xticks(np.arange(r_lo,r_hi,r_int*4),labels=(np.arange(r_lo,r_hi,4*r_int) *fac[a]).astype(int),rotation=90)
        
        axs[n2,n1].axvline(x = np.nanmedian(FC), color = fc)
        axs[n2,n1].axvline(x = np.nanmedian(CC), color = cc)
        
        axs[n2,n1].axvline(x = np.nanpercentile(FC,25), color = fc, linestyle='dashed')
        axs[n2,n1].axvline(x = np.nanpercentile(CC,25), color = cc, linestyle='dashed')
        
        axs[n2,n1].axvline(x = np.nanpercentile(FC,75), color = fc, linestyle='dashed')
        axs[n2,n1].axvline(x = np.nanpercentile(CC,75), color = cc, linestyle='dashed')
        
        if b==0: axs[n2,n1].legend(loc='upper right')#bbox_to_anchor=(0.9,1.5))
        axs[n2,n1].set_title(title)
        b+=1

    plt.suptitle(titles[a])
    plt.tight_layout()
    plt.savefig(figpath + 'comb' + header + '_dens_hist.png',dpi=300)
    plt.show()
    a+=1

#%% frequency shifts
import scipy
fig,ax=plt.subplots(figsize=(10,5))
c=0
FCS=[]
CCS=[]
for reg in list(subdomains.keys())[2:]:
    FCS_y=[]
    CCS_y=[]
    for yr in np.unique(LC_prop_fc.year):
        LC=LC_prop_fc[LC_prop_fc.year==yr]
        FCS_y.append(np.nansum(LC[reg]))
    for yr in np.unique(LC_prop_cc.year):
        LC=LC_prop_cc[LC_prop_cc.year==yr]
        CCS_y.append(np.nansum(LC[reg]))
    print(CCS_y,FCS_y)
    print(reg, scipy.stats.wilcoxon(CCS_y,FCS_y)[1]<=0.05, scipy.stats.wilcoxon(CCS_y,FCS_y))
        
    FC = np.nansum(LC_prop_fc[reg])/11
    CC = np.nansum(LC_prop_cc[reg])/11
    FCS.append(FC); CCS.append(CC)
    c+=1
    ax.bar(c-0.2,CC,width=0.3,color=c1,label='current climate')
    ax.bar(c+0.2,FC,width=0.3,color=c4,label='future climate')
    if c==1: ax.legend()
ax.set_xticks(np.arange(1,14),list(subdomains.keys())[2:])
plt.title('Number of supercells initiating per year in each region')
plt.savefig(figpath + 'comb_tracknumber.png',dpi=300)
plt.show()

#%%

FCS_y=[]
CCS_y=[]
for yr in np.unique(LC_prop_fc.year):
    LC=LC_prop_fc[LC_prop_fc.year==yr]
    FCS_y.append(np.nansum(LC))
for yr in np.unique(LC_prop_cc.year):
    LC=LC_prop_cc[LC_prop_cc.year==yr]
    CCS_y.append(np.nansum(LC))
print(CCS_y,FCS_y)
print( scipy.stats.wilcoxon(CCS_y,FCS_y)[1]<=0.05, scipy.stats.wilcoxon(CCS_y,FCS_y))
    
FC = np.nansum(LC_prop_fc)/11
CC = np.nansum(LC_prop_cc)/11



#%% mover shifts
regions=np.array(list(subdomains.keys())[2:])
regions = regions[[1,2,3,4,5,8,10,11,12]]
import scipy
fig,ax=plt.subplots(figsize=(10,5))
c=0
FCS=[]
CCS=[]
labelss = []
for reg in regions:
    FCS_y=[]
    CCS_y=[]
    for yr in np.unique(LC_prop_fc.year):
        LC=LC_prop_fc[LC_prop_fc.year==yr]
        RM = np.nansum(LC[LC.SC_mover>0][reg])
        LM = np.nansum(LC[LC.SC_mover<0][reg])
        if RM+LM==0: perc=0
        else: perc = RM/(RM+LM)
        FCS_y.append(perc)
    for yr in np.unique(LC_prop_cc.year):
        LC=LC_prop_cc[LC_prop_cc.year==yr]
        RM = np.nansum(LC[LC.SC_mover>0][reg])
        LM = np.nansum(LC[LC.SC_mover<0][reg])
        if RM+LM==0: perc=0
        else: perc = RM/(RM+LM)
        CCS_y.append(perc)
    print(CCS_y,FCS_y)
    print(reg, scipy.stats.wilcoxon(CCS_y,FCS_y)[1]<=0.05, scipy.stats.wilcoxon(CCS_y,FCS_y))
    test = scipy.stats.wilcoxon(CCS_y,FCS_y)[1]<=0.05
    FC_RM = np.nansum(LC_prop_fc[LC_prop_fc.SC_mover>0][reg])
    FC_LM = np.nansum(LC_prop_fc[LC_prop_fc.SC_mover<0][reg])
    CC_RM = np.nansum(LC_prop_cc[LC_prop_cc.SC_mover>0][reg])
    CC_LM = np.nansum(LC_prop_cc[LC_prop_cc.SC_mover<0][reg])
    perc_fc = FC_RM / (FC_RM+FC_LM) * 100
    perc_cc = CC_RM / (CC_RM+CC_LM) * 100
    FCS.append(perc_fc); CCS.append(perc_cc)
    label = reg
    if test: label+=' *'
    labelss.append(label)
    c+=1
    ax.bar(c-0.2,perc_cc,width=0.3,color=cc,label='current climate')
    ax.bar(c+0.2,perc_fc,width=0.3,color=fc,label='future climate')
    if c==1: ax.legend(loc='lower left')
ax.set_xticks(np.arange(1,10),labelss)
plt.title('Percentage of right movers in each region')
plt.savefig(figpath + 'comb_movers.png',dpi=300)
plt.show()

#%%
FCS_y=[]
CCS_y=[]
for yr in np.unique(LC_prop_fc.year):
    LC=LC_prop_fc[LC_prop_fc.year==yr]
    RM = np.nansum(LC[LC.SC_mover>0])
    LM = np.nansum(LC[LC.SC_mover<0])
    if RM+LM==0: perc=0
    else: perc = RM/(RM+LM)
    FCS_y.append(perc)
for yr in np.unique(LC_prop_cc.year):
    LC=LC_prop_cc[LC_prop_cc.year==yr]
    RM = np.nansum(LC[LC.SC_mover>0])
    LM = np.nansum(LC[LC.SC_mover<0])
    if RM+LM==0: perc=0
    else: perc = RM/(RM+LM)
    CCS_y.append(perc)
print(CCS_y,FCS_y)
print(scipy.stats.wilcoxon(CCS_y,FCS_y)[1]<=0.05, scipy.stats.wilcoxon(CCS_y,FCS_y))
test = scipy.stats.wilcoxon(CCS_y,FCS_y)[1]<=0.05
FC_RM = np.nansum(LC_prop_fc[LC_prop_fc.SC_mover>0])
FC_LM = np.nansum(LC_prop_fc[LC_prop_fc.SC_mover<0])
CC_RM = np.nansum(LC_prop_cc[LC_prop_cc.SC_mover>0])
CC_LM = np.nansum(LC_prop_cc[LC_prop_cc.SC_mover<0])
perc_fc = FC_RM / (FC_RM+FC_LM) * 100
perc_cc = CC_RM / (CC_RM+CC_LM) * 100

#%%
def convert_to_doy(year,month,day):
    daymonth=[31,28,31,30,31,30,31,31,30,31,30,31]
    if year%4==0:
        daymonth=[31,29,31,30,31,30,31,31,30,31,30,31]
    doy = np.nansum(daymonth[:month-1])+day
    return int(doy)
#%%
import seaborn as sns
cc='mediumblue'
fc='crimson'
subset_mod = LC_prop_cc[np.nansum([LC_prop_cc.NAL,LC_prop_cc.SAL],axis=0).astype(bool)]
#subset_mod = subset_mod[subset_mod.SC_init_lon<11]

usecols = ['ID', 'time', 'mesostorm', 'mesohailstorm', 'lon', 'lat','pos','neg']
fullset = pd.read_csv("/media/mfeldmann@giub.local/Elements/supercell_climate/observations/Full_dataset_thunderstorm_types.csv", sep=';', usecols=usecols)
subset = fullset[fullset.mesostorm==1]
subset = subset.drop_duplicates(subset='ID',keep='first')
subset['year']=np.floor(subset.time/100000000).astype(int)
subset['date']=np.round((subset.time.values - subset.year.values*100000000)/10000).astype(int)
subset['month']=np.floor(subset.date/100).astype(int)
subset['day']=np.round(subset.date%100).astype(int)
subset['daytime']=(subset.time%10000).astype(int)
subset['doy']=np.zeros(len(subset))

for i in range(len(subset)):
    subset['doy'].iloc[i] = convert_to_doy(subset['year'].iloc[i],subset['month'].iloc[i],subset['day'].iloc[i])

subset.daytime[subset.daytime<500]+=2400
subset_mod.SC_init_h[subset_mod.SC_init_h<5]+=24
subset_mod_2 = subset_mod[subset_mod.SC_init_h!=28]

p1=sns.distplot(subset.doy,bins=[91,106,121,136,152,167,182,197,213,228,244,259,274,289,305],color=cc)
sns.distplot(subset_mod_2.SC_date,bins=[91,106,121,136,152,167,182,197,213,228,244,259,274,289,305],color=fc,ax=p1)
plt.xticks([91,121,152,182,213,244,274,305],['APR-1','MAY-1','JUN-1','JUL-1','AUG-1','SEP-1','OCT-1','NOV-1'],rotation=90)
plt.title('Seasonal cycle')
p1.set(xlabel=None)
plt.tight_layout()
plt.savefig(figpath + 'verif_season.png',dpi=300)
plt.show()



p2=sns.distplot(subset.daytime/100,bins=np.arange(4,30,2),color=cc)
sns.distplot(subset_mod_2.SC_init_h,bins=np.arange(4,30,2),color=fc,ax=p2)
plt.xticks(np.arange(5,29,1),['05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','00','01','02','03','04'],rotation=90)
plt.xticks(np.arange(6,29,2),['06:00','08:00','10:00','12:00','14:00','16:00','18:00','20:00','22:00','00:00','02:00','04:00'],rotation=90)
#plt.xticks([5,8,11,14,17,20,23,26],['05:00','08:00','11:00','14:00','17:00','20:00','23:00','02:00'])
plt.title('Diurnal cycle')
p2.set(xlabel=None)
p2.set(xlim=[5,28])
#plt.xlabel('Time of day, UTC [h]')
#plt.legend()
plt.tight_layout()
plt.savefig(figpath + 'verif_diurnal.png',dpi=300)
plt.show()



plt.hist(subset.daytime/100,bins=np.arange(4,30,2),color=cc,label='radar')
plt.hist(subset_mod_2.SC_init_h,bins=np.arange(4,30,2),color=fc,label='model')
plt.xticks(np.arange(5,29,1),['05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','00','01','02','03','04'],rotation=90)
plt.xticks(np.arange(6,29,2),['06:00','08:00','10:00','12:00','14:00','16:00','18:00','20:00','22:00','00:00','02:00','04:00'],rotation=90)
#plt.xticks([5,8,11,14,17,20,23,26],['05:00','08:00','11:00','14:00','17:00','20:00','23:00','02:00'])
plt.title('Diurnal cycle')
plt.legend()
plt.tight_layout()
plt.savefig(figpath + 'legend.png',dpi=300)
plt.show()

sns.distplot(subset.doy,bins=[91,106,121,136,152,167,182,197,213,228,244,259,274,289,305],color=cc)
