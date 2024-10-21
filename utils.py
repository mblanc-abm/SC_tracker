# this script aims at plotting the one-season distribution map of modelled supercells
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
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib
matplotlib.rcParams.update({'font.size': 24})
from matplotlib import colors
#from sklearn.neighbors import KernelDensity

#==================================================================================================================================================
# FUNCTIONS
#==================================================================================================================================================
def plot_dem(lons, lats, surf, subdomains, save=True, addname="", figpath=""):
    """
    Plots the desired decadal (default) or seasonal frequency map and saves the figure
    for the decadal maps expressed in anual average, directly provide the annually averaged counts in input
    
    Parameters
    ----------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    fmap : 2D array
        the frequency / distribution map to be plotted
        filtering and convolution/dilation processes are assumed to be beforehand handled, ie fmap ready to be plotted
    save : bool
        option to save figure
    addname: str
        option to add additional information to figure name and title

    Returns
    -------
    Plots the desired frequency map and saves it if requested.
    """
    
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
    
    
    
 

    zl, zr, zb, zt = 130, 25, 195, 25 #smart cut for entire domain
    cbar_orientation = "vertical"
    cmap = colormaps["terrain"]; 
    
    new_cmap = colors.LinearSegmentedColormap.from_list(
            'cropped',
            cmap(np.linspace(0.2, 1, 256)))
    cmap = new_cmap
    cmap.set_under('lightgrey')
    # surf[surf<0]=np.nan; cmap.set_bad('lightsteelblue')

    #cont = ax.contourf(lons[zl:-zr], lats[zb:-zt], surf[zb:-zt,zl:-zr],vmin=0,vmax=3500,levels=[0,200,400,600,800,1000,1500,2000,2500,3000,3500], cmap=cmap, transform=rp,extend='both')
    cont = ax.contourf(lons, lats, surf,vmin=0,vmax=3500,levels=[0,200,400,600,800,1000,1500,2000,2500,3000,3500], cmap=cmap, transform=rp,extend='both')
    
    
    #ax.plot([-3.88815558,-3.88815558,0.99421144,0.99421144,-3.88815558],[-1.86351067, 1.50951803, 1.50951803,-1.86351067,-1.86351067], color='r', linestyle='--')
    ax.plot(lons[[zl,zl,-zr,-zr,zl]],lats[[zb,-zt,-zt,zb,zb]], color='k', linestyle='--')

    domains = ['BI','EE','FR','IP','MD','CE','BA','NAL','SAL']
    for key in domains:
        im = subdomains[key].isel(time=0).values
        im[np.isnan(im)]=0
        ax.contour(lons,lats,im,levels=[0.9],colors='w',linestyles='-',linewidths=2)

    # cont = ax.contourf(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], surf[zb:-zt,zl:-zr],vmin=0,vmax=3500,levels=[0,200,400,600,800,1000,1500,2000,2500,3000,3500], cmap='terrain', transform=rp)
    
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=0.5, color='gray', alpha=1, linestyle='--')
    gl.top_labels = True
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    cbar = plt.colorbar(cont, orientation=cbar_orientation, label='elevation [m]')
    # cbar.set_ticks(bounds)
    # cbar.set_ticklabels(bounds_str)
        
    plt.title('a) Model domain and elevation')
    plt.tight_layout()
    if save:
        fig.savefig(figpath+'dem.png', dpi=300)
    plt.show()
    plt.close()
    
    return




def plot_fmap_simplified(lons, lats, fmap, title, lab, bounds, bounds_str, cmap, figname, hatch=False, hatching=None, delta=False, cycle=False, boundaries=np.zeros(1), CH=False, cbar='vertical'):
    """
    Plots the desired decadal (default) or seasonal frequency map and saves the figure
    for the decadal maps expressed in anual average, directly provide the annually averaged counts in input
    
    Parameters
    ----------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    fmap : 2D array
        the frequency / distribution map to be plotted
        filtering and convolution/dilation processes are assumed to be beforehand handled, ie fmap ready to be plotted
    typ: str
        type of the frequency map, eg "supercell", "rain", "mesocyclone"
    r_disk : int
        radius of the disk footprint in grid point
    climate : str
        "future" or "current"
    season : bool
        True for a seasonal frequency map, False for the decadal period. The default is False.
    diurnal : bool
        True for a diurnal frequency map, False for the decadal period. The default is False.
    year : str or None
        if seasonal, year YYYY of the season. if diurnal, hh for the hour. Elsewise, None. The default is None.
    zoom : bool, optional
        False: smart cut of whole domain (ocean and Northern Africa discarded) ; True: zoom on the Alps.
    save : bool
        option to save figure
    addname: str
        option to add additional information to figure name and title

    Returns
    -------
    Plots the desired frequency map and saves it if requested.
    """
    
    
    ncolors=256
    if cmap=='twilight_shifted': 
        cmap = colormaps[cmap]
        #ncolors *= 2
        new_cmap = colors.LinearSegmentedColormap.from_list(
                'cropped',
                cmap(np.linspace(0.1, 0.9, 512)))
        cmap = new_cmap
        cmap.set_bad('grey')
    else: 
        cmap = colormaps[cmap]
        cmap.set_bad('grey')
    if delta:
        extend='both'
    elif cycle:
        extend='neither'
        
    else:
        extend='max'
    if not boundaries.any():
        boundaries=bounds
    
    cmap
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
    cbar_orientation = cbar

    cont = ax.pcolormesh(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], fmap[zb:-zt,zl:-zr], 
                         norm=norm, cmap=cmap, transform=ccrs.PlateCarree())
    if hatch:
        # ax.add_patch(hatching,hatch='x',edgecolor='k',transform=ccrs.PlateCarree())
        ax.contourf(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], hatching[zb:-zt,zl:-zr],
                    levels=[-0.1,0.9],colors='grey',alpha=0.3,hatches='..', 
                    transform=ccrs.PlateCarree())
    if CH:
        ax.set_extent([-3.88815558,  0.99421144, -1.86351067,  1.50951803], crs=rp)
        rlats = [47.284333348,46.425102252,46.040754913,46.370650060,46.834974607]
        rlons = [8.511999921,6.099415777,8.833216354,7.486550845,9.794466983]
        r_ix = []
        r_iy = []
        r_disk=45
        fp_ind = np.argwhere(disk(r_disk))
        rad_filt = np.ones(fmap.shape)
        for ii in range(len(rlats)):
            dist = abs(lats-rlats[ii]) + abs(lons-rlons[ii])
            x,y = np.where(dist == np.nanmin(dist))
            r_ix.append(x); r_iy.append(y)
            rad_filt[x-r_disk+fp_ind[:,0], y-r_disk+fp_ind[:,1]] = 0
        filt = ax.contourf(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], rad_filt[zb:-zt,zl:-zr], 
                             colors='k', levels=[0.5,1.5,2], transform=ccrs.PlateCarree(),alpha=0.5)
        filtc = ax.contour(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], rad_filt[zb:-zt,zl:-zr], 
                             colors='k', levels=[0.5,1.5,2], transform=ccrs.PlateCarree())
    
    
    cbar = plt.colorbar(cont, orientation=cbar_orientation, label=lab, boundaries=bounds)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels(bounds_str)
        
    plt.title(title)
    plt.tight_layout()
    

    fig.savefig(figname, dpi=300)#, format="svg")
    plt.show()
    plt.close()
    
    return


def plot_fmap_CH(lons, lats, fmap, title, lab, bounds, bounds_str, cmap, figname, hatch=False, hatching=None, delta=False, cycle=False, boundaries=np.zeros(1), CH=False, cbar='vertical'):
    """
    Plots the desired decadal (default) or seasonal frequency map and saves the figure
    for the decadal maps expressed in anual average, directly provide the annually averaged counts in input
    
    Parameters
    ----------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    fmap : 2D array
        the frequency / distribution map to be plotted
        filtering and convolution/dilation processes are assumed to be beforehand handled, ie fmap ready to be plotted
    typ: str
        type of the frequency map, eg "supercell", "rain", "mesocyclone"
    r_disk : int
        radius of the disk footprint in grid point
    climate : str
        "future" or "current"
    season : bool
        True for a seasonal frequency map, False for the decadal period. The default is False.
    diurnal : bool
        True for a diurnal frequency map, False for the decadal period. The default is False.
    year : str or None
        if seasonal, year YYYY of the season. if diurnal, hh for the hour. Elsewise, None. The default is None.
    zoom : bool, optional
        False: smart cut of whole domain (ocean and Northern Africa discarded) ; True: zoom on the Alps.
    save : bool
        option to save figure
    addname: str
        option to add additional information to figure name and title

    Returns
    -------
    Plots the desired frequency map and saves it if requested.
    """
    
    
    ncolors=256
    if cmap=='twilight_shifted': 
        cmap = colormaps[cmap]
        #ncolors *= 2
        new_cmap = colors.LinearSegmentedColormap.from_list(
                'cropped',
                cmap(np.linspace(0.1, 0.9, 512)))
        cmap = new_cmap
        cmap.set_bad('grey')
    else: 
        cmap = colormaps[cmap]
        cmap.set_bad('grey')
    if delta:
        extend='both'
    elif cycle:
        extend='neither'
        
    else:
        extend='max'
    if not boundaries.any():
        boundaries=bounds
    
    cmap
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
    cbar_orientation = cbar

    cont = ax.pcolormesh(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], fmap[zb:-zt,zl:-zr], 
                         norm=norm, cmap=cmap, transform=ccrs.PlateCarree())
    if hatch:
        # ax.add_patch(hatching,hatch='x',edgecolor='k',transform=ccrs.PlateCarree())
        ax.contourf(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], hatching[zb:-zt,zl:-zr],
                    levels=[-0.1,0.9],colors='grey',alpha=0.3,hatches='..', 
                    transform=ccrs.PlateCarree())
    if CH:
        ax.set_extent([-3.88815558,  0.99421144, -1.86351067,  1.50951803], crs=rp)
    
    
    cbar = plt.colorbar(cont, orientation=cbar_orientation, label=lab, boundaries=bounds)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels(bounds_str)
        
    plt.title(title)
    plt.tight_layout()
    

    fig.savefig(figname, dpi=300)#, format="svg")
    plt.show()
    plt.close()
    
    return


def resolve_overlaps(rain_masks):
    """
    resolves overlaps for masks method: for every grid point, discards the additional counts of the rain cells overlapping several times
    with themselves, so that a single, overlapping raintracks only counts as 1

    Parameters
    ----------
    rain_masks : 3D array
        labeled rain cells (nan is background, cell labels start at 0), 2D arrays concatenated with a 5 min resolution, time is the first dimension

    Returns
    -------
    counts : 2D array
        number of counts at every grid point for the considered day
    """
    
    counts = np.zeros_like(rain_masks[0], dtype=int)
    for i in range(rain_masks.shape[1]):
        for j in range(rain_masks.shape[2]):
            ids = np.unique(rain_masks[:,i,j]) #consider the time column
            ids = ids[~np.isnan(ids)].astype(int) #remove nans and convert from float to int type
            counts[i,j] = len(ids)
    
    return counts


def yearly_supercell_tracks_model_fmap(start_day, end_day, path, r_disk, twoMD=False, skipped_days=None):
    """
    Computes the seasonal supercell frequency map using tracks method over the model whole domain
    Takes care of overlaps by default
    
    Parameters
    ----------
    start_day : str
        first day of the considered period, YYYYmmdd
    end_day : str
        last day of the considered period, YYYYmmdd
    path : str
        path to the SDT output files
    r_disk : int
        radius of the disk footprint in grid point
    twoMD : bool
        if True, filters out the supercells exhibiting less than 2 mesocyclone detections at different time steps
    skipped_days : list of str
        list of missing days which consequently must be skipped

    Returns
    -------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    counts : 2D array
        the desired seasonal supercell frequency map
    """
    
    start_day = pd.to_datetime(start_day)
    end_day = pd.to_datetime(end_day)
    daylist = pd.date_range(start_day, end_day)
    
    # load lons and lats static fields
    with xr.open_dataset(path + "/meso_masks_" + daylist[0].strftime("%Y%m%d") + ".nc") as dset:
        lons = dset["lon"].values
        lats = dset["lat"].values
    
    counts = np.zeros_like(lons, dtype=int) # initialise the counts matrix
    fp_ind = np.argwhere(disk(r_disk)) # footprint
    
    # remove skipped days from daylist
    if skipped_days:
        skipped_days = pd.to_datetime(skipped_days, format="%Y%m%d")
        daylist = [day for day in daylist if day not in skipped_days]
    
    SC_files = []
    for day in daylist:
        SC_files.append(path + "/supercell_" + day.strftime("%Y%m%d") + ".json")
    
    # loop over the days
    for SC_file in SC_files:
        # initialise an intermediate counts matrix for this day, which will contain its supercells tracks footprint; this incidentally avoids overlaps
        counts_day = np.zeros_like(lons, dtype=int) 
        
        # load the supercells of the given day
        with open(SC_file, "r") as read_file:
            SC_info = json.load(read_file)['supercell_data']
        
        # loop over the supercells of the day
        for SC in SC_info:
            
            # if 2 meso detections are required, filter out one-meso detection supercells
            if twoMD:
                if len(np.unique(SC['meso_datelist'])) < 2:
                    continue
            
            # initialise an intermediate counts matrix for this SC, which will contain its track footprint; this incidentally avoids overlaps
            counts_SC = np.zeros_like(lons, dtype=int)
            
            SC_lons = SC['cell_lon']
            SC_lats = SC['cell_lat']
            
            for j in range(len(SC_lons)): # for each SC centre of mass location
                distances = (lons - SC_lons[j])**2 + (lats - SC_lats[j])**2 # interpolate it from the lon-lat coords to the grid indices
                k,l = np.unravel_index(np.argmin(distances), distances.shape) #indices on grid corresponding to the SC centre of mass interpolation
                
                try:
                    counts_SC[k-r_disk+fp_ind[:,0], l-r_disk+fp_ind[:,1]] = 1
                except IndexError:
                    # Handle the case where the index is out of bounds -> simply skip the footprint printing
                    print('Index is out of bounds for the matrix and therefore skipped')
            
            counts_day += counts_SC
            
        counts += counts_day
        
    return lons, lats, counts

def seasonal_supercell_tracks_model_fmap(month, path, r_disk, twoMD=False):
    """
    Computes the seasonal supercell frequency map using tracks method over the model whole domain
    Takes care of overlaps by default
    
    Parameters
    ----------
    path : str
        path to the SDT output files
    r_disk : int
        radius of the disk footprint in grid point
    twoMD : bool
        if True, filters out the supercells exhibiting less than 2 mesocyclone detections at different time steps

    Returns
    -------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    counts : 2D array
        the desired seasonal supercell frequency map
    """
    
    filelist = sorted(glob.glob(path + "/supercell_????" + str(month).zfill(2) + "??.json"))
    filelist2 = sorted(glob.glob(path + "/meso_masks_????" + str(month).zfill(2) + "??.nc"))
    print(filelist, filelist2)

    
    # load lons and lats static fields
    with xr.open_dataset(filelist2[0]) as dset:
        lons = dset["lon"].values
        lats = dset["lat"].values
    
    counts = np.zeros_like(lons, dtype=int) # initialise the counts matrix
    fp_ind = np.argwhere(disk(r_disk)) # footprint
    
    # loop over the days
    for SC_file in filelist:
        print (SC_file)
        # initialise an intermediate counts matrix for this day, which will contain its supercells tracks footprint; this incidentally avoids overlaps
        counts_day = np.zeros_like(lons, dtype=int) 
        
        # load the supercells of the given day
        with open(SC_file, "r") as read_file:
            SC_info = json.load(read_file)['supercell_data']
        if len(SC_info)==0: continue
        
        # loop over the supercells of the day
        for SC in SC_info:
            
            ## if 2 meso detections are required, filter out one-meso detection supercells
            # if twoMD:
            #     if len(np.unique(SC['meso_datelist'])) < 2:
            #         continue
            
            # initialise an intermediate counts matrix for this SC, which will contain its track footprint; this incidentally avoids overlaps
            counts_SC = np.zeros_like(lons, dtype=int)
            
            
            SC_lons = np.array(SC['cell_lon'])
            SC_lats = np.array(SC['cell_lat'])
            
            for j in range(len(SC_lons)): # for each SC centre of mass location
                distances = (lons - SC_lons[j])**2 + (lats - SC_lats[j])**2 # interpolate it from the lon-lat coords to the grid indices
                k,l = np.unravel_index(np.argmin(distances), distances.shape) #indices on grid corresponding to the SC centre of mass interpolation
                
                try:
                    counts_SC[k-r_disk+fp_ind[:,0], l-r_disk+fp_ind[:,1]] = 1
                except IndexError:
                    # Handle the case where the index is out of bounds -> simply skip the footprint printing
                    print('Index is out of bounds for the matrix and therefore skipped')
            
            counts_day += counts_SC
            
        counts += counts_day
        

        
    return lons, lats, counts


def seasonal_supercell_tracks_obs_fmap(start_day, end_day, path1, path2, r_disk, conv=True):
    """
    Computes the seasonal supercell frequency map using tracks method over the observational domain (swiss radar network)
    The season is delineated by the starting and ending days
    Takes care of overlaps by default
    
    Parameters
    ----------
    start_day : str
        first day of the season, YYYYmmdd
    end_day : str
        last day of the season, YYYYmmdd
    conv : bool
        False: number of storms per grid box, ie per 4.84 km^2; True: 4-connectivity sum convolution yields the number of storms per 24.2 km^2

    Returns
    -------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    counts : 2D array
        the desired seasonal supercell frequency map
    """
    
    start_day = pd.to_datetime(start_day, format="%Y%m%d")
    end_day = pd.to_datetime(end_day, format="%Y%m%d")
    daylist = pd.date_range(start_day, end_day) # comprehensive days spanning the season
    daylist_str = [day.strftime("%Y%m%d") for day in daylist] # comprehensive days spanning the season
    season = start_day.strftime("%Y")
    
    # load lons and lats static fields
    with xr.open_dataset(path1 + "/meso_masks_" + daylist[0].strftime("%Y%m%d") + ".nc") as dset:
        lons = dset["lon"].values
        lats = dset["lat"].values
    
    counts = np.zeros_like(lons, dtype=int) # initialise the counts matrix
    fp_ind = np.argwhere(disk(r_disk)) # footprint: 4-connectivity; will serve for marking the supercell core footprint
    
    if season == "2022": # season 2022, special meso dataset
        
        # open meso dataset
        usecols = ['ID','time','lon','lat']
        mesoset = pd.read_csv(path2+"observations/TRTc_mesostorm_2022.csv", sep=';', usecols=usecols)
        mesoset['time'] = pd.to_datetime(mesoset['time'], format="%Y%m%d%H%M")
        mesoset['ID'] = [int(round(x)) for x in mesoset['ID']]
        
        # restrict to the considered period
        days_str = [dt.strftime("%Y%m%d") for dt in mesoset['time']]
        selection = mesoset[np.isin(days_str, daylist_str)]
        SC_ids = np.unique(selection['ID']) # extract the IDs of all the supercells of the considered period
        
        # loop over the supercells of the season
        for SC_ID in SC_ids:
            counts_SC = np.zeros_like(lons, dtype=int) # initialise an intermediate counts matrix for this SC, which will contain its track footprint; this incidentally avoids overlaps
            SC_lons = selection['lon'][selection['ID'] == SC_ID] # get the lon-lat coordinates of the SC track
            SC_lons = np.reshape(SC_lons, len(SC_lons))
            SC_lats = selection['lat'][selection['ID'] == SC_ID]
            SC_lats = np.reshape(SC_lats, len(SC_lats))
            if len(SC_lons)<12: continue
            for j in range(len(SC_lons)): # for each SC centroid location
                distances = (lons - SC_lons[j])**2 + (lats - SC_lats[j])**2 # interpolate it from the lon-lat coords to the grid indices
                k,l = np.unravel_index(np.argmin(distances), distances.shape) #indices on grid corresponding to the SC centre of mass interpolation
                
                try:
                    counts_SC[k-r_disk+fp_ind[:,0], l-r_disk+fp_ind[:,1]] = 1
                except IndexError:
                    # Handle the case where the index is out of bounds -> simply skip the footprint printing
                    print('Index is out of bounds for the matrix and therefore skipped')
            
            counts += counts_SC
            
    else: # season within the 2016-2021 dataset
        
        # open full observational dataset
        usecols = ['ID','time','mesostorm','lon','lat']
        fullset = pd.read_csv(path2+"/observations/Full_dataset_thunderstorm_types.csv", sep=';', usecols=usecols)
        fullset['time'] = pd.to_datetime(fullset['time'], format="%Y%m%d%H%M")
        days_str = [dt.strftime("%Y%m%d") for dt in fullset['time']]

        # restrict to the given season and select the supercells
        selection = fullset[np.isin(days_str, daylist_str)]
        selection = selection[selection['mesostorm']==1]
        selection['ID'] = [int(round(x)) for x in selection['ID']]
        SC_ids = np.unique(selection['ID']) # extract the IDs of all the supercells of the considered period
        
        # loop over the supercells of the season
        for SC_ID in SC_ids:
            print(SC_ID)
            counts_SC = np.zeros_like(lons, dtype=int) # initialise an intermediate counts matrix for this SC, which will contain its track footprint; this incidentally avoids overlaps
            SC_lons = selection['lon'][selection['ID'] == SC_ID] # get the lon-lat coordinates of the SC track
            SC_lons = np.array(np.reshape(SC_lons, len(SC_lons)))
            SC_lats = selection['lat'][selection['ID'] == SC_ID]
            SC_lats = np.array(np.reshape(SC_lats, len(SC_lats)))
            print(SC_lons[0],SC_lats[0])
            if len(SC_lons)<12: continue
            for j in range(len(SC_lons)): # for each SC centroid location
                distances = (lons - SC_lons[j])**2 + (lats - SC_lats[j])**2 # interpolate it from the lon-lat coords to the grid indices
                k,l = np.unravel_index(np.argmin(distances), distances.shape) #indices on grid corresponding to the SC centre of mass interpolation
                
                try:
                    counts_SC[k-r_disk+fp_ind[:,0], l-r_disk+fp_ind[:,1]] = 1
                except IndexError:
                    # Handle the case where the index is out of bounds -> simply skip the footprint printing
                    print('Index is out of bounds for the matrix and therefore skipped')
            
            counts += counts_SC
            
    return lons, lats, counts


def mover_supercell_tracks_model_fmap(path, r_disk, twoMD=False):
    """
    Computes the seasonal supercell frequency map using tracks method over the model whole domain
    Takes care of overlaps by default
    
    Parameters
    ----------
    path : str
        path to the SDT output files
    r_disk : int
        radius of the disk footprint in grid point
    twoMD : bool
        if True, filters out the supercells exhibiting less than 2 mesocyclone detections at different time steps

    Returns
    -------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    counts : 2D array
        the desired seasonal supercell frequency map
    """
    
    filelist = sorted(glob.glob(path + "/supercell_*.json"))
    filelist2 = sorted(glob.glob(path + "/meso_masks_*.nc"))
    
    # load lons and lats static fields
    with xr.open_dataset(filelist2[0]) as dset:
        lons = dset["lon"].values
        lats = dset["lat"].values
    
    # load lons and lats static fields
    with xr.open_dataset(filelist2[0]) as dset:
        lons = dset["lon"].values
        lats = dset["lat"].values
    
    counts_r = np.zeros_like(lons, dtype=int) # initialise the counts matrix
    counts_l = np.zeros_like(lons, dtype=int)
    fp_ind = np.argwhere(disk(r_disk)) # footprint
    
    # loop over the days
    for SC_file in filelist:
        print (SC_file)
        # initialise an intermediate counts matrix for this day, which will contain its supercells tracks footprint; this incidentally avoids overlaps
        counts_day_r = np.zeros_like(lons, dtype=int) 
        counts_day_l = np.zeros_like(lons, dtype=int) 
        
        # load the supercells of the given day
        with open(SC_file, "r") as read_file:
            SC_info = json.load(read_file)['supercell_data']
        if len(SC_info)==0: continue
        
        # loop over the supercells of the day
        for SC in SC_info:
            
            sigs = np.array(SC['signature'])
            sig = np.nanmean(sigs)
            
            ## if 2 meso detections are required, filter out one-meso detection supercells
            # if twoMD:
            #     if len(np.unique(SC['meso_datelist'])) < 2:
            #         continue
            
            # initialise an intermediate counts matrix for this SC, which will contain its track footprint; this incidentally avoids overlaps
            counts_SC_r = np.zeros_like(lons, dtype=int)
            counts_SC_l = np.zeros_like(lons, dtype=int)
            
            
            SC_lons = np.array(SC['cell_lon'])
            SC_lats = np.array(SC['cell_lat'])
            
            for j in range(len(SC_lons)): # for each SC centre of mass location
                distances = (lons - SC_lons[j])**2 + (lats - SC_lats[j])**2 # interpolate it from the lon-lat coords to the grid indices
                k,l = np.unravel_index(np.argmin(distances), distances.shape) #indices on grid corresponding to the SC centre of mass interpolation
                if sig>=0.2:
                    try:
                        counts_SC_r[k-r_disk+fp_ind[:,0], l-r_disk+fp_ind[:,1]] = 1
                    except IndexError:
                        # Handle the case where the index is out of bounds -> simply skip the footprint printing
                        print('Index is out of bounds for the matrix and therefore skipped')
                elif sig <=-0.2:
                    try:
                        counts_SC_l[k-r_disk+fp_ind[:,0], l-r_disk+fp_ind[:,1]] = 1
                    except IndexError:
                        # Handle the case where the index is out of bounds -> simply skip the footprint printing
                        print('Index is out of bounds for the matrix and therefore skipped')
            
            counts_day_r += counts_SC_r
            counts_day_l += counts_SC_l
            
        counts_r += counts_day_r
        counts_l += counts_day_l
        

        
    return lons, lats, counts_r, counts_l

def diurnal_supercell_tracks_model_fmap(start_hr, end_hr, path, r_disk, twoMD=False):
    """
    Computes the seasonal supercell frequency map using tracks method over the model whole domain
    Takes care of overlaps by default
    
    Parameters
    ----------
    start_hr : str
        first hour of the considered period, hh
    end_hr : str
        last hour of the considered period, hh
    path : str
        path to the SDT output files
    r_disk : int
        radius of the disk footprint in grid point
    twoMD : bool
        if True, filters out the supercells exhibiting less than 2 mesocyclone detections at different time steps

    Returns
    -------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    counts : 2D array
        the desired seasonal supercell frequency map
    """
    filelist = sorted(glob.glob(path + "/supercell_*.json"))
    filelist2 = sorted(glob.glob(path + "/meso_masks_*.nc"))
    
    # load lons and lats static fields
    with xr.open_dataset(filelist2[0]) as dset:
        lons = dset["lon"].values
        lats = dset["lat"].values
    
    counts = np.zeros_like(lons, dtype=int) # initialise the counts matrix
    fp_ind = np.argwhere(disk(r_disk)) # footprint
    
    # loop over the days
    for SC_file in filelist:
        print (SC_file)
        # initialise an intermediate counts matrix for this day, which will contain its supercells tracks footprint; this incidentally avoids overlaps
        counts_day = np.zeros_like(lons, dtype=int) 
        
        # load the supercells of the given day
        with open(SC_file, "r") as read_file:
            SC_info = json.load(read_file)['supercell_data']
        if len(SC_info)==0: continue
        
        # loop over the supercells of the day
        for SC in SC_info:
            
            # if 2 meso detections are required, filter out one-meso detection supercells
            if twoMD:
                if len(np.unique(SC['meso_datelist'])) < 2:
                    continue
            
            # initialise an intermediate counts matrix for this SC, which will contain its track footprint; this incidentally avoids overlaps
            counts_SC = np.zeros_like(lons, dtype=int)
            
            SC_info_hours = pd.DatetimeIndex(SC['cell_datelist']).hour
            ii = np.where((SC_info_hours>=int(start_hr)) & (SC_info_hours<=int(end_hr)))[0]
            #SC_info = SC_info[(SC_info_hours>=int(start_hr)) & (SC_info_hours<=int(end_hr))]
            
            SC_lons = np.array(SC['cell_lon'])[ii]
            SC_lats = np.array(SC['cell_lat'])[ii]
            
            for j in range(len(SC_lons)): # for each SC centre of mass location
                distances = (lons - SC_lons[j])**2 + (lats - SC_lats[j])**2 # interpolate it from the lon-lat coords to the grid indices
                k,l = np.unravel_index(np.argmin(distances), distances.shape) #indices on grid corresponding to the SC centre of mass interpolation
                
                try:
                    counts_SC[k-r_disk+fp_ind[:,0], l-r_disk+fp_ind[:,1]] = 1
                except IndexError:
                    # Handle the case where the index is out of bounds -> simply skip the footprint printing
                    print('Index is out of bounds for the matrix and therefore skipped')
            
            counts_day += counts_SC
            
        counts += counts_day
        
    return lons, lats, counts

def dist_latlon(lats,lons):
    distance = []
    for n in range(len(lats)-1):
        lat1 = math.radians(lats[n]); lat2 = math.radians(lats[n+1])
        lon1 = math.radians(lons[n]); lon2 = math.radians(lons[n+1])
        
        inter = ( math.sin(lat1) * math.sin(lat2) )  + ( math.cos(lat1) * math.cos(lat2)  * (math.cos(lon2 - lon1) ) )
        if abs(inter) > 1 : dist = np.nan
        else: dist = math.acos( inter )  * 6371
    
        distance.append(dist)
    
    length = np.nansum(distance)
    
    return length

def bearing_latlon(lats,lons):
    
    
    lat1 = math.radians(lats[0]); lat2 = math.radians(lats[-1])
    lon1 = math.radians(lons[0]); lon2 = math.radians(lons[-1])

    y = math.sin(lon2-lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2-lon1)

    brng = math.atan2(y, x)  
    
    bearing = np.rad2deg(brng)

    return bearing

def supercell_lifecycle(path, rainpath, twoMD=False, skipped_days=None):
    """
    Extracts lifecycle properties of supercells
    
    Parameters
    ----------

    path : str
        path to the SDT output files
    twoMD : bool
        if True, filters out the supercells exhibiting less than 2 mesocyclone detections at different time steps
    skipped_days : list of str
        list of missing days which consequently must be skipped

    Returns
    -------
    SC_properties : pandas dataframe
        lifecycle properties of supercells
    """
    
    
    
    SC_files = sorted(glob.glob(path+'/*.json'))
    

    SC_length = []
    SC_duration = []
    SC_marea_sum = []
    SC_marea_av = []
    SC_hail_max = []
    SC_hail_av = []
    SC_init_h = []
    SC_init_lat = []
    SC_init_lon = []
    SC_dir = []
    SC_mover = []
    SC_rarea_sum = []
    SC_rarea_av = []
    SC_zeta_max = []
    SC_zeta_av = []
    SC_w_max = []
    SC_w_av = []
    SC_v_max = []
    SC_v_av = []
    SC_r_max = []
    SC_r_av = []
    SC_date = []
    SC_id = []
    SC_n_meso = []
    # loop over the days
    a=0
    for SC_file in SC_files:
        print (SC_file)
        # initialise an intermediate counts matrix for this day, which will contain its supercells tracks footprint; this incidentally avoids overlaps
        
        # load the supercells of the given day
        with open(SC_file, "r") as read_file:
            SC_info = json.load(read_file)['supercell_data']
            if len(SC_info)==0: continue
            #MC_info = json.load(read_file)['na_vortex_data']
        #if len(SC_info)==0: continue
        
        # loop over the supercells of the day
        #a=0
        for SC in SC_info:
            #MC = MC_info[a]
            # if 2 meso detections are required, filter out one-meso detection supercells
            if twoMD:
                if len(np.unique(SC['meso_datelist'])) < 2:
                    continue
            
            
            SC_lons = np.array(SC['cell_lon'])
            SC_lats = np.array(SC['cell_lat'])
            

            SC_length.append(dist_latlon(SC_lats, SC_lons))
            SC_duration.append(pd.to_datetime(SC['cell_datelist'][-1]) - pd.to_datetime(SC['cell_datelist'][0]))
            SC_date.append(SC['cell_datelist'][0])
            SC_id.append(SC['rain_cell_id'])
            SC_marea_sum.append(np.nansum(np.array(SC['area'])))
            SC_marea_av.append(np.nanmean(np.array(SC['area'])))
            SC_zeta_max.append(np.nanmax(abs(np.array(SC['max_zeta']))))
            SC_zeta_av.append(np.nanmean(abs(np.array(SC['max_zeta']))))
            SC_w_max.append(np.nanmax(np.array(SC['max_w'])))
            SC_w_av.append(np.nanmean(np.array(SC['max_w'])))
            SC_hail_max.append(np.nanmax(np.array(SC['cell_max_hail'])))
            SC_hail_av.append(np.nanmean(np.array(SC['cell_max_hail'])))
            SC_v_max.append(np.nanmax(np.array(SC['cell_max_wind'])))
            SC_v_av.append(np.nanmean(np.array(SC['cell_max_wind'])))
            SC_r_max.append(np.nanmax(np.array(SC['cell_max_rain'])))
            SC_r_av.append(np.nanmean(np.array(SC['cell_max_rain'])))
            SC_init_h.append(pd.DatetimeIndex(SC['cell_datelist']).hour[0])
            SC_init_lat.append(SC_lats[0])
            SC_init_lon.append(SC_lons[0])
            SC_dir.append(bearing_latlon(SC_lats, SC_lons))
            SC_n_meso.append(len(SC['signature']))
            sigs = np.array(SC['signature'])
            if np.nanmean(sigs) == 1:
                SC_mover.append(1)
            elif np.nanmean(sigs) == -1:
                SC_mover.append(-1)
            elif np.nanmean(sigs) > 0.2:
                SC_mover.append(0.5)
            elif np.nanmean(sigs) < -0.2:
                SC_mover.append(-0.5)
            else:
                SC_mover.append(0)
            date = SC_file[-13:-5]
            rain_file = glob.glob(rainpath+'*'+date+'*.json')[0]
            #print(rain_file)
            with open(rain_file, "r") as read_file:
                R_info = json.load(read_file)['cell_data']
                for RC in R_info:
                    if RC['cell_id']==SC['rain_cell_id']:
                        
                        SC_rarea_sum.append(np.nansum(np.array(RC['area_gp'])))
                        SC_rarea_av.append(np.nanmean(np.array(RC['area_gp'])))
            
            a += 1
            
            # if a == 100:
            #     return SC_length, SC_duration, SC_marea_sum, SC_marea_av, \
            #             SC_hail_max, SC_hail_av, SC_init_h, SC_init_lat, SC_init_lon, \
            #             SC_dir, SC_mover, SC_rarea_sum, SC_rarea_av, SC_zeta_max, SC_zeta_av, \
            #             SC_w_max, SC_w_av, SC_v_max, SC_v_av, SC_r_max, SC_r_av, SC_date, \
            #             SC_id, SC_n_meso
            #     break

    
    SC_properties = pd.DataFrame({
        'SC_date': SC_date,
        'SC_id': SC_id,
        'SC_length': SC_length,
        'SC_duration': SC_duration,
        'SC_marea_sum': SC_marea_sum,
        'SC_marea_av': SC_marea_av,
        'SC_init_h': SC_init_h,
        'SC_init_lat': SC_init_lat,
        'SC_init_lon': SC_init_lon,
        'SC_dir': SC_dir,
        'SC_hail_max': SC_hail_max,
        'SC_hail_av': SC_hail_av,
        'SC_mover': SC_mover,
        'SC_v_max': SC_v_max,
        'SC_v_av': SC_v_av,
        'SC_r_max': SC_r_max,
        'SC_r_av': SC_r_av,
        'SC_rarea_sum': SC_rarea_sum,
        'SC_rarea_av': SC_rarea_av,
        'SC_zeta_max': SC_zeta_max,
        'SC_zeta_av': SC_zeta_av,
        'SC_w_max': SC_w_max,
        'SC_w_av': SC_w_av,
        'SC_n_meso': SC_n_meso,
        })
            
        
    return SC_properties




def write_to_netcdf(lons, lats, counts, filename):
    """
    Writes netcdf file containing mesocyclones binary masks

    in
    lons: longitude at each gridpoint, 2D array
    lats: latitude at each gridpoint, 2D array
    filename: file path and name, string

    out
    ds: 2D xarray dataset containing the frequency map, xarray dataset
    """

    coords = {
        "y": np.arange(lats.shape[0]),
        "x": np.arange(lons.shape[1]),
    }
    data_structure = {
        "frequency_map": (["y", "x"], counts),
        "lat": (["y", "x"], lats),
        "lon": (["y", "x"], lons),
    }

    # create netcdf file
    ds = xr.Dataset(data_structure, coords=coords)

    # write to netcdf file
    # ds.to_netcdf(filename)
    ds.to_netcdf(filename, encoding={'frequency_map': {'zlib': True, 'complevel': 9}})


