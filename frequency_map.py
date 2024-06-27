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
from matplotlib import gridspec
#from sklearn.neighbors import KernelDensity

#==================================================================================================================================================
# FUNCTIONS
#==================================================================================================================================================

def plot_fmap(lons, lats, fmap, typ, r_disk, climate="current", season=False, year=None, zoom=False, save=False, addname=""):
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
    year : str or None
        if seasonal, year YYYY of the season. Elsewise, None. The default is None.
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
    
    if typ == "supercell" and season:
        # mask the 0 values
        fmap = np.array(fmap.astype(float))
        fmap[fmap<0.001] = np.nan
        # set the norm
        bounds = [1, 2, 3, 4, 5, 6, 7]
        bounds_str = [str(x) for x in bounds]
        norm = BoundaryNorm(boundaries=bounds, ncolors=256, extend='max')
    
    elif typ == "supercell" and not season:
        # set the norm
        if r_disk==2:
            bounds = [0, 1/11, 2/11, 4/11, 6/11, 8/11, 1, 14/11, 17/11, 20/11, 2, 25/11]
            bounds_str = ["0", "1/11", "2/11", "4/11", "6/11", "8/11", "1", "14/11", "17/11", "20/11", "2", "25/11"]
        elif r_disk==0:
            bounds = [0, 1/11, 2/11, 3/11, 4/11, 5/11, 6/11, 7/11, 8/11, 9/11]
            bounds_str = ["0", "1/11", "2/11", "3/11", "4/11", "5/11", "6/11", "7/11", "8/11", "9/11"]
        elif r_disk==5:
            bounds = [0, 1/11, 4/11, 8/11, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
            bounds_str = ["0", "1/11", "4/11", "8/11", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5"]
        else:
            raise ValueError("Radius of influence ot considered")
        
        norm = BoundaryNorm(boundaries=bounds, ncolors=256, extend='max')
        #norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)
        
    elif typ == "mesocyclone":
        # mask the 0 values
        fmap = np.array(fmap.astype(float))
        fmap[fmap<0.01] = np.nan
        # set the norm
        bounds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        bounds_str = [str(x) for x in bounds]
        norm = BoundaryNorm(boundaries=bounds, ncolors=256, extend='max')
    else:
        raise TypeError("function not designed for rain so far")
    
    # determine the area of influence based on the footprint
    aoi = np.count_nonzero(disk(r_disk))*4.84
    aoi = round(aoi)
        
    # load geographic features
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol,
                                        facecolor='none', edgecolor='k', linestyle='-', alpha=1, linewidth=0.7)
    coastline = cfeature.NaturalEarthFeature(category='physical', name='coastline', scale=resol, facecolor='none',
                                             edgecolor='k', linestyle='-', alpha=1, linewidth=0.7)
    lakes = cfeature.NaturalEarthFeature(category='physical', name='lakes', scale=resol, facecolor='none',
                                         edgecolor='blue', linestyle='-', alpha=1, linewidth=0.8)
    rp = ccrs.RotatedPole(pole_longitude = -170, pole_latitude = 43)
    
    if zoom:
        fig = plt.figure(figsize=(10,10))
    else:
        fig = plt.figure(figsize=(10,8))
    
    ax = plt.axes(projection=rp)
    ax.add_feature(bodr)
    ax.add_feature(coastline)
    
    if season:
        figname = "season" + year + "_" + typ + "_disk" + str(r_disk)
        title = "Season " + year + " " + typ + " distribution map"
        lab = r"number of " + typ + "s per " + str(aoi) + " $km^2$"
    else:
        figname = "decadal_" + typ + "_disk" + str(r_disk)
        title = climate + " climate decadal " + typ + " distribution map"
        lab = r"number of " + typ + "s per year per " + str(aoi) + " $km^2$"
    
    if zoom:
        zl, zr, zb, zt = 750, 400, 570, 700 #cut for Alpine region
        figname += "_alps"
        cbar_orientation = "horizontal"
        ax.add_feature(lakes)
    else:
        zl, zr, zb, zt = 180, 25, 195, 25 #smart cut for entire domain
        cbar_orientation = "vertical"

    cont = ax.pcolormesh(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], fmap[zb:-zt,zl:-zr], norm=norm, cmap="CMRmap_r", transform=ccrs.PlateCarree())
    
    if addname:
        figname += "_" + addname + ".png"
        title += ", " + addname
    
    cbar = plt.colorbar(cont, orientation=cbar_orientation, label=lab)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels(bounds_str)
        
    plt.title(title)
    
    if save:
        fig.savefig(figname, dpi=300)
    
    return

def plot_supercell_seasonal_fmap(start_day, end_day, path, r_disk, climate, zoom=False, save=False, addname=""):
    """
    from SDT2 output data, computes and plots the (averaged per year) supercell seasonal frequency map of the requested climate
    of the period delimited by the starting and ending days; designed for consecutive months

    Parameters
    ----------
    start_day : str
        first day of the considered period, mmdd
    end_day : str
        last day of the considered period, mmdd
    path : str
        path to the SDT output files
    r_disk : int
        radius of the disk footprint in grid point
    climate : str
        "future" or "current"
    zoom : bool, optional
        False: smart cut of whole domain (ocean and Northern Africa discarded) ; True: zoom on the Alps. The default is False.
    save : bool
        option to save figure
    addname: str
        option to add additional information to figure name and title

    Returns
    -------
    plots the desired map and saves it if requested
    """
    
    if climate=="current":
        years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    else:
        years = ['2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095']
    
    # determine the delimiting months from the dates
    months_num = ["04", "05", "06", "07", "08", "09", "10", "11"]
    months = ["April", "May", "June", "July", "August", "September", "October", "November"]
    start_month = months[months_num.index(start_day[:2])]
    end_month = months[months_num.index(end_day[:2])]
    
    # compute the seasonal map
    for i, year in enumerate(years):
        day_i = year + start_day
        day_f = year + end_day
        if i==0:
            lons, lats, counts_SC = seasonal_supercell_tracks_model_fmap(day_i, day_f, path, r_disk)
        else:
            _, _, counts = seasonal_supercell_tracks_model_fmap(day_i, day_f, path, r_disk)
            counts_SC += counts
    counts_SC = counts_SC/len(years)
    
    # set the bounds and norm
    if r_disk==2:
        bounds = [0, 1/11, 2/11, 3/11, 4/11, 5/11, 6/11, 7/11, 8/11, 9/11, 10/11, 1]
        bounds_str = ["0", "1/11", "2/11", "3/11", "4/11", "5/11", "6/11", "7/11", "8/11", "9/11", "10/11", "1"]
    elif r_disk==5:
        bounds = [0, 1/11, 2/11, 4/11, 6/11, 8/11, 1, 14/11, 17/11, 20/11, 2, 25/11]
        bounds_str = ["0", "1/11", "2/11", "4/11", "6/11", "8/11", "1", "14/11", "17/11", "20/11", "2", "25/11"]
    else:
        raise ValueError("Radius of influence not considered")
    norm = BoundaryNorm(boundaries=bounds, ncolors=256, extend='max')
    
    # determine the area of influence based on the footprint
    aoi = np.count_nonzero(disk(r_disk))*4.84
    aoi = round(aoi)
        
    # load geographic features
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol,
                                        facecolor='none', edgecolor='k', linestyle='-', alpha=1, linewidth=0.7)
    coastline = cfeature.NaturalEarthFeature(category='physical', name='coastline', scale=resol, facecolor='none',
                                             edgecolor='k', linestyle='-', alpha=1, linewidth=0.7)
    lakes = cfeature.NaturalEarthFeature(category='physical', name='lakes', scale=resol, facecolor='none',
                                         edgecolor='blue', linestyle='-', alpha=1, linewidth=0.8)
    rp = ccrs.RotatedPole(pole_longitude = -170, pole_latitude = 43)
    
    if zoom:
        fig = plt.figure(figsize=(10,10))
    else:
        fig = plt.figure(figsize=(10,8))
    
    ax = plt.axes(projection=rp)
    ax.add_feature(bodr)
    ax.add_feature(coastline)
    
    figname = "season_" + start_month + "_" + end_month + "_disk" + str(r_disk)
    title = climate + " climate seasonal " + start_month + "-" + end_month + " supercell distribution map"
    lab = r"number of supercells per year per " + str(aoi) + " $km^2$"
    
    if zoom:
        zl, zr, zb, zt = 750, 400, 570, 700 #cut for Alpine region
        figname += "_alps"
        cbar_orientation = "horizontal"
        ax.add_feature(lakes)
    else:
        zl, zr, zb, zt = 180, 25, 195, 25 #smart cut for entire domain
        cbar_orientation = "vertical"

    cont = ax.pcolormesh(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], counts_SC[zb:-zt,zl:-zr], norm=norm, cmap="CMRmap_r", transform=ccrs.PlateCarree())
    
    if addname:
        figname += "_" + addname + ".png"
        title += ", " + addname
    
    cbar = plt.colorbar(cont, orientation=cbar_orientation, label=lab)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels(bounds_str)
        
    plt.title(title)
    
    if save:
        fig.savefig(figname, dpi=300)
    
    return


def plot_supercell_tracks_model_delta_map(r_disk, perc=False, zoom=False, save=False, addname=""):
    """
    from seasonally stored data, computes the current and future climate decadal supercell frequency maps and plots the
    difference future - current

    Parameters
    ----------
    r_disk : int
        radius of the disk footprint in grid point
    perc : boo
        if True, the change is expressed in percentage; if False, the absolute change is computed
    zoom : bool, optional
        False: smart cut of whole domain (ocean and Northern Africa discarded) ; True: zoom on the Alps. The default is False.
    save : bool
        option to save figure
    addname: str
        option to add additional information to figure name and title

    Returns
    -------
    plots the desired delat map and saves it if requested
    """
    
    # determine the area of influence based on the footprint
    aoi = np.count_nonzero(disk(r_disk))*4.84
    aoi = round(aoi)
    
    years_CC = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    years_FC = ['2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095']
    
    for i, season in enumerate(years_CC):
        fname = "/scratch/snx3000/mblanc/fmaps_data/SDT2/current_climate/model_tracks/SC_season" + season + "_disk" + str(r_disk) + ".nc"
        if i == 0:
            with xr.open_dataset(fname) as dset:
                counts_CC = dset['frequency_map']
                lons = dset['lon'].values
                lats = dset['lat'].values
        else:
            with xr.open_dataset(fname) as dset:
                counts = dset['frequency_map']
            counts_CC += counts
    
    for i, season in enumerate(years_FC):
        fname = "/scratch/snx3000/mblanc/fmaps_data/SDT2/future_climate/model_tracks/SC_season" + season + "_disk" + str(r_disk) + ".nc"
        if i == 0:
            with xr.open_dataset(fname) as dset:
                counts_FC = dset['frequency_map']
        else:
            with xr.open_dataset(fname) as dset:
                counts = dset['frequency_map']
            counts_FC += counts
            
    counts_CC, counts_FC = counts_CC/11, counts_FC/11
    
    if perc:
        counts_CC = np.array(counts_CC.astype(float))
        counts_CC[counts_CC==0] = np.nan # replace 0 values by nans
        delta = 100*(counts_FC - counts_CC)/counts_CC # compute relative change
        #print(np.nanmin(delta), np.nanmax(delta))
        lab = r"Delta (%)"
        figname = "delta_decadal_supercell_disk" + str(r_disk) + "_rel"
        if r_disk==2:
            bounds = [-100, -90, -70, -50, -30, -10, 10, 40, 80, 130, 200, 500]
            bounds_str = ["-100", "-90", "-70", "-50", "-30", "-10", "10", "40", "80", "130", "200", "500"]
        elif r_disk==5:
            bounds = [-100, -90, -70, -50, -30, -10, 10, 40, 80, 130, 200, 500]
            bounds_str = ["-100", "-90", "-70", "-50", "-30", "-10", "10", "40", "80", "130", "200", "500"]
        else:
            raise ValueError("Radius of influence not considered")
    else:
        delta = counts_FC - counts_CC # compute absolute change
        #print(np.min(delta), np.max(delta))
        lab = r"Delta (supercells per year per " + str(aoi) + " $km^2$)"
        figname = "delta_decadal_supercell_disk" + str(r_disk) + "_abs"
        eps = 1e-3
        if r_disk==2:
            bounds = [-2+eps, -19/11+eps, -15/11+eps, -1, -7/11+eps, -4/11+eps, -1/11+eps, 1/11, 4/11, 7/11, 1, 15/11, 19/11, 2]
            bounds_str = ["-2", "-19/11", "-15/11", "-1", "-7/11", "-4/11", "-1/11", "1/11", "4/11", "7/11", "1", "15/11", "19/11", "2"]
        elif r_disk==5:
            bounds = [-2+eps, -19/11+eps, -15/11+eps, -1, -7/11+eps, -4/11+eps, -1/11+eps, 1/11, 4/11, 7/11, 1, 15/11, 19/11, 2]
            bounds_str = ["-2", "-19/11", "-15/11", "-1", "-7/11", "-4/11", "-1/11", "1/11", "4/11", "7/11", "1", "15/11", "19/11", "2"]
        else:
            raise ValueError("Radius of influence ot considered")
    norm = BoundaryNorm(boundaries=bounds, ncolors=256, extend='both')
    
    # load geographic features
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol,
                                        facecolor='none', edgecolor='k', linestyle='-', alpha=1, linewidth=0.7)
    coastline = cfeature.NaturalEarthFeature(category='physical', name='coastline', scale=resol, facecolor='none',
                                             edgecolor='k', linestyle='-', alpha=1, linewidth=0.7)
    lakes = cfeature.NaturalEarthFeature(category='physical', name='lakes', scale=resol, facecolor='none',
                                         edgecolor='blue', linestyle='-', alpha=1, linewidth=0.8)
    rp = ccrs.RotatedPole(pole_longitude = -170, pole_latitude = 43)
    
    if zoom:
        fig = plt.figure(figsize=(10,10))
    else:
        fig = plt.figure(figsize=(10,8))
    
    ax = plt.axes(projection=rp)
    ax.add_feature(bodr)
    ax.add_feature(coastline)
    title = "delta decadal supercell distribution map"
    
    if zoom:
        zl, zr, zb, zt = 750, 400, 570, 700 #cut for Alpine region
        figname += "_alps"
        cbar_orientation = "horizontal"
        ax.add_feature(lakes)
    else:
        zl, zr, zb, zt = 180, 25, 195, 25 #smart cut for entire domain
        cbar_orientation = "vertical"

    cont = ax.pcolormesh(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], delta[zb:-zt,zl:-zr], norm=norm, cmap="seismic", transform=ccrs.PlateCarree())
    
    if addname:
        figname += "_" + addname + ".png"
        title += ", " + addname
    
    cbar = plt.colorbar(cont, orientation=cbar_orientation, label=lab)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels(bounds_str)
        
    plt.title(title)
    
    if save:
        fig.savefig(figname, dpi=300)
    
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


def seasonal_supercell_tracks_model_fmap(start_day, end_day, path, r_disk, twoMD=False, skipped_days=None):
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


def seasonal_rain_tracks_model_fmap(season, path, skipped_days=None, conv=True):
    """
    Computes the seasonal rain frequency map using tracks method over the model whole domain
    Takes care of overlaps by default
    
    Parameters
    ----------
    season : str
        considered season, YYYY
    path : str
        path to the rain tracks (cell tracker outputs)
    skipped_days : list of str
        list of missing days which consequently must be skipped
    conv : bool
        False: number of storms per grid box, ie per 4.84 km^2; True: 4-connectivity sum convolution yields the number of storms per 24.2 km^2
    
    Returns
    -------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    counts : 2D array
        the desired seasonal rain frequency map
    """
    
    start_day = pd.to_datetime(season + "0401")
    end_day = pd.to_datetime(season + "0930")
    daylist = pd.date_range(start_day, end_day)
    
    # remove skipped days from daylist
    if skipped_days:
        skipped_days = pd.to_datetime(skipped_days, format="%Y%m%d")
        daylist = [day for day in daylist if day not in skipped_days]
    
    # load lons and lats static fields
    with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/model_masks/meso_season2011.nc") as dset:
        lons = dset["lon"].values
        lats = dset["lat"].values
    
    files = []
    for day in daylist:
        files.append(path + "cell_tracks_" + day.strftime("%Y%m%d") + ".json")
    
    counts = np.zeros_like(lons, dtype=int) # initialise the counts matrix
    fp_ind = np.argwhere(disk(1)) # footprint: 4-connectivity; will serve for marking the supercell core footprin
    
    # loop over the days of the season
    for file in files:
        # initialise an intermediate counts matrix for this day, which will contain its supercells tracks footprint; this incidentally avoids overlaps
        counts_day = np.zeros_like(lons, dtype=int) 
        
        # load the supercells of the given day
        with open(file, "r") as read_file:
            info = json.load(read_file)['cell_data']
        
        # loop over the supercells of the day
        for cell in info:
            
            # initialise an intermediate counts matrix for this cell, which will contain its track footprint; this incidentally avoids overlaps
            counts_cell = np.zeros_like(lons, dtype=int)
            cell_lons = cell['lon']
            cell_lats = cell['lat']
            
            for j in range(len(cell_lons)): # for each cell centre of mass location
                distances = (lons - cell_lons[j])**2 + (lats - cell_lats[j])**2 # interpolate it from the lon-lat coords to the grid indices
                k,l = np.unravel_index(np.argmin(distances), distances.shape) #indices on grid corresponding to the cell centre of mass interpolation
                counts_cell[k-1+fp_ind[:,0], l-1+fp_ind[:,1]] = 1 # the 4-connectivity disk around centre of mass as proxy of cell core
            
            if conv:
                counts_cell = dilation(counts_cell, disk(1)) # 4-connectivity dilation -> avoids SC overlaps with themselves
            
            counts_day += counts_cell
            
        counts += counts_day
        
    return lons, lats, counts


def seasonal_supercell_tracks_obs_fmap(start_day, end_day, conv=True):
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
    with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/model_masks/meso_season2011.nc") as dset:
        lons = dset["lon"].values
        lats = dset["lat"].values
    
    counts = np.zeros_like(lons, dtype=int) # initialise the counts matrix
    fp_ind = np.argwhere(disk(1)) # footprint: 4-connectivity; will serve for marking the supercell core footprint
    
    if season == "2022": # season 2022, special meso dataset
        
        # open meso dataset
        usecols = ['ID','time','lon','lat']
        mesoset = pd.read_csv("/scratch/snx3000/mblanc/observations/TRTc_mesostorm_2022.csv", sep=';', usecols=usecols)
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
            
            for j in range(len(SC_lons)): # for each SC centroid location
                distances = (lons - SC_lons[j])**2 + (lats - SC_lats[j])**2 # interpolate it from the lon-lat coords to the grid indices
                k,l = np.unravel_index(np.argmin(distances), distances.shape) #indices on grid corresponding to the SC centroid interpolation
                counts_SC[k-1+fp_ind[:,0], l-1+fp_ind[:,1]] = 1 # the 4-connectivity disk around centre of mass as proxy of SC core
            
            if conv:
                counts_SC = dilation(counts_SC, disk(1)) # 4-connectivity dilation -> avoids SC overlaps with themselves
            
            counts += counts_SC
        
    else: # season within the 2016-2021 dataset
        
        # open full observational dataset
        usecols = ['ID','time','mesostorm','lon','lat']
        fullset = pd.read_csv("/scratch/snx3000/mblanc/observations/Full_dataset_thunderstorm_types.csv", sep=';', usecols=usecols)
        fullset['time'] = pd.to_datetime(fullset['time'], format="%Y%m%d%H%M")
        days_str = [dt.strftime("%Y%m%d") for dt in fullset['time']]

        # restrict to the given season and select the supercells
        selection = fullset[np.isin(days_str, daylist_str)]
        selection = selection[selection['mesostorm']==1]
        selection['ID'] = [int(round(x)) for x in selection['ID']]
        SC_ids = np.unique(selection['ID']) # extract the IDs of all the supercells of the considered period
        
        # loop over the supercells of the season
        for SC_ID in SC_ids:
            counts_SC = np.zeros_like(lons, dtype=int) # initialise an intermediate counts matrix for this SC, which will contain its track footprint; this incidentally avoids overlaps
            SC_lons = selection['lon'][selection['ID'] == SC_ID] # get the lon-lat coordinates of the SC track
            SC_lons = np.reshape(SC_lons, len(SC_lons))
            SC_lats = selection['lat'][selection['ID'] == SC_ID]
            SC_lats = np.reshape(SC_lats, len(SC_lats))
            
            for j in range(len(SC_lons)): # for each SC centroid location
                distances = (lons - SC_lons[j])**2 + (lats - SC_lats[j])**2 # interpolate it from the lon-lat coords to the grid indices
                k,l = np.unravel_index(np.argmin(distances), distances.shape) #indices on grid corresponding to the SC centroid interpolation
                counts_SC[k-1+fp_ind[:,0], l-1+fp_ind[:,1]] = 1 # the 4-connectivity disk around centre of mass as proxy of SC core
            
            if conv:
                counts_SC = dilation(counts_SC, disk(1)) # 4-connectivity dilation -> avoids SC overlaps with themselves
            
            counts += counts_SC
    
    return lons, lats, counts


def seasonal_meso_masks_model_fmap(season, path):
    """
    Computes the seasonal mesocyclone frequency map using masks method over the model domain
    
    Parameters
    ----------
    season : str
        considered season, YYYY
    path : str
        path to the mesocyclone masks, SDT2 outputs

    Returns
    -------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    counts : 2D array
        the desired seasonal frequency map
    """
    
    start_day = pd.to_datetime(season + "0401")
    end_day = pd.to_datetime(season + "0930")
    daylist = pd.date_range(start_day, end_day)
    
    meso_masks_files = []
    for day in daylist:
        meso_masks_files.append(path + "/meso_masks_" + day.strftime("%Y%m%d") + ".nc")
    
    for i, meso_masks_file in enumerate(meso_masks_files):
        
        with xr.open_dataset(meso_masks_file) as dset:
            meso_masks = dset['meso_mask']
            if i == 0:
                lons = dset["lon"].values
                lats = dset["lat"].values
        
        if i == 0:
            counts = np.count_nonzero(meso_masks, axis=0)
        else:
            counts += np.count_nonzero(meso_masks, axis=0)
    
    return lons, lats, counts


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


def supercell_tracks_model_obs_comp_2016_2021_fmaps(conv=True, save=False):
    """
    Computes the sub-period 2016-2021 April-September observational and modelled supercell frequency maps using tracks method over the Apline region
    Assumes the model seasonal maps are already computed and stored
    Takes care of overlaps by default
    filtering and conv of model data depends on the stored seasonal maps
    
    Parameters
    ----------
    conv : bool, for the observational data
        False: number of storms per grid box, ie per 4.84 km^2; True: 4-connectivity sum convolution yields the number of storms per 24.2 km^2
    save : bool
        option to save the figure

    Returns
    -------
    plots the 2 maps in a subplot figure and saves it if requested
    """
    
    years = ["2016", "2017", "2018", "2019", "2020", "2021"]
    
    # model data : April-September seasonal maps already stored
    for i, year in enumerate(years):
        if i == 0:
            with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/model_tracks/SC_season" + year + "_filtered_conv.nc") as dset:
                counts_model = dset['frequency_map']
                lons = dset['lon'].values
                lats = dset['lat'].values
        else:
            with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/model_tracks/SC_season" + year + "_filtered_conv.nc") as dset:
                counts = dset['frequency_map']
            counts_model += counts
        
    # obs data : April-September seasonal maps to be computed
    for i, year in enumerate(years):
        start_day = year + "0401"
        end_day = year + "0930"
        if i == 0:
            _, _, counts_obs = seasonal_supercell_tracks_obs_fmap(start_day, end_day, conv=conv)
        else:
            _, _, counts = seasonal_supercell_tracks_obs_fmap(start_day, end_day, conv=conv)
            counts_obs += counts
    
    # convert the counts per unit year
    counts_obs = counts_obs/6
    counts_model = counts_model/6
    
    # colorbar parameters
    max_count = max(np.max(counts_model), np.max(counts_obs))
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5*max_count, vmax=max_count)
    
    # load geographic features
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', scale=resol, facecolor='none')
    zl, zr, zb, zt = 780, 550, 630, 720 # cut for swiss radar network
    
    # figure
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

    # Subplot 1
    ax1 = plt.subplot(gs[0], projection=ccrs.PlateCarree())
    ax1.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    ax1.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.2)
    cont_model = ax1.pcolormesh(lons[zb:-zt, zl:-zr], lats[zb:-zt, zl:-zr], counts_model[zb:-zt, zl:-zr], norm=norm, cmap="Reds", transform=ccrs.PlateCarree())
    ax1.set_title("modelled supercells")

    # Subplot 2
    ax2 = plt.subplot(gs[1], projection=ccrs.PlateCarree(), sharex=ax1, sharey=ax1)
    ax2.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    ax2.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.2)
    cont_obs = ax2.pcolormesh(lons[zb:-zt, zl:-zr], lats[zb:-zt, zl:-zr], counts_obs[zb:-zt, zl:-zr], norm=norm, cmap="Reds", transform=ccrs.PlateCarree())
    ax2.set_title("observed supercells")

    # Colorbar
    cbar_ax = plt.subplot(gs[2])
    if conv:
        plt.colorbar(cont_obs, cax=cbar_ax, orientation='vertical', label=r"number of supercells per year per 24.2 $km^2$")
    else:
        plt.colorbar(cont_obs, cax=cbar_ax, orientation='vertical', label=r"number of supercells per year per 4.84 $km^2$")
    # obs has more counts than model so we use the obs frequency maps as the base for the colorbar
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the rectangle to make room for suptitle
    
    # Suptitle
    fig.suptitle("2016-2021 April-September", fontsize=14)
    
    if save:
        fig.savefig("2016_2021_model_obs_comp_SCfmaps.png", dpi=300)
   
#==================================================================================================================================================
# MAIN
#==================================================================================================================================================
# create data, plot and store them

## model data ##

# parser = argparse.ArgumentParser()
# parser.add_argument("r_disk", type=int)
# args = parser.parse_args()
# r_disk = args.r_disk

# # skipped_days = ['20120604', '20140923', '20150725', '20160927', '20170725']
#climate = "future"
#start_day = "0401"
#end_day = "1130"
#years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
#years = ['2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095']
# #years = ["2019", "2020", "2021"]
#method = "model_tracks"
# ##iuh_thresh = args.iuh_thresh#
#path = "/scratch/snx3000/mblanc/SDT/SDT2_output/" + climate + "_climate/domain/XPT_1MD_zetath5_wth5/"
#r_disk = 5

# for season in years:
#     lons, lats, counts_meso = seasonal_meso_masks_model_fmap(season, path)
#     #plot_fmap(lons, lats, counts_meso, "mesocyclone", r_disk=0, season=True, year=season, zoom=False, save=True, addname="")
#     #plot_fmap(lons, lats, counts_meso, "mesocyclone", r_disk=0, season=True, year=season, zoom=True, save=True, addname="")
#     filename_meso = "/scratch/snx3000/mblanc/fmaps_data/SDT2/" + climate + "_climate/" + method + "/meso_season" + season + ".nc"
#     write_to_netcdf(lons, lats, counts_meso, filename_meso)

# lons, lats, counts_SC = seasonal_masks_fmap(season, "supercell", climate, resolve_ovl, filtering=filtering, skipped_days=skipped_days)
# plot_fmap(lons, lats, counts_SC, "supercell", season=True, year=season, filtering=filtering, conv=True)
# plot_fmap(lons, lats, counts_SC, "supercell", season=True, year=season, zoom=True, filtering=filtering, conv=True)
# filename_SC = "/scratch/snx3000/mblanc/fmaps_data/" + method + "/SC_season" + season + "_filtered.nc"
# write_to_netcdf(lons, lats, counts_SC, filename_SC)

# for season in years:
#     day_i = season + start_day
#     day_f = season + end_day
#     lons, lats, counts_SC = seasonal_supercell_tracks_model_fmap(day_i, day_f, path, r_disk)
#     # plot_fmap(lons, lats, counts_SC, "supercell", r_disk=2, season=True, year=season, zoom=False, save=True, addname="")
#     # plot_fmap(lons, lats, counts_SC, "supercell", r_disk=2, season=True, year=season, zoom=True, save=True, addname="")
#     filename_SC = "/scratch/snx3000/mblanc/fmaps_data/SDT2/" + climate + "_climate/" + method + "/SC_season" + season + "_disk" + str(r_disk) + ".nc"
#     write_to_netcdf(lons, lats, counts_SC, filename_SC)

## obs data ##

# parser = argparse.ArgumentParser()
# parser.add_argument("start_day", help="start day", type=str)
# parser.add_argument("end_day", help="end day", type=str)
# args = parser.parse_args()

# method = "obs_tracks"
# start_day = args.start_day
# end_day = args.end_day
# season = pd.to_datetime(start_day).strftime("%Y")

# lons, lats, counts_SC = seasonal_supercell_tracks_obs_fmap(start_day, end_day, conv=True)
# plot_fmap(lons, lats, counts_SC, "supercell", season=True, year=season, zoom=True, conv=True, save=True)
# filename_SC = "/scratch/snx3000/mblanc/fmaps_data/" + method + "/Apr-Oct/SC_season" + season + "_conv.nc"
# write_to_netcdf(lons, lats, counts_SC, filename_SC)

#==================================================================================================================================================
## seasonal map from stored data ##

# compare different versions

# season = "2019"
# climate = "current"
# years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
# method = "model_tracks"
# typ = "supercell"

# with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/SDT1/" + method + "/SC_season" + season + "_filtered_conv.nc") as dset:
#     counts1 = dset['frequency_map']
#     lons = dset['lon'].values
#     lats = dset['lat'].values

# with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/SDT2/" + method + "/SC_season" + season + "_disk2_zetath4_wth6.nc") as dset:
#     counts2 = dset['frequency_map']

# with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/SDT2/" + method + "/SC_season" + season + "_disk2_zetath5_wth5.nc") as dset:
#     counts3 = dset['frequency_map']

# plot_fmap(lons, lats, counts1, typ, 2, season=True, year=season, zoom=False, save=False, addname="SDT1")
# plot_fmap(lons, lats, counts2, typ, 2, season=True, year=season, zoom=False, save=True, addname="SDT2_4_6")
# plot_fmap(lons, lats, counts3, typ, 2, season=True, year=season, zoom=False, save=True, addname="SDT2_5_5")
# plot_fmap(lons, lats, counts1, typ, 2, season=True, year=season, zoom=True, save=False, addname="SDT1")
# plot_fmap(lons, lats, counts2, typ, 2, season=True, year=season, zoom=True, save=True, addname="SDT2_4_6")
# plot_fmap(lons, lats, counts3, typ, 2, season=True, year=season, zoom=True, save=True, addname="SDT2_5_5")

# for season in years: 
#     with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/SDT2/" + climate + "_climate/" + method + "/SC_season" + season + "_disk2_zetath5_wth5.nc") as dset:
#         counts_SC = dset['frequency_map']
#         lons = dset['lon'].values
#         lats = dset['lat'].values
#     plot_fmap(lons, lats, counts_SC, typ, 2, season=True, year=season, zoom=False, save=True, addname="5_5")
#     plot_fmap(lons, lats, counts_SC, typ, 2, season=True, year=season, zoom=True, save=True, addname="5_5")

#==================================================================================================================================================
# decadal frequency map from stored data

# years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
# years = ['2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095']
# years = ["2019", "2020", "2021"]
# method = "model_tracks"
# #iuhpts = [85, 95, 105]
#climate = "current"
#typ = "supercell"

# mesocyclone map
# for i, year in enumerate(years):
#     if i == 0:
#         with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/SDT2/" + climate + "_climate/" + method + "/meso_season" + year + ".nc") as dset:
#             counts_meso = dset['frequency_map']
#             lons = dset['lon'].values
#             lats = dset['lat'].values
#     else:
#         with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/SDT2/" + climate + "_climate/" + method + "/meso_season" + year + ".nc") as dset:
#             counts = dset['frequency_map']
#         counts_meso += counts

# plot_fmap(lons, lats, counts_meso/len(years), "mesocyclone", r_disk=0, zoom=False, save=True, addname="")
# plot_fmap(lons, lats, counts_meso/len(years), "mesocyclone", r_disk=0, zoom=True, save=True, addname="")

# supercell map

# for i, season in enumerate(years):
#     fname = "/scratch/snx3000/mblanc/fmaps_data/SDT2/" + climate + "_climate/" + method + "/SC_season" + season + "_disk" + str(r_disk) + ".nc"
#     if i == 0:
#         with xr.open_dataset(fname) as dset:
#             counts_SC = dset['frequency_map']
#             lons = dset['lon'].values
#             lats = dset['lat'].values
#     else:
#         with xr.open_dataset(fname) as dset:
#             counts = dset['frequency_map']
#         counts_SC += counts

# plot_fmap(lons, lats, counts_SC/len(years), typ, r_disk, climate, zoom=False, save=True, addname="")
# plot_fmap(lons, lats, counts_SC/len(years), typ, r_disk, climate, zoom=True, save=True, addname="")

# for iuhpt in iuhpts:
#     for i, year in enumerate(years):
#         fname = "/scratch/snx3000/mblanc/fmaps_data/" + method + "/SC_season" + year + "_filtered_conv_iuhpt" + str(iuhpt) + ".nc"
#         if i == 0:
#             with xr.open_dataset(fname) as dset:
#                 counts_SC = dset['frequency_map']
#         else:
#             with xr.open_dataset(fname) as dset:
#                 counts = dset['frequency_map']
#             counts_SC += counts
    
#     plot_fmap(rlons, rlats, counts_SC/len(years), "supercell", filtering=True, zoom=False, save=True, iuh_thresh=iuhpt, maxval=maxval)
#     plot_fmap(rlons, rlats, counts_SC/len(years), "supercell", filtering=True, zoom=True, save=True, iuh_thresh=iuhpt, maxval=maxval)

#==================================================================================================================================================
# seasonal rain frequency map

# season = "2021"
# path2 = "/scratch/snx3000/mblanc/CT2/current_climate/outfiles/"
# path1 = "/scratch/snx3000/mblanc/CT1_output/"
# typ = "rain"
# filename2 = "/scratch/snx3000/mblanc/fmaps_data/model_tracks/rain_season" + season + "_conv_CT2.nc"
# filename1 = "/scratch/snx3000/mblanc/fmaps_data/model_tracks/rain_season" + season + "_conv_CT1.nc"


# lons, lats, counts2 = seasonal_rain_tracks_model_fmap(season, path2)
# _, _, counts1 = seasonal_rain_tracks_model_fmap(season, path1)
# plot_fmap(lons, lats, counts2, typ, season=True, year=season, zoom=True, filtering=False, conv=True, save=True, iuh_thresh=None, addname="CT2")
# plot_fmap(lons, lats, counts2, typ, season=True, year=season, zoom=False, filtering=False, conv=True, save=True, iuh_thresh=None, addname="CT2")
# plot_fmap(lons, lats, counts1, typ, season=True, year=season, zoom=True, filtering=False, conv=True, save=True, iuh_thresh=None, addname="CT1")
# plot_fmap(lons, lats, counts1, typ, season=True, year=season, zoom=False, filtering=False, conv=True, save=True, iuh_thresh=None, addname="CT1")
# write_to_netcdf(lons, lats, counts2, filename2)
# write_to_netcdf(lons, lats, counts1, filename1)

#==================================================================================================================================================
# delta maps

# plot_supercell_tracks_model_delta_map(r_disk=2, perc=True, zoom=False, save=True, addname="")
# plot_supercell_tracks_model_delta_map(r_disk=2, perc=False, zoom=False, save=True, addname="")

# plot_supercell_tracks_model_delta_map(r_disk=5, perc=True, zoom=False, save=True, addname="")
# plot_supercell_tracks_model_delta_map(r_disk=5, perc=False, zoom=False, save=True, addname="")

#==================================================================================================================================================
# seasonal maps

# parser = argparse.ArgumentParser()
# parser.add_argument("start_day", type=str)
# parser.add_argument("end_day", type=str)
# parser.add_argument("climate", type=str)
# parser.add_argument("r_disk", type=int)
# args = parser.parse_args()

# start_day = args.start_day
# end_day = args.end_day
# climate = args.climate
# r_disk = args.r_disk
# path = "/scratch/snx3000/mblanc/SDT/SDT2_output/" + climate + "_climate/domain/XPT_1MD_zetath5_wth5/"

# start_day = "0601"
# end_day = "0731"
# climate = "current"
# r_disk = 5
# path = "/scratch/snx3000/mblanc/SDT/SDT2_output/" + climate + "_climate/domain/XPT_1MD_zetath5_wth5/"

# plot_supercell_seasonal_fmap(start_day, end_day, path, r_disk, climate, zoom=False, save=True, addname="")
