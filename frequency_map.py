# this script aims at plotting the one-season distribution map of modelled supercells
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#==================================================================================================================================================
# FUNCTIONS
#==================================================================================================================================================

def plot_fmap(lons, lats, fmap, typ, season=False, year=None, z=0):
    """
    Plots the desired decadal (default) or seasonal frequency map
    
    Parameters
    ----------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    fmap : 2D array
        the frequency / distribution map to be plotted
    typ: str
        type of the frequency map, eg "supercell", "rain", "mesocyclone"
    season : bool
        True for a seasonal frequency map, False for the decadal period. The default is False.
    year : str or None
        if seasonal, year YYYY of the season. Elsewise, None. The default is None.
    z : int, optional
        0: no zoom (whole domain) ; 500: zoom over France and Switzerland. The default is 0.

    Returns
    -------
    Plots the desired frequency map and saves it.
    """
    # bleach the background -> set 0 counts to nans
    fmap = fmap.astype(float)
    fmap[fmap==0] = np.nan
    
    # load geographic features
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    #ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', scale=resol, facecolor='none')
    
    if season:

        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
        #ax.add_feature(ocean, linewidth=0.2)
        ax.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.4)
        if z:
            cont = plt.pcolormesh(lons[z:-z,z:-z], lats[z:-z,z:-z], fmap[z:-z,z:-z], cmap="Reds", transform=ccrs.PlateCarree())
            figname = typ + "_season" + year + "_zoom.png"
        else:
            cont = plt.pcolormesh(lons, lats, fmap, cmap="Reds", transform=ccrs.PlateCarree())
            figname = typ + "_season" + year + ".png"
        plt.colorbar(cont, orientation='horizontal', label="number of " + typ + " mask counts")
        plt.title("Season " + year + " " + typ + " distribution map")
        fig.savefig(figname, dpi=300)
        
    else:
        
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
        #ax.add_feature(ocean, linewidth=0.2)
        ax.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.4)
        if z:
            cont = plt.pcolormesh(lons[z:-z,z:-z], lats[z:-z,z:-z], fmap[z:-z,z:-z], cmap="Reds", transform=ccrs.PlateCarree())
            figname = typ + "_decadal" + "_zoom.png"
        else:
            cont = plt.pcolormesh(lons, lats, fmap, cmap="Reds", transform=ccrs.PlateCarree())
            figname = typ + "_decadal" + ".png"
        plt.colorbar(cont, orientation='horizontal', label="number of " + typ + " mask counts")
        plt.title("Decadal " + typ + " distribution map")
        fig.savefig(figname, dpi=300)
    
    return


def resolve_overlaps(rain_masks):
    """
    resolves overlaps for frequency map: for every grid point, discards the additional counts of the rain cells overlapping several times
    with themselves, so that a single, overlapping raintrack only counts as 1

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


def seasonal_masks_fmap(season, typ, climate, resolve_ovl=False, skipped_days=None):
    """
    Compute the seasonal frequency map over whole domain
    
    Parameters
    ----------
    season : str
        considered season, YYYY
    typ : str
        type of the frequency map, eg "supercell", "rain", "mesocyclone"
    climate : str
        "current" or "future", depending on the climate you are analising
    resolve_ovl : bool
        for "rain" and "supercell" types, discards single cell overlaps
    skipped_days : list of str
        list of missing days which consequently must be skipped

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
    
    # remove skipped days from daylist
    if skipped_days:
        skipped_days = pd.to_datetime(skipped_days, format="%Y%m%d")
        daylist = [day for day in daylist if day not in skipped_days]
    
    if typ == "rain" or typ == "supercell":
        
        SC_path = "/scratch/snx3000/mblanc/SDT_output/seasons/" + season + "/"  #supercell_20210617.json
        mask_path = "/scratch/snx3000/mblanc/cell_tracker_output/" + climate + "_climate/" #cell_masks_20210920.nc

        SC_files = []
        mask_files = []
        for day in daylist:
            SC_files.append(SC_path + "supercell_" + day.strftime("%Y%m%d") + ".json")
            mask_files.append(mask_path + "cell_masks_" + day.strftime("%Y%m%d") + ".nc")

        for i in range(len(SC_files)):
            
            # rain masks
            with xr.open_dataset(mask_files[i]) as dset:
                rain_masks = dset['cell_mask'] # 3D matrix
                if i == 0:
                    lons = dset["lon"].values
                    lats = dset["lat"].values
             
            if typ == "supercell":
                # SC info -> extract cell ids corresponding to supercells
                with open(SC_files[i], "r") as read_file:
                    SC_info = json.load(read_file)['supercell_data']
                SC_ids = [SC_info[j]['rain_cell_id'] for j in range(len(SC_info))]
                rain_masks = np.where(np.isin(rain_masks, SC_ids), rain_masks, np.nan)
            
            if resolve_ovl:
                if i == 0:
                    counts = resolve_overlaps(rain_masks)
                else:
                    counts += resolve_overlaps(rain_masks)
            else:
                bin_masks = np.logical_not(np.isnan(rain_masks))
                if i == 0:
                    counts = np.count_nonzero(bin_masks, axis=0)
                else:
                    counts += np.count_nonzero(bin_masks, axis=0)
            
                
    elif typ == "mesocyclone":
        
        meso_path = "/scratch/snx3000/mblanc/SDT_output/seasons/" + season + "/"  # meso_masks_20170801.nc
        meso_masks_files = []
        for day in daylist:
            meso_masks_files.append(meso_path + "meso_masks_" + day.strftime("%Y%m%d") + ".nc")
        
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
    
    else:
        print("error: type is not part of the three compatible ones: supercell, rain or mesocyclone")
        return
    
    return lons, lats, counts


def decadal_masks_fmap(typ, climate, resolve_ovl=False, skipped_days=None):
    """
    Compute the decadal frequency map over whole domain
    
    Parameters
    ----------
    typ: str
        type of the frequency map, eg "supercell", "rain", "mesocyclone"
    climate : str
        "current" or "future", depending on the climate you are analising

    Returns
    -------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    counts : 2D array
        the desired seasonal frequency map
    """
    if climate == "current":
        years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    elif climate == "future":
        years = ["2085", "2086", "2087", "2088", "2089", "2090", "2091", "2092", "2093", "2094", "2095"]
    else:
        print("error: climate wrongly specified !")
    
    for i, year in enumerate(years):
        if i == 0:
            lons, lats, counts = seasonal_masks_fmap(year, typ, climate, resolve_ovl, skipped_days)
        else:
            _, _, counts2 = seasonal_masks_fmap(year, typ, climate, resolve_ovl, skipped_days)
            counts += counts2
    
    return lons, lats, counts


#==================================================================================================================================================
# MAIN
#==================================================================================================================================================

skipped_days = ['20120604', '20140923', '20150725', '20160927', '20170725']
climate = "current"
#typ = "mesocyclone"
season = "2020"
resolve_ovl = True #only for rain and supercell types

lons, lats, counts_meso = seasonal_masks_fmap(season, "mesocyclone", climate, skipped_days=skipped_days)
plot_fmap(lons, lats, counts_meso, "mesocyclone", season=True, year=season, z=0)
plot_fmap(lons, lats, counts_meso, "mesocyclone", season=True, year=season, z=500)

# _, _, counts_SC = seasonal_masks_fmap(season, "supercell", climate, resolve_ovl, skipped_days=skipped_days)
# plot_fmap(lons, lats, counts_SC, "supercell", season=True, year=season, z=0)
# plot_fmap(lons, lats, counts_SC, "supercell", season=True, year=season, z=500)

# lons, lats, counts_rain = seasonal_masks_fmap(season, "rain", climate, resolve_ovl, skipped_days=skipped_days)
# plot_fmap(lons, lats, counts_rain, "rain", season=True, year=season, z=0)
# plot_fmap(lons, lats, counts_rain, "rain", season=True, year=season, z=500)
