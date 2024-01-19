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
    
    # load geographic features
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
    
    if season:

        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.5)
        ax.add_feature(ocean, linewidth=0.2)
        if z:
            cont = plt.pcolormesh(lons[z:-z,z:-z], lats[z:-z,z:-z], fmap[z:-z,z:-z], cmap="Reds", transform=ccrs.PlateCarree())
            figname = "season" + year + "_zoom.png"
        else:
            cont = plt.pcolormesh(lons, lats, fmap, cmap="Reds", transform=ccrs.PlateCarree())
            figname = "season" + year + ".png"
        plt.colorbar(cont, orientation='horizontal', label="number of " + typ + " mask counts")
        plt.title("Season " + year)
        fig.savefig(figname, dpi=300)
        
    else:
        
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.5)
        ax.add_feature(ocean, linewidth=0.2)
        if z:
            cont = plt.pcolormesh(lons[z:-z,z:-z], lats[z:-z,z:-z], fmap[z:-z,z:-z], cmap="Reds", transform=ccrs.PlateCarree())
            figname = "decadal" + "_zoom.png"
        else:
            cont = plt.pcolormesh(lons, lats, fmap, cmap="Reds", transform=ccrs.PlateCarree())
            figname = "decadal" + ".png"
        plt.colorbar(cont, orientation='horizontal', label="number of " + typ + " mask counts")
        plt.title("Decadal distribution map")
        fig.savefig(figname, dpi=300)
    
    return


# so far designed for supercell and rain types only
def seasonal_masks_fmap(season, typ):
    """
    Compute the seasonal frequency map over whole domain
    
    Parameters
    ----------
    season : str
        considered season, YYYY
    typ : str
        type of the frequency map, eg "supercell", "rain", "mesocyclone"

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
    SC_path = "/scratch/snx3000/mblanc/SDT_output/seasons/2021/"  #supercell_20210617.json
    mask_path = "/project/pr133/mblanc/cell_tracker_output/current_climate/" #cell_masks_20210920.nc

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
                SC_info = json.load(read_file)['SC_data']
            SC_ids = [SC_info[j]['rain_cell_id'] for j in range(len(SC_info))]
            rain_masks = np.where(np.isin(rain_masks, SC_ids), rain_masks, np.nan)
            
        bin_masks = np.logical_not(np.isnan(rain_masks))
        
        if i == 0:
            counts = np.count_nonzero(bin_masks, axis=0)
        else:
            counts += np.count_nonzero(bin_masks, axis=0)
    
    return lons, lats, counts


# so far designed for supercell and rain types only
def decadal_masks_fmap(typ):
    """
    Compute the decadal frequency map over whole domain
    
    Parameters
    ----------
    typ: str
        type of the frequency map, eg "supercell", "rain", "mesocyclone"

    Returns
    -------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    counts : 2D array
        the desired seasonal frequency map
    """
    
    years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    
    for i, year in enumerate(years):
        if i == 0:
            lons, lats, counts = seasonal_masks_fmap(year, typ)
        else:
            _, _, counts2 = seasonal_masks_fmap(year, typ)
            counts += counts2
    
    return lons, lats, counts


#==================================================================================================================================================
# MAIN
#==================================================================================================================================================

typ = "supercell"
season = "2021"
lons, lats, counts = seasonal_masks_fmap(season, typ)

zoom = 0
plot_fmap(lons, lats, counts, typ, season=True, year=season, z=zoom)
