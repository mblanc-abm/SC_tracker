# this script aims at plotting the one-season distribution map of modelled supercells
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.signal import convolve2d
from skimage.morphology import disk, dilation
import argparse
from matplotlib.colors import TwoSlopeNorm
from matplotlib import gridspec
#from sklearn.neighbors import KernelDensity

#==================================================================================================================================================
# FUNCTIONS
#==================================================================================================================================================
def plot_fmap(lons, lats, fmap, typ, season=False, year=None, zoom=False, filtering=False, conv=True, save=False, iuh_thresh=None):
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
    season : bool
        True for a seasonal frequency map, False for the decadal period. The default is False.
    year : str or None
        if seasonal, year YYYY of the season. Elsewise, None. The default is None.
    zoom : bool, optional
        False: smart cut of whole domain (ocean and Northern Africa discarded) ; True: zoom on the Alps. The default is False.
    filtering : bool
        for "supercell" type, filters out the mask patches associated with the cells whose max rain rate does not reach the thr+prom = 13.7 mm/h criterion
    conv : bool
        False: number of storms per grid box, ie per 4.84 km^2; True: 4-connectivity sum convolution yields the number of storms per 24.2 km^2
        caution: handling of dilation/convolution is performed beforehand, not in this plotting function
    iuh_thresh: float
        option to filter out supercell tracks whose mesocyclone peak intensity is weaker than iuh_thresh; default if None, equivalent to 75 m^2/s^2
        displayed in the figure title and name

    Returns
    -------
    Plots the desired frequency map and saves it.
    """
        
    # load geographic features
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', scale=resol, facecolor='none')
    
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    ax.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.2)
    
    if season:

        if zoom:
            zl, zr, zb, zt = 750, 400, 570, 700 #cut for Alpine region
            cont = plt.pcolormesh(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], fmap[zb:-zt,zl:-zr], cmap="Reds", transform=ccrs.PlateCarree())
            figname = "season" + year + "_" + typ + "_alps"
        else:
            zl, zr, zb, zt = 180, 25, 195, 25 #smart cut for entire domain
            cont = plt.pcolormesh(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], fmap[zb:-zt,zl:-zr], cmap="Reds", transform=ccrs.PlateCarree())
            figname = "season" + year + "_" + typ
        
        if filtering:
            figname += "_filtered"
        if conv:
            figname += "_conv"
        if iuh_thresh:
            figname += "_iuhpt" + str(round(iuh_thresh))
        figname += ".png"
        
        if conv:
            plt.colorbar(cont, orientation='horizontal', label=r"number of " + typ + "s per 24.2 $km^2$")
        else:
            plt.colorbar(cont, orientation='horizontal', label=r"number of " + typ + "s per 4.84 $km^2$")
        
        title = "Season " + year + " " + typ + " distribution map"
        if iuh_thresh:
            title += ", iuh thresh = " + str(round(iuh_thresh))
        
    else:
        
        if zoom:
            zl, zr, zb, zt = 750, 400, 570, 700 #cut for Alpine region
            cont = plt.pcolormesh(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], fmap[zb:-zt,zl:-zr], cmap="Reds", transform=ccrs.PlateCarree())
            figname = "decadal_" + typ + "_alps"
        else:
            zl, zr, zb, zt = 180, 25, 195, 25 #smart cut for entire domain
            cont = plt.pcolormesh(lons[zb:-zt,zl:-zr], lats[zb:-zt,zl:-zr], fmap[zb:-zt,zl:-zr], cmap="Reds", transform=ccrs.PlateCarree())
            figname = "decadal_" + typ
        
        if filtering:
            figname += "_filtered"
        if conv:
            figname += "_conv"
        if iuh_thresh:
            figname += "_iuhpt" + str(round(iuh_thresh))
        figname += ".png"
        
        if conv:
            plt.colorbar(cont, orientation='horizontal', label=r"number of " + typ + "s per year per 24.2 $km^2$")
        else:
            plt.colorbar(cont, orientation='horizontal', label=r"number of " + typ + "s per year per 4.84 $km^2$")
        
        title = "Decadal " + typ + " distribution map"
        if iuh_thresh:
            title += ", iuh thresh = " + str(round(iuh_thresh))
        
    plt.title(title)
    
    if save:
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


def seasonal_supercell_tracks_model_fmap(season, filtering=True, skipped_days=None, conv=True, iuh_thresh=None):
    """
    Computes the seasonal supercell frequency map using tracks method over the model whole domain
    Takes care of overlaps by default
    
    Parameters
    ----------
    season : str
        considered season, YYYY
    filtering : bool
        filters out the mask patches associated with the cells whose max rain rate does not reach the thr+prom = 13.7 mm/h criterion
    skipped_days : list of str
        list of missing days which consequently must be skipped
    conv : bool
        False: number of storms per grid box, ie per 4.84 km^2; True: 4-connectivity sum convolution yields the number of storms per 24.2 km^2
    iuh_thresh: float
        option to filter out supercell tracks whose mesocyclone peak intensity is weaker than iuh_thresh; default if None, equivalent to 75 m^2/s^2

    Returns
    -------
    lons : 2D array
        longitude at each gridpoint
    lats : 2D array
        latitude at each gridpoint
    counts : 2D array
        the desired seasonal supercell frequency map
    """
    
    start_day = pd.to_datetime(season + "0401")
    end_day = pd.to_datetime(season + "0930")
    daylist = pd.date_range(start_day, end_day)
    
    # load lons and lats static fields
    with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/model_masks/meso_season2011.nc") as dset:
        lons = dset["lon"].values
        lats = dset["lat"].values
    
    counts = np.zeros_like(lons, dtype=int) # initialise the counts matrix
    fp_ind = np.argwhere(disk(1)) # footprint: 4-connectivity; will serve for marking the supercell core footprint
    
    # remove skipped days from daylist
    if skipped_days:
        skipped_days = pd.to_datetime(skipped_days, format="%Y%m%d")
        daylist = [day for day in daylist if day not in skipped_days]
    
    SC_path = "/scratch/snx3000/mblanc/SDT_output/seasons/" + season + "/"
    SC_files = []
    for day in daylist:
        SC_files.append(SC_path + "supercell_" + day.strftime("%Y%m%d") + ".json")
    
    # loop over the days of the season
    for SC_file in SC_files:
        # initialise an intermediate counts matrix for this day, which will contain its supercells tracks footprint; this incidentally avoids overlaps
        counts_day = np.zeros_like(lons, dtype=int) 
        
        # load the supercells of the given day
        with open(SC_file, "r") as read_file:
            SC_info = json.load(read_file)['supercell_data']
        
        # loop over the supercells of the day
        for SC in SC_info:
            
            # check if the mesocyclone max intensity exceeds iuh_thresh and filter out weak supercells if necessary
            if iuh_thresh:
                SC_max_iuh = np.max(np.abs(SC['max_val']))
                if SC_max_iuh < iuh_thresh:
                    continue
            
            # initialise an intermediate counts matrix for this SC, which will contain its track footprint; this incidentally avoids overlaps
            counts_SC = np.zeros_like(lons, dtype=int)
            
            if filtering:
                indices = np.array(SC['cell_max_rain']) >= 13.7 # select the indices of the track that must be kept
                SC_lons = np.array(SC['cell_lon'])[indices]
                SC_lats = np.array(SC['cell_lat'])[indices]
            else:
                SC_lons = SC['cell_lon']
                SC_lats = SC['cell_lat']
            
            for j in range(len(SC_lons)): # for each SC centre of mass location
                distances = (lons - SC_lons[j])**2 + (lats - SC_lats[j])**2 # interpolate it from the lon-lat coords to the grid indices
                k,l = np.unravel_index(np.argmin(distances), distances.shape) #indices on grid corresponding to the SC centre of mass interpolation
                counts_SC[k-1+fp_ind[:,0], l-1+fp_ind[:,1]] = 1 # the 4-connectivity disk around centre of mass as proxy of SC core
            
            if conv:
                counts_SC = dilation(counts_SC, disk(1)) # 4-connectivity dilation -> avoids SC overlaps with themselves
            
            counts_day += counts_SC
            
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


def seasonal_masks_fmap(season, typ, climate, resolve_ovl=True, filtering=True, skipped_days=None, conv=True):
    """
    Computes the seasonal frequency map using masks method over the model whole domain
    
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
    filtering : bool
        for "supercell" type, filters out the mask patches associated with the cells whose max rain rate does not reach the thr+prom = 13.7 mm/h criterion
    skipped_days : list of str
        list of missing days which consequently must be skipped
    conv : bool, for mesocyclone only (rain cells sufficiently widespread + significant computational cost)
        False: number of storms per grid box, ie per 4.84 km^2; True: 4-connectivity sum convolution yields the number of storms per 24.2 km^2

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
        
        SC_path = "/scratch/snx3000/mblanc/SDT_output/seasons/" + season + "/"
        mask_path = "/scratch/snx3000/mblanc/cell_tracker_output/" + climate + "_climate/"

        SC_files = []
        mask_files = []
        for day in daylist:
            SC_files.append(SC_path + "supercell_" + day.strftime("%Y%m%d") + ".json")
            mask_files.append(mask_path + "cell_masks_" + day.strftime("%Y%m%d") + ".nc")

        for i in range(len(mask_files)):
            
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
                SC_ids = [SC['rain_cell_id'] for SC in SC_info]
                                
                if filtering: # prepare and convert the rain masks datelist to the cells datelist format, ie YYYY-mm-dd HH:mm (remove seconds)
                    rain_masks_times = pd.to_datetime([str(t)[:16] for t in np.array(rain_masks['time'])]) # otherwise may be mismatches                    
                
                rain_masks = np.where(np.isin(rain_masks, SC_ids), rain_masks, np.nan) # discard cells that are not supercells. This line incidentally drops the time coord values
                
                # filter out the mask patches associated with the insufficiently precipitating cells
                if filtering:
                    for SC in SC_info:
                        indices = np.array(SC['cell_max_rain']) < 13.7 # select the indices of the track that must be droped
                        if any(indices): # if supercell SC has at leat one mask to discard throughout its lifetime
                            ID = SC['rain_cell_id']
                            times_to_drop = pd.to_datetime(np.array(SC['cell_datelist'])[indices]) # extract the times at which it occurs
                            # finally discard the masks of the cell in question at the previously selected time slices
                            rain_masks[np.isin(rain_masks_times, times_to_drop)] = np.where(rain_masks[np.isin(rain_masks_times, times_to_drop)]==ID, np.nan, rain_masks[np.isin(rain_masks_times, times_to_drop)])
            
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
        
        meso_path = "/scratch/snx3000/mblanc/SDT_output/seasons/" + season + "/"
        meso_masks_files = []
        for day in daylist:
            meso_masks_files.append(meso_path + "meso_masks_" + day.strftime("%Y%m%d") + ".nc")
        
        for i, meso_masks_file in enumerate(meso_masks_files):
            
            with xr.open_dataset(meso_masks_file) as dset:
                meso_masks = dset['meso_mask']
                if i == 0:
                    lons = dset["lon"].values
                    lats = dset["lat"].values
            
            if conv:
                for j in range(len(meso_masks)):
                    meso_masks[j] = dilation(meso_masks[j], disk(1)) # 4-connectivity dilation -> avoids meso overlaps with themselves
            
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
    Computes the decadal frequency map over whole domain
    
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

# model data ##

parser = argparse.ArgumentParser()
parser.add_argument("iuh_thresh", type=float)
args = parser.parse_args()

skipped_days = ['20120604', '20140923', '20150725', '20160927', '20170725']
climate = "current"
#season = args.season
years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
method = "model_tracks"
iuh_thresh = args.iuh_thresh

# lons, lats, counts_meso = seasonal_masks_fmap(season, "mesocyclone", climate, skipped_days=skipped_days, conv=True)
# plot_fmap(lons, lats, counts_meso, "mesocyclone", season=True, year=season, zoom=False, conv=True, save=True)
# plot_fmap(lons, lats, counts_meso, "mesocyclone", season=True, year=season, zoom=True, conv=True, save=True)
# filename_meso = "/scratch/snx3000/mblanc/fmaps_data/" + method + "/meso_season" + season + "_conv.nc"
# write_to_netcdf(lons, lats, counts_meso, filename_meso)

# lons, lats, counts_SC = seasonal_masks_fmap(season, "supercell", climate, resolve_ovl, filtering=filtering, skipped_days=skipped_days)
# plot_fmap(lons, lats, counts_SC, "supercell", season=True, year=season, filtering=filtering, conv=True)
# plot_fmap(lons, lats, counts_SC, "supercell", season=True, year=season, zoom=True, filtering=filtering, conv=True)
# filename_SC = "/scratch/snx3000/mblanc/fmaps_data/" + method + "/SC_season" + season + "_filtered.nc"
# write_to_netcdf(lons, lats, counts_SC, filename_SC)

for season in years:
    lons, lats, counts_SC = seasonal_supercell_tracks_model_fmap(season, skipped_days=skipped_days, iuh_thresh=iuh_thresh)
    plot_fmap(lons, lats, counts_SC, "supercell", season=True, year=season, zoom=False, filtering=True, conv=True, save=True, iuh_thresh=iuh_thresh)
    plot_fmap(lons, lats, counts_SC, "supercell", season=True, year=season, zoom=True, filtering=True, conv=True, save=True, iuh_thresh=iuh_thresh)
    filename_SC = "/scratch/snx3000/mblanc/fmaps_data/" + method + "/SC_season" + season + "_filtered_conv_iuhpt" + str(round(iuh_thresh)) + ".nc"
    write_to_netcdf(lons, lats, counts_SC, filename_SC)

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
# seasonal map from stored data

#season = "2017"
# years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
# method = "model_masks"

# for season in years:
    
#     with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/" + method + "/meso_season" + season + ".nc") as dset:
#         counts_meso = dset['frequency_map']
#         lons = dset['lon'].values
#         lats = dset['lat'].values
#     plot_fmap(lons, lats, counts_meso, "mesocyclone", season=True, year=season, conv=True)
#     plot_fmap(lons, lats, counts_meso, "mesocyclone", season=True, year=season, zoom=True, conv=True)

#     with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/" + method + "/SC_season" + season + "_filtered.nc") as dset:
#         counts_SC = dset['frequency_map']
#     plot_fmap(lons, lats, counts_SC, "supercell", season=True, year=season, filtering=True, conv=True)
#     plot_fmap(lons, lats, counts_SC, "supercell", season=True, year=season, zoom=True, filtering=True, conv=True)

#==================================================================================================================================================
# decadal frequency map from stored data

# years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
#years = ["2016", "2017", "2018", "2019", "2020", "2021", "2022"]
#method = "obs_tracks"

# # mesocyclone map
# for i, year in enumerate(years):
#     if i == 0:
#         with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/" + method + "/meso_season" + year + ".nc") as dset:
#             counts_meso = dset['frequency_map']
#             lons = dset['lon'].values
#             lats = dset['lat'].values
#     else:
#         with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/" + method + "/meso_season" + year + ".nc") as dset:
#             counts = dset['frequency_map']
#         counts_meso += counts

# plot_fmap(lons, lats, counts_meso/11, "mesocyclone", zoom=False, conv=False, save=True)
# plot_fmap(lons, lats, counts_meso/11, "mesocyclone", zoom=True, conv=False, save=True)

# supercell map 
# for i, year in enumerate(years):
#     if i == 0:
#         with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/" + method + "/Apr-Oct/SC_season" + year + "_conv.nc") as dset:
#             counts_SC = dset['frequency_map']
#             lons = dset['lon'].values
#             lats = dset['lat'].values
#     else:
#         with xr.open_dataset("/scratch/snx3000/mblanc/fmaps_data/" + method + "/Apr-Oct/SC_season" + year + "_conv.nc") as dset:
#             counts = dset['frequency_map']
#         counts_SC += counts

# #plot_fmap(lons, lats, counts_SC/7, "supercell", zoom=False, filtering=True, conv=True, save=True)
# plot_fmap(lons, lats, counts_SC/7, "supercell", zoom=True, filtering=False, conv=True, save=True)