# this script aims at plotting supercells trajectories from both observational and model data, on a case study
# it is based on the preliminary script trajectories.py from first_visu

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#=====================================================================================================================================================
# FUNCTIONS
#=====================================================================================================================================================

def plot_tracks(day, cut, obs_ids, usecols, save=True):
    """
    plot the tracks of the modeled and observed supercells occuring the given day

    Parameters
    ----------
    day : str
        day of the case study, YYYYMMDD
    cut : str
        type of the cut, "largecut" or "swisscut"
    obs_ids : list of int
        list of the selected supercells IDs from the observational dataset
    usecols : list of str
        parameters/variables meant to be extracted from the observational dataset

    Returns
    -------
    plots the desired tracks comparison on two different subplots and saves the figure
    """
    
    ## modeled supercells ##
    # SC info
    with open("/scratch/snx3000/mblanc/SDT_output/CaseStudies/supercell_" + day + ".json", "r") as read_file:
        SC_info = json.load(read_file)['supercell_data']
    mod_ids = [SC_info[i]['rain_cell_id'] for i in range(len(SC_info))]

    # rain tracks
    with open("/scratch/snx3000/mblanc/cell_tracker/CaseStudies/outfiles/cell_tracks_" + day + ".json", "r") as read_file:
        rain_tracks = json.load(read_file)['cell_data']
    ncells = len(rain_tracks)

    # restrict rain tracks to the supercells
    rain_tracks = np.array(rain_tracks)
    mod_tracks = rain_tracks[np.isin(np.arange(ncells), mod_ids)]
    
    ## observed supercells ##
    # open full observational dataset
    fullset = pd.read_csv("/scratch/snx3000/mblanc/observations/Full_dataset_thunderstorm_types.csv", sep=';', usecols=usecols)
    fullset['time'] = pd.to_datetime(fullset['time'], format="%Y%m%d%H%M")
    fullset['ID'] = round(fullset['ID'])

    # selection based on given observational IDs
    obs_ids_disp = [j%10000 for j in obs_ids]
    selection = fullset[np.isin(fullset['ID'], obs_ids)]
    selection['ID'] = round(selection['ID'] % 10000) #discard the date and time info, keep only the raw ID

    ## plot the trajectories ##
    # load geographic features
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    #ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', scale=resol, facecolor='none')

    # figure
    fig = plt.figure()#figsize=(6, 8))
    fig.suptitle(pd.to_datetime(day,format='%Y%m%d').strftime('%d/%m/%Y'))

    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    #ax1.add_feature(ocean, linewidth=0.2)
    ax1.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.3)
    for i in range(len(mod_tracks)):
        ax1.plot(mod_tracks[i]['lon'], mod_tracks[i]['lat'], 'x--', linewidth=1, markersize=3, label=str(mod_tracks[i]['cell_id']), transform=ccrs.PlateCarree())
    ax1.legend(loc='lower right', fontsize='8')
    ax1.title.set_text("modeled supercells")

    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    #ax2.add_feature(ocean, linewidth=0.2)
    ax2.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.3)
    for ID in obs_ids_disp:
        ax2.plot(selection['lon'][selection['ID']==ID], selection['lat'][selection['ID']==ID], 'x--', linewidth=1, markersize=3, label=str(ID), transform=ccrs.PlateCarree())
    ax2.legend(loc='upper left', fontsize='7')
    ax2.title.set_text("observed supercells")
    
    if save:
        fig.savefig(day + "_tracks_comp.png", dpi=300)

#=====================================================================================================================================================
# MAIN
#=====================================================================================================================================================

usecols = ['ID', 'time', 'mesostorm', 'mesohailstorm', 'lon', 'lat', 'area', 'vel_x', 'vel_y', 'altitude', 'slope', 'max_CPC', 'mean_CPC', 'max_MESHS',
           'mean_MESHS', 'p_radar', 'p_dz', 'p_z_0', 'p_z_100', 'p_v_mean', 'p_d_mean', 'n_radar', 'n_dz', 'n_z_0', 'n_z_100', 'n_v_mean', 'n_d_mean']

# data to be filled
day = "20210620"
cut = "swisscut"
obs_ids = [2021062010000041, 2021062011300077, 2021062012200142, 2021062014350190, 2021062015400153, 2021062013100019]

#obs_ids = [2017080121350019, 2017080118100168, 2017080117350013, 2017080114050065, 2017080116500049, 2017080113400043, 2017080116050003, 2017080114400086, 2017080114500105, 2017080114100055]

plot_tracks(day, cut, obs_ids, usecols, False)
