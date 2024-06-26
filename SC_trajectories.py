# this script plota supercells trajectories from both observational and model data, on a case study
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

def find_obs_supercells(day, usecols):
    """
    searches in the observational dataset for supercells (mesostorms) on a given day and returns their IDs

    Parameters
    ----------
    day : str
        day of the case study, YYYYMMDD
    usecols : list of str
        parameters/variables meant to be extracted from the observational dataset

    Returns
    -------
    list of the supercells IDs along with a list of their respective lifetime (in min)
    """
    
    # open full observational dataset
    fullset = pd.read_csv("/scratch/snx3000/mblanc/observations/Full_dataset_thunderstorm_types.csv", sep=';', usecols=usecols)
    fullset['time'] = pd.to_datetime(fullset['time'], format="%Y%m%d%H%M")
    fullset = fullset.reindex(columns=usecols)
    strdays = [dt.strftime("%Y%m%d") for dt in fullset['time']]
    fullset['ID'] = round(fullset['ID'])

    #selection based on the given day
    selection = fullset[np.isin(strdays, day)]
    #selection['ID'] = selection['ID'] % 10000
    SC_ids = np.unique(selection['ID'][selection['mesostorm']==1])
    SC_ids = [int(x) for x in SC_ids]
    
    #computing supercells lifetime
    SC_lifetimes = []
    for ID in SC_ids:
        datetimes = np.array(selection['time'][selection['ID']==ID])
        lft = (datetimes[-1]-datetimes[0])/np.timedelta64(1, "m")
        SC_lifetimes.append(int(lft))
    
    return SC_ids, SC_lifetimes


def plot_SDT1_obs_tracks(daystr, obs_ids, usecols, save=False, filter_mod=True):
    """
    plot the tracks of SDT1 modelled against observed supercells (meso detections indicated with crosses) occuring the given day
    option to filter the modelled tracks, ie remove supercell locations where the rain cell max rain rate does not reach the thrsh + prom = 13.7 mm/h criterion
    tailor the plot diplay to the needs

    Parameters
    ----------
    daystr : str
        day of the case study, YYYYmmdd
    obs_ids : list of int
        list of the selected supercells IDs from the observational dataset
    usecols : list of str
        parameters/variables meant to be extracted from the observational dataset
    save : bool
        saves the figure
    filter_mod : bool
        filter the modelled tracks according to the threshold + prominence = 13.7 mm/h criterion

    Returns
    -------
    plots the desired tracks comparison on two different subplots and saves the figure
    """
    
    ## modelled supercells ##
    # SC info : contains coords of tracks as well as coords of meso-detections
    with open("/scratch/snx3000/mblanc/SDT/SDT1_output/CaseStudies/supercell_" + daystr + ".json", "r") as read_file:
        SC_info = json.load(read_file)['supercell_data']
    
    # filtering
    if filter_mod:
        for SC in SC_info:
            indices = np.array(SC['cell_max_rain']) >= 13.7 # select the indices of the track above the thr+prom criterion
            SC['cell_lon'] = np.array(SC['cell_lon'])[indices] # reduce the coordinates to the locations where the thr+prom criterion holds
            SC['cell_lat'] = np.array(SC['cell_lat'])[indices]
        
    ## observed supercells ##
    # open full observational dataset
    fullset = pd.read_csv("/scratch/snx3000/mblanc/observations/Full_dataset_thunderstorm_types.csv", sep=';', usecols=usecols)
    #fullset['time'] = pd.to_datetime(fullset['time'], format="%Y%m%d%H%M")
    fullset['ID'] = round(fullset['ID'])

    # selection based on given observational IDs
    #obs_ids_disp = [j%10000 for j in obs_ids]
    selection = fullset[np.isin(fullset['ID'], obs_ids)]

    ## plot the trajectories ##
    # load geographic features
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', scale=resol, facecolor='none')

    # figure
    fig = plt.figure()#figsize=(6, 8))
    fig.suptitle(pd.to_datetime(daystr,format='%Y%m%d').strftime('%d/%m/%Y'))

    ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
    ax1.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    ax1.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.3)
    cmap = plt.cm.tab10.colors
    for i in range(len(SC_info)):
        col = cmap[i]
        ax1.plot(SC_info[i]['cell_lon'], SC_info[i]['cell_lat'], '--', linewidth=0.7, color=col, transform=ccrs.PlateCarree())
        ax1.plot(SC_info[i]['meso_lon'], SC_info[i]['meso_lat'], 'x', markersize=2, color=col, transform=ccrs.PlateCarree())
    ax1.title.set_text("modelled supercells")

    ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree(), sharex=ax1, sharey=ax1)
    ax2.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    ax2.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.3)
    cmap = plt.cm.tab20.colors
    for i, ID in enumerate(obs_ids):
        col = cmap[i]
        ax2.plot(selection['lon'][selection['ID']==ID], selection['lat'][selection['ID']==ID], '--', linewidth=0.7, color=col, transform=ccrs.PlateCarree())
    ax2.title.set_text("observed supercells")
    
    if save:
        fig.savefig(daystr + "_filtered_tracks_comp.png", dpi=300)


def plot_obs_tracks(daystr, obs_ids, usecols, save=False):
    """
    plot the tracks observed supercells occuring the given day
    tailor the plot diplay to the needs

    Parameters
    ----------
    daystr : str
        day of the case study, YYYYmmdd
    obs_ids : list of int
        list of the selected supercells IDs from the observational dataset
    usecols : list of str
        parameters/variables meant to be extracted from the observational dataset
    save : bool
        option to save the figure

    Returns
    -------
    plots the desired observational tracks and saves the figure if requested
    """
        
    ## observed supercells ##
    # open full observational dataset
    fullset = pd.read_csv("/scratch/snx3000/mblanc/observations/Full_dataset_thunderstorm_types.csv", sep=';', usecols=usecols)
    #fullset['time'] = pd.to_datetime(fullset['time'], format="%Y%m%d%H%M")
    fullset['ID'] = round(fullset['ID'])

    # selection based on given observational IDs
    #obs_ids_disp = [j%10000 for j in obs_ids]
    selection = fullset[np.isin(fullset['ID'], obs_ids)]

    ## plot the trajectories ##
    # load geographic features
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', scale=resol, facecolor='none')
    #rp = ccrs.RotatedPole(pole_longitude = -170, pole_latitude = 43)

    # figure
    fig = plt.figure()#figsize=(5,8))
    fig.suptitle(pd.to_datetime(daystr,format='%Y%m%d').strftime('%d/%m/%Y'))
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    ax.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.2)
    
    cmap = plt.cm.tab20.colors
    for i, ID in enumerate(obs_ids):
        col = cmap[i]
        ax.plot(selection['lon'][selection['ID']==ID], selection['lat'][selection['ID']==ID], '--', linewidth=0.7, color=col, transform=ccrs.PlateCarree())
    ax.title.set_text("observed supercells")
    
    if save:
        fig.savefig(daystr + "_obs.png", dpi=300)


def plot_SDT2_tracks(daystr, version, zeta_th=None, w_th=None, save=False):
    """
    plots the tracks of SDT2 modelled supercells occuring during the given case study
    meso detections indicated with crosses
    tailor the plot diplay to the needs

    Parameters
    ----------
    daystr : str
        day of the case study, YYYYmmdd
    version : str
        version of SDT2, ie "XPT_1MD", "XPT_2MD", "PT_1MD", "PT_2MD"
    zeta_th : float
        if version == "XPT_1MD", specify vorticity threshold
    w_th : float
        if version == "XPT_1MD", specify updraught velocity threshold
    save : bool
        option to save the figure

    Returns
    -------
    plots the desired modelled tracks on a single plot and saves the figure if requested
    """
    
    if version == "XPT_1MD":
        zeta_th = str(zeta_th)
        w_th = str(w_th)
        tracks_name = "/scratch/snx3000/mblanc/SDT/SDT2_output/current_climate/CaseStudies/XPT_1MD/supercell_"+"zetath"+zeta_th+"_wth"+w_th+"_"+daystr+".json"
    else:
        tracks_name = "/scratch/snx3000/mblanc/SDT/SDT2_output/current_climate/CaseStudies/" + version + "/supercell_" + daystr + ".json"
    
    with open(tracks_name, "r") as read_file:
        SC_info = json.load(read_file)['supercell_data']

    ## plot the trajectories ##
    # load geographic features
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', scale=resol, facecolor='none')

    # figure
    fig = plt.figure()#figsize=(6, 8))
    fig.suptitle(pd.to_datetime(daystr,format='%Y%m%d').strftime('%d/%m/%Y'))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    ax.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.2)

    cmap = plt.cm.tab20.colors
    cmap = [cmap[i%20] for i in range(20*int(np.ceil(len(SC_info)/20)))]
    for i in range(len(SC_info)):
        col = cmap[i]
        ax.plot(SC_info[i]['cell_lon'], SC_info[i]['cell_lat'], '--', linewidth=0.7, color=col, transform=ccrs.PlateCarree())
        ax.plot(SC_info[i]['meso_lon'], SC_info[i]['meso_lat'], 'x', markersize=2, color=col, transform=ccrs.PlateCarree())
    
    if version == "XPT_1MD":
        title = r"modelled supercells; XPT_1MD; $\zeta_{th}=$"+zeta_th+", $w_{th}=$"+w_th
        figname = daystr + "_SDT2_zetath" + zeta_th + "_wth" + w_th + ".png"
    elif version == "XPT_2MD":
        title = r"modelled supercells; XPT_2MD; $\zeta_{th}=0.004$, $w_{th}=6$"
        figname = daystr + "_SDT2.png"
    else:
        title = r"modelled supercells; " + version + "; $\zeta_{th}=0.004$, $w_{th}=6$, $\zeta_{pk}=0.006$"
        figname = daystr + "_SDT2.png"
    
    ax.title.set_text(title)
    
    if save:
        fig.savefig(figname, dpi=300)

#=====================================================================================================================================================
# MAIN
#=====================================================================================================================================================

# usecols = ['ID', 'time', 'mesostorm', 'mesohailstorm', 'lon', 'lat', 'area', 'vel_x', 'vel_y', 'altitude', 'slope', 'max_CPC', 'mean_CPC', 'max_MESHS', 'mean_MESHS',
#            'p_radar', 'p_x', 'p_y', 'p_dz', 'p_z_0', 'p_z_100', 'p_v_mean', 'p_d_mean', 'n_radar', 'n_x', 'n_y', 'n_dz', 'n_z_0', 'n_z_100', 'n_v_mean', 'n_d_mean']
#usecols = ['ID', 'time', 'mesostorm', 'mesohailstorm', 'lon', 'lat']
version = "XPT_1MD"

# search for supercells occuring on the case study day
#ids, dts = find_obs_supercells(day, usecols)

# data to be filled
daysstr = ['20170801', '20190610', '20190611', '20190614', '20190820', '20210620', '20210628', '20210708', '20210712', '20210713']

obs_ids_CSs = [[2017080121350019, 2017080118100168, 2017080117350013, 2017080114050065, 2017080116500049, 2017080113400043, 2017080116050003, 2017080114400086, 2017080114500105, 2017080114100055], 
               [2019061008000081, 2019061020000016, 2019061007450069, 2019061021100019, 2019061018050015, 2019061018500085, 2019061020500017, 2019061023150106, 2019061023450003, 2019061013450052, 2019061021350004],
               [2019061114300008, 2019061122050060, 2019061119050038, 2019061118550036, 2019061113400021, 2019061200250058, 2019061202350071],
               [2019061423000095, 2019061420250066, 2019061421050077],
               [2019082014000037],
               [2021062010000041, 2021062011300077, 2021062012200142, 2021062012350124, 2021062013450008, 2021062014350190, 2021062014550008, 2021062016450008, 2021062017250017, 2021062019300063, 2021062015400153, 2021062013100019],
               [2021062822050041, 2021062821000019, 2021062815300020, 2021062812050049, 2021062812300057, 2021062812350056, 2021062813250074],
               [2021070812450046, 2021070814150283, 2021070811350222, 2021070812300143, 2021070806550029, 2021070805050110, 2021070805150110, 2021070804000089, 2021070805050002, 2021070808050060, 2021070808550012, 2021070811550216, 2021070812450013, 2021070812500143, 2021070813450281, 2021070814100012],
               [2021071216000078, 2021071218100030, 2021071221500035, 2021071303550123, 2021071303500063],
               [2021071305200023, 2021071316500115, 2021071313500163, 2021071308450138, 2021071308250110, 2021071306450059, 2021071305400035, 2021071315500027, 2021071314100163, 2021071314100031, 2021071313100125, 2021071312400110, 2021071304100142]]

# observed supercells classified by days according to their onset time and 04-04 rule
#obs_ids = [2017080121350019, 2017080118100168, 2017080117350013, 2017080114050065, 2017080116500049, 2017080113400043, 2017080116050003, 2017080114400086, 2017080114500105, 2017080114100055]
#obs_ids = [2019061008000081, 2019061020000016, 2019061007450069, 2019061021100019, 2019061018050015, 2019061018500085, 2019061020500017, 2019061023150106, 2019061023450003, 2019061013450052, 2019061021350004]
#obs_ids = [2019061114300008, 2019061122050060, 2019061119050038, 2019061118550036, 2019061113400021, 2019061200250058, 2019061202350071]
#obs_ids = [2019061423000095, 2019061420250066, 2019061421050077]
#obs_ids = [2019082014000037]
#obs_ids = [2021062010000041, 2021062011300077, 2021062012200142, 2021062012350124, 2021062013450008, 2021062014350190, 2021062014550008, 2021062016450008, 2021062017250017, 2021062019300063, 2021062015400153, 2021062013100019]
#obs_ids = [2021062822050041, 2021062821000019, 2021062815300020, 2021062812050049, 2021062812300057, 2021062812350056, 2021062813250074]
#obs_ids = [2021070812450046, 2021070814150283, 2021070811350222, 2021070812300143, 2021070806550029, 2021070805050110, 2021070805150110, 2021070804000089, 2021070805050002, 2021070808050060, 2021070808550012, 2021070811550216, 2021070812450013, 2021070812500143, 2021070813450281, 2021070814100012]
#obs_ids = [2021071216000078, 2021071218100030, 2021071221500035, 2021071303550123, 2021071303500063]
#obs_ids = [2021071305200023, 2021071316500115, 2021071313500163, 2021071308450138, 2021071308250110, 2021071306450059, 2021071305400035, 2021071315500027, 2021071314100163, 2021071314100031, 2021071313100125, 2021071312400110, 2021071304100142]

zeta_th = 5e-3
w_th = 6

for daystr in daysstr:
    plot_SDT2_tracks(daystr, version, zeta_th, w_th, save=True)
