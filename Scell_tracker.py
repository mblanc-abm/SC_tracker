import json
import numpy as np
import pandas as pd
#import math
#from tqdm import tqdm
import xarray as xr
from scipy import ndimage
#from scipy import interpolate
#from scipy.ndimage import rotate
#import itertools
#import collections
#from skimage.feature import peak_local_max
from skimage.segmentation import expand_labels
#from skimage.morphology import disk
import datetime as dt

import os
import sys
sys.path.append("../first_visu")
from CaseStudies import zeta_plev

#=======================================================================================================================================

def track_Scells(day, timesteps, fnames_p, fnames_s, path_h, rain_masks_name, rain_tracks_name, zeta_th, zeta_pk, w_th, min_area,
                 aura, two_meso_detections, CS=False):
    """
    Main function managing mesocyclone detection and meso-rain association.

    Parameters
    ----------
    day: datetime
        considered day
    timesteps: list of datetime objects
        list of hourly datetimes, corresponding to the integer hours of the considered day where the vorticity field is available
    fnames_p : list of str
        complete path and names of the 3D pressure levels files containing the wind fields, for vorticity
        must correspond to timesteps element-wise
    fnames_s : list of str
        complete path and names of the 2D surface pressure files, for vorticity and 10m max wind speed
        must correspond to timesteps element-wise
    path_h : str
        path to the concatenated 5min surface file, for max hail diameter
    rain_masks_name : str
        path to the netcdf rain masks file: labeled rain cells (nan is background, cell labels start at 0), 2D arrays, concatenated
        with a 5 min resolution
    rain_tracks_name : str
        path to the output json file of the cell tracker algorithm 
    zeta_th : float
        vorticity threshold for each pressure level
    w_th : float
        updraught velocity threshold for each pressure level
    min_area : int
        minimum horizontal area threshold for each pressure level in grid point
    aura : int
        number of gridpoints to dilate labels, supercell post-dilation intervenes after min_area criterion is invoked
    CS : bool
        if running on a case study where domain is cropped into a subdomain, hourly 2D surface files contain only surface pressure and
        not 10m max wind speed -> the latter variable will therefore not be considered
    zeta_pk : None or float
        vorticity peak threshold for each pressure level, default is None
    
    Returns
    -------
    supercells: list of SuperCell objects
    na_vorticies: list of NA_Vortex objects (vorticies not assigned to any rain cell)
    lons: longitude at each gridpoint, 2D array
    lats: latitude at each gridpoint, 2D array
    """
    
    # rain masks, 5min timesteps and static fields opening
    with xr.open_dataset(rain_masks_name) as dset:
        rain_masks = dset['cell_mask'] # 3D matrix
        times_5min = pd.to_datetime(dset['time'].dt.round("5min").values) # get rid of timestamp errors
        lons = dset["lon"].values
        lats = dset["lat"].values
    
    # decrease temporal resolution to match hourly one
    rain_masks_hourly = [mask for i, mask in enumerate(rain_masks) if times_5min[i] in timesteps]
    
    # rain tracks opening
    with open(rain_tracks_name, "r") as read_file:
        dset = json.load(read_file)
    rain_tracks = dset['cell_data'] # rain_tracks[j] contains all info about cell j
   
    active_cells = [] #in terms of supercellular activity
    active_cells_ids = [] #and their corresponding ids
    na_vorticies = [] # rotating updraughts which did not overlap with any rain cell
    
    
    # hail file opening; distinguish case study from domain cases
    if CS: # the 5min_2D concatenated files already span the case study time range
        
        fname_h = os.path.join(path_h, "cut_HAILlffd" + day.strftime("%Y%m%d") + ".nc")
        with xr.open_dataset(fname_h, engine="netcdf4") as dset:
            hail_fields = dset["DHAIL_MX"].values # 3D array
            times_hail = pd.to_datetime(dset['time'].dt.round("5min").values)
            
    else: # for domain, need to switch 2D fields and times from 06-06 to 04-04 UTC, so the time steps match those of rain masks. TO BE TESTED !
        
        fname_h = os.path.join(path_h, "lffd" + day.strftime("%Y%m%d") + "_0606.nz")
        with xr.open_dataset(fname_h, engine="netcdf4") as dset:
            hail_fields = dset["DHAIL_MX"][:-24].values # 3D array
            times_hail = pd.to_datetime(dset['time'].dt.round("5min")[:-24].values)
        
        day_before = day - pd.Timedelta(1,"d")
        fname_h = os.path.join(path_h, "lffd" + day_before.strftime("%Y%m%d") + "_0606.nz")
        with xr.open_dataset(fname_h, engine="netcdf4") as dset:
            hail_fields_before = dset["DHAIL_MX"][-24:].values # 3D array
            times_hail_before = pd.to_datetime(dset['time'].dt.round("5min")[-24:].values)
        
        hail_fields = np.concatenate([hail_fields_before, hail_fields])
        times_hail = np.concatenate([times_hail_before, times_hail])
    
    # check whether rain masks time steps are contained in hail fields time steps, crash the algorithm otherwise
    if not np.all(np.isin(times_5min, times_hail)):
        print("rain masks time steps")
        print(times_5min)
        print("hail fields time steps")
        print(times_hail)
        raise IndexError("rain masks time steps are not all contained in hail fields time steps.")
    
    # check whether rain masks and hail fields have same shapes, and therefore whether tha max hail values can be determined
    determine_cell_max_hail = True
    if np.shape(hail_fields[0]) != np.shape(rain_masks[0]):
        determine_cell_max_hail = False
        print("Warning: hail fields and rain masks shapes do not match; the cell max hail diameter cannot be determined.")
    
    
    for i, nowdate in enumerate(timesteps): # loop over integer hours
        
        # compute vorticity fields on the 4 considered pressure levels
        zeta_400 = zeta_plev(fnames_p[i], fnames_s[i], 2)
        zeta_500 = zeta_plev(fnames_p[i], fnames_s[i], 3)
        zeta_600 = zeta_plev(fnames_p[i], fnames_s[i], 4)
        zeta_700 = zeta_plev(fnames_p[i], fnames_s[i], 5)
        zeta_4lev = np.stack([zeta_400, zeta_500, zeta_600, zeta_700])
        
        # load updraught velocity fileds on the 4 considered pressure levels
        with xr.open_dataset(fnames_p[i]) as dset:
            w_400 = dset['W'][0][2]
            w_500 = dset['W'][0][3]
            w_600 = dset['W'][0][4]
            w_700 = dset['W'][0][5]
        w_4lev = np.stack([w_400, w_500, w_600, w_700])
        
        # check whether rain masks and wind fields have same shapes; if not the case, crash the algorithm
        if np.shape(w_400) != np.shape(rain_masks[0]):
            raise IndexError("Wind fields and rain masks have different shapes.")
        
        # discriminate between positively and negatively signed vorticies; label mesocyclone canditates with 2D masks
        labeled_pos = label_above_thresholds(zeta_4lev, w_4lev, zeta_th, zeta_pk, w_th, min_area, aura, 1)
        labeled_neg = label_above_thresholds(zeta_4lev, w_4lev, zeta_th, zeta_pk, w_th, min_area, aura, -1)
        
        # find overlaps between mesocyclone canditates and rain cells, assign the mesocyclones to cells, store the other vorticies
        overlaps_pos, no_overlaps_pos = find_vortex_rain_overlaps(zeta_4lev, w_4lev, labeled_pos, rain_masks_hourly[i], zeta_th, w_th, 1)
        overlaps_neg, no_overlaps_neg = find_vortex_rain_overlaps(zeta_4lev, w_4lev, labeled_neg, rain_masks_hourly[i], zeta_th, w_th, -1)
        # merge positive and negative vorticies into one dictionnary
        overlaps = merge_dictionaries(overlaps_pos, overlaps_neg)
        no_overlaps = merge_dictionaries(no_overlaps_pos, no_overlaps_neg)
        
        # determine the current hail field slice for meso max hail
        current_hail_field = hail_fields[times_hail == nowdate][0] # 2D matrix
        
        for j, SC_id in enumerate(overlaps['cell_id']): # loop over mesocyclones which are assigned to a rain cell
            
            # determine the rain cell id of the supercell
            if np.size(SC_id) > 1: # choose the biggest overlap
                index = np.argmax(overlaps['overlap'][j])
                rain_cell_id = SC_id[index]
                overlap = overlaps['overlap'][j][index]
                sub_overlaps = [ovl for k, ovl in enumerate(overlaps['overlap'][j]) if k != index]
                sub_ids = [ID for k, ID in enumerate(SC_id) if k != index]
            else:
                rain_cell_id = SC_id
                overlap = overlaps['overlap'][j]
                sub_overlaps = None
                sub_ids = None
            
            # determine the mesocyclone centroid, in lon-lat coordinates
            coords = overlaps['coord'][j]
            cent_lon = float(np.mean(lons[coords[:,0], coords[:,1]]))
            cent_lat = float(np.mean(lats[coords[:,0], coords[:,1]]))
            
            # determine the max hail diameter within the mesocyclone
            meso_max_hail = float(np.nanmax(current_hail_field[coords[:,0], coords[:,1]]))
            
            # determine whether the cell is a new supercell or not
            if rain_cell_id in active_cells_ids:
                index = active_cells_ids.index(rain_cell_id)
                active_cells[index].append_candidate(nowdate, overlaps['signature'][j], overlaps['area'][j], float(overlaps['max_zeta'][j]),
                                                     float(overlaps['mean_zeta'][j]), float(overlaps['max_w'][j]), float(overlaps['mean_w'][j]),
                                                     np.round(meso_max_hail,1), coords, np.round(cent_lon,3), np.round(cent_lat,3), overlap, sub_ids, sub_overlaps)
            else:
                active_cells_ids.append(rain_cell_id)
                new_SC = SuperCell(rain_cell_id, nowdate, overlaps['signature'][j], overlaps['area'][j], float(overlaps['max_zeta'][j]),
                                   float(overlaps['mean_zeta'][j]), float(overlaps['max_w'][j]), float(overlaps['mean_w'][j]),
                                   np.round(meso_max_hail,1), coords, np.round(cent_lon,3), np.round(cent_lat,3), overlap, sub_ids, sub_overlaps)
                active_cells.append(new_SC)
        
        for j, sgn in enumerate(no_overlaps["signature"]): # loop over not assigned vorticies
            
            # determine the vortex centroid
            coords = no_overlaps['coord'][j]
            cent_lon = float(np.mean(lons[coords[:,0], coords[:,1]]))
            cent_lat = float(np.mean(lats[coords[:,0], coords[:,1]]))
            
            # determine the max hail diameter within the vortex
            vx_max_hail = float(np.nanmax(current_hail_field[coords[:,0], coords[:,1]]))
            
            na_vorticies.append(NA_Vortex(nowdate, sgn, no_overlaps['area'][j], float(no_overlaps['max_zeta'][j]), float(no_overlaps['mean_zeta'][j]),
                                          float(no_overlaps['max_w'][j]), float(no_overlaps['mean_w'][j]), np.round(vx_max_hail,1), coords,
                                          np.round(cent_lon,3), np.round(cent_lat,3)))
            
    
    # include the rain cells attributes (datelist, centroid coordinates, max rain, max hail and max surface wind) to the supercells
    for i, ID in enumerate(active_cells_ids):
        # extract cell centroid lon/lat coordinates, datelist and max rain rate values from rain tracks
        cell_tracks = rain_tracks[ID]
        cell_lon = [np.round(x,3) for x in cell_tracks['lon']]
        cell_lat = [np.round(x,3) for x in cell_tracks['lat']]
        cell_datelist = pd.to_datetime(cell_tracks['datelist'])
        cell_max_rain = [np.round(el*12,1) for el in cell_tracks['max_val']] # conversion of rain rate in mm/h and round
        
        # determine maximum hail diameter within the rain cell for each 5min-timestep in cell lifetime
        if not determine_cell_max_hail:
            cell_max_hail = np.nan
        else:
            cell_max_hail = []
            rain_masks_for_hail = [np.array(mask) for k, mask in enumerate(rain_masks) if times_5min[k] in cell_datelist]
            hail_fields_for_hail = [f for k, f in enumerate(hail_fields) if times_hail[k] in cell_datelist]
            if len(rain_masks_for_hail) != len(hail_fields_for_hail): # check that the 5min_2D hail fields time steps contain cell datelist
                raise IndexError("5min_2D concatenated files time steps do not fully contain cell datelist")
            for k, hail_field in enumerate(hail_fields_for_hail):
                hail_values = hail_field[rain_masks_for_hail[k] == ID]
                # during some merging/splitting processes, the cell changes its id, making it undetectable on the rain masks ...
                if len(hail_values) > 0: 
                    cell_max_hail.append(np.round(float(np.max(hail_values)), 1))
                else:
                    cell_max_hail.append(np.nan)
        
        # for wole domain: maximum 10m wind speed within the rain cell for each hourly timestep in cell lifetime
        if CS:
            cell_max_wind = np.nan
        else:
            cell_max_wind = []
            hourly_cell_datelist = [t for t in cell_datelist if t.strftime("%M")=="00"]
            rain_masks_for_wind = [np.array(mask) for k, mask in enumerate(rain_masks) if times_5min[k] in hourly_cell_datelist]
            fnames_s_for_wind = [f for k, f in enumerate(fnames_s) if timesteps[k] in hourly_cell_datelist]
            for j, fname_s in enumerate(fnames_s_for_wind):
                with xr.open_dataset(fname_s) as dset:
                    max_wind = np.array(dset['VMAX_10M'][0])
                wind_values = max_wind[rain_masks_for_wind[j] == ID]
                # during some merging/splitting processes, the cell changes its id, making it undetectable on the rain masks ...
                if len(wind_values) > 0:
                    cell_max_wind.append(np.round(float(np.max(wind_values)), 1))
                else:
                    cell_max_wind.append(np.nan)
        
        # add rain cell attributes to supercell
        active_cells[i].add_raincell_attributes(cell_datelist, cell_lon, cell_lat, cell_max_rain, cell_max_hail, cell_max_wind)
    
    # determine the minimum mesocyclone lifespan
    _ = [cell.post_processing() for cell in active_cells]
    
    # if two meso detections (at different time steps) are requested, filter out supercells with unique meso detection
    if two_meso_detections:
        active_cells = [cell for cell in active_cells if (len(np.unique(cell.meso_datelist)) > 1)]
    
    return active_cells, na_vorticies, lons, lats



def label_above_thresholds(zeta_4lev, w_4lev, zeta_th, zeta_pk, w_th, min_area, aura, sgn):
    """
    Labels above-thresholds and sufficiently large areas.
    Finds cohesive coincident vorticity and updraught velocity regions above the respective thresholds on the three levels
    and applies the vertical consistency criterion.
    
    in
    zeta_4lev : 2D vorticity field on the 4 pressure levels, 3D array
    w_4lev : 2D updraught velocity field on the 4 pressure levels, 3D array
    zeta_th : vorticity threshold value considered for all pixels within cell, float
    zeta_pk : vorticity peak threshold value within cell, default is None (no peak criterion), float
    w_th : updraught velocity threshold value considered as within cell, float
    min_area: minimum horizontal area threshold for each pressure level in grid point, int
    aura : number of gridpoints to dilate labels, dilation intervenes after min_area criterion is invoked, int
    sgn : signature of the vorticies, either 1 (positive) or -1 (negative), int

    out
    footprint : 2D vorticies footprints labeled with unique labels, starting from 1, 2D array
    """
    
    # mask above-zeta-threshold regions, above-w-threshold regions and apply duo-threshold criterion on each level
    zeta_above_threshold = sgn*zeta_4lev >= zeta_th
    w_above_threshold = w_4lev >= w_th
    above_thresholds = np.logical_and(zeta_above_threshold, w_above_threshold)
    
    # label all cohesive above-thresholds regions, filter out patterns not fulfilling min area criterion and dilate labels, on each level
    # reliable method unless patterns of the same sign are less than 3 pixels away from each other
    labeled = []
    for i in range(4):
        lbd, _ = ndimage.label(above_thresholds[i], structure=np.ones((3,3))) # 8-connectivity
        labels, counts = np.unique(lbd, return_counts=True)
        labels, counts = labels[1:], counts[1:]  # remove zero, corresponding to the background
        for label, count in zip(labels, counts):
            if zeta_pk:
                zeta_vals = zeta_4lev[i][lbd == label]
                zeta_max = np.max(np.abs(zeta_vals))
                peak_th_crit = zeta_max >= zeta_pk
            else:
                peak_th_crit = True
            if count < min_area or not peak_th_crit:
                lbd[lbd == label] = 0
        lbd = expand_labels(lbd, distance=aura) # dilates label to counteract vortex tilting with height
        labeled.append(lbd)
    labeled = np.stack(labeled)
    
    # check vertical consistency
    labeled_bin = np.where(labeled > 0, 1, 0) # binary labeling
    vc_3lev = np.logical_and(labeled_bin[2], np.logical_or(labeled_bin[1], labeled_bin[3])) # require at least 2 neighbours among the 3 lower levels
    vc_4lev = np.logical_or(vc_3lev, np.logical_and(labeled_bin[0], labeled_bin[1])) # or among the 2 upper
    
    # keep only the vertically consistent patterns on each level and find the zeta and w max/mean values
    for i in range(4):
        # labels of current level patterns fulfilling all the criteria, if any; should not contain any background pixel; can be empty
        labels_tokeep = np.unique(labeled[i][vc_4lev])
        labels_tokeep = labels_tokeep[labels_tokeep != 0]  # remove zero, corresponding to the background, if any
        labeled[i] = np.where(np.isin(labeled[i], labels_tokeep), 1, 0) # filter out vertically inconsistent patterns and make binary labeling        
    
    # aggregate the binary masks on the horizontal plane to create a 2D footprint and re-label the patterns
    footprint = np.sum(labeled, axis=0)
    footprint, _ = ndimage.label(footprint, structure=np.ones((3,3)))
    
    return footprint



def find_vortex_rain_overlaps(zeta_4lev, w_4lev, labeled, rain_mask, zeta_th, w_th, sgn):
    """
    finds overlaps between vorticies and rain cells for mesocyclone -> rain cell association
    outputs SCs to rain cells association with overlap, SC signature, max values, mean values and area
    
    in
    zeta_4lev : raw vorticity fields on the 4 levels, 3D array
    w_4lev : raw updraught velocity fields on the 4 levels, 3D array
    labeled: 2D labeled vorticies footprints (0 is background, starts at 1), 2D array
    rain_mask: labeled rain cells (nan is background, cell ids start at 0), 2D array
    zeta_th : vorticity threshold value, float
    w_th : updraught velocity threshold value, float
    sgn : signature of the vorticies, either 1 (positive) or -1 (negative), int
    
    out
    overlaps: a set containing 6 columns; detected mesocyclones are classified according to their signature (RM of LM), max zeta and w values,
              mean zeta and w values, area, gridpoint coordinates, and assigned to a(several) rain cell(s) with their corresponding overlap (in gp), set
    no_overlaps: set containing similar information, exept the overlap and corresponding rain cell ID, about not assigned vorticies
    """
    
    labels_VX = np.unique(labeled) # get a list of the vorticies labels
    labels_VX = labels_VX[labels_VX != 0] # remove zero, which corresponds to the background
    #cell_ids = np.unique(rain_mask)[:-1] # get a list of the cell ids and remove nans, corresponding to background
    
    #determine the overlap between vorticies and rain cells, and document not assigned vorticies
    overlaps = {"cell_id": [], "overlap": [], "signature": [], "area": [], "max_zeta": [], "mean_zeta": [], "max_w": [], "mean_w": [], "coord": []}
    no_overlaps = {"signature": [], "area": [], "max_zeta": [], "mean_zeta": [], "max_w": [], "mean_w": [], "coord": []}
    
    for VX_label in labels_VX:
        coordinates = np.argwhere(labeled == VX_label) #find (gridpoints) coordinates of the vortex
        ovl_matrix = np.logical_and(labeled == VX_label, np.logical_not(np.isnan(rain_mask)))
        ovl = np.count_nonzero(ovl_matrix)
        rain_mask = np.array(rain_mask)
        ovl_matrix = np.array(ovl_matrix)
        corr_cell_id = np.unique(rain_mask[ovl_matrix]).astype(int)
        #corr_cell_id = corr_cell_id[~np.isnan(corr_cell_id)].astype(int) #remove nans and convert from float to int type
        if np.any(np.isnan(corr_cell_id)):
            raise TypeError("overlap matrix between rain cell(s) " + str(corr_cell_id) + " and vortex " + str(VX_label + "containes nan(s)."))
                
        # for mean and max, consider all values above the surface footprint that fulfill the duo-threshold criterion
        # zeta_values = []
        # w_values = []        
        # for lev in range(4):
        #     above_ths_coords = np.argwhere(np.logical_and(np.abs(zeta_4lev[lev]) >= zeta_th, w_4lev[lev] >= w_th))
        #     val_coords = [c for c in coordinates.tolist() if c in above_ths_coords] # restrict footprint coordinates to above-thrs pixels, ie to the mesocyclone patch at that level
        #     val_coords = np.array(val_coords)
        #     if len(val_coords) > 0:
        #         zeta_values.extend(zeta_4lev[lev, val_coords[:,0], val_coords[:,1]])
        #         w_values.extend(w_4lev[lev, val_coords[:,0], val_coords[:,1]])
        
        # for mean and max values, consider all values above the surface footprint that fulfill the duo-threshold criterion
        zeta_values = []
        w_values = []
        for lev in range(4):
            zeta_val_lev = zeta_4lev[lev, coordinates[:,0], coordinates[:,1]]
            w_val_lev = w_4lev[lev, coordinates[:,0], coordinates[:,1]]
            to_keep = np.logical_and(sgn*zeta_val_lev >= zeta_th, w_val_lev >= w_th)
            if np.count_nonzero(to_keep) > 0:
                zeta_val_lev, w_val_lev = zeta_val_lev[to_keep], w_val_lev[to_keep]
                zeta_values.extend(zeta_val_lev)
                w_values.extend(w_val_lev)
        
        mean_zeta = np.round(np.mean(zeta_values), 5)
        #sgn = np.sign(mean_zeta).astype(int)
        #mean_zeta = "{:.2e}".format(mean_zeta)
        max_zeta = np.max(np.abs(zeta_values))
        max_zeta = np.round(sgn*max_zeta, 5)
        #max_zeta = "{:.2e}".format(max_zeta)
        mean_w = np.round(np.mean(w_values), 1)
        max_w = np.round(np.max(w_values), 1)
        area = len(coordinates) # np.count_nonzero(labeled == VX_label)
                
        # if the current SC overlaps with an unique rain cell, append the corresponding cell id and overlap area
        if ovl > 0 and len(corr_cell_id) == 1:
            overlaps["cell_id"].append(int(corr_cell_id))
            overlaps["overlap"].append(int(ovl))
            overlaps["signature"].append(int(sgn))
            overlaps["mean_zeta"].append(mean_zeta)
            overlaps["max_zeta"].append(max_zeta)
            overlaps["mean_w"].append(mean_w)
            overlaps["max_w"].append(max_w)
            overlaps["area"].append(area)
            overlaps["coord"].append(coordinates)
        elif ovl > 0 and len(corr_cell_id) > 1:
            sub_ovl = [] #by construction at least one pixel of the SC must overlap with either of the the rain cells
            for cell_id in corr_cell_id:
                sub_ovl_matrix = np.logical_and(labeled == VX_label, rain_mask == cell_id)
                sub_ovl.append(int(np.count_nonzero(sub_ovl_matrix)))
            overlaps["cell_id"].append(corr_cell_id.tolist())
            overlaps["overlap"].append(sub_ovl) #this time the new cell_id and overlap elements are both vectors
            overlaps["signature"].append(int(sgn))
            overlaps["mean_zeta"].append(mean_zeta)
            overlaps["max_zeta"].append(max_zeta)
            overlaps["mean_w"].append(mean_w)
            overlaps["max_w"].append(max_w)
            overlaps["area"].append(area)
            overlaps["coord"].append(coordinates)
        else: # no overlaps with any rain cell -> not assigned vortex
            no_overlaps["signature"].append(int(sgn))
            no_overlaps["mean_zeta"].append(mean_zeta)
            no_overlaps["max_zeta"].append(max_zeta)
            no_overlaps["mean_w"].append(mean_w)
            no_overlaps["max_w"].append(max_w)
            no_overlaps["area"].append(area)
            no_overlaps["coord"].append(coordinates)   
    
    return overlaps, no_overlaps


def merge_dictionaries(dict1, dict2):
    merged_dict = dict1.copy()  # start with a copy of the first dictionary
    
    for variable, value in dict2.items():
        if variable in merged_dict:
            merged_dict[variable].extend(value)  # extend the list if variable exists
        else:
            merged_dict[variable] = value  # add the new variable-value pair

    return merged_dict


class SuperCell:
    """
    supercell class with atributes for each cell
    """

    def __init__(self, rain_cell_id, nowdate, signature, area, max_zeta, mean_zeta, max_w, mean_w, meso_max_hail, coords, cent_lon, cent_lat,
                 overlap, sub_ids, sub_overlaps):
        """
        initialise cell object

        in
        rain_cell_id: rain cell id that has just been detected as supercell, unique, int
        nowdate: current hourly date, datetime
        signature: signature of the supercell, +1 or -1, int
        area: area (in gridpoints) of the dilated mesocyclone footprint, int
        max_zeta: maximum vorticity value of the mesocyclone, float
        mean_zeta: average vorticity value of the mesocyclone, float
        max_w: maximum updraught velocity value of the mesocyclone, float
        mean_w: average updraught velocity value of the mesocyclone, float
        meso_max_hail: maximum hail diameter within the mesocyclone, float
        coords: gridpoint coordinates of the mesocyclone, list of 2D coordinates
        cent_lon: longitude of mesocyclone centroid, float
        cent_lat: latitude of mesocyclone centroid, float
        overlap: overlap (in gridpoints) between the mesocyclone and the rain cell, int
        sub_ids: if the mesocyclone overlaps with more than one rain cell, list of the other rain cells ids, list of int
        sub_overlaps: if the mesocyclone overlaps with more than one rain cell, list of the respective overlaps (in gridpoints), list of int
        """
        self.rain_cell_id = rain_cell_id # ID of the rain cell the supercell was assigned to
        self.meso_datelist = [] # integer hours at which a mesocyclone was detected within the supercell
        self.meso_datelist.append(nowdate)
        self.signature = [] 
        self.signature.append(signature)
        self.area = []
        self.area.append(area)
        self.max_zeta = [] 
        self.max_zeta.append(max_zeta)
        self.mean_zeta = [] 
        self.mean_zeta.append(mean_zeta)
        self.max_w = [] 
        self.max_w.append(max_w)
        self.mean_w = [] 
        self.mean_w.append(mean_w)
        self.meso_max_hail = []
        self.meso_max_hail.append(meso_max_hail)
        self.overlap = [] # overlaps between the respective mesocyclones and the supercell
        self.overlap.append(overlap)
        self.updraft_min_lifespan = None # will be filled in if several mesocyclones are detected at different time steps
        self.coordinates = []
        self.coordinates.append(coords)
        self.meso_lon = []
        self.meso_lon.append(cent_lon)
        self.meso_lat = []
        self.meso_lat.append(cent_lat)
        self.sub_ids = []
        self.sub_ids.append(sub_ids)
        self.sub_overlaps = []
        self.sub_overlaps.append(sub_overlaps)
        
        # define the presence of the subsequently-added rain cell attributes
        self.cell_datelist = None
        self.cell_lon = None
        self.cell_lat = None
        self.cell_max_rain = None
        self.cell_max_hail = None
        self.cell_max_wind = None


    def post_processing(self):
        """
        post process cells
        used on calculations that are performed after tracking
        """
        self.updraft_min_lifespan = self.meso_datelist[-1] - self.meso_datelist[0]
        
    
    def append_candidate(self, nowdate, signature, area, max_zeta, mean_zeta, max_w, mean_w, meso_max_hail, coords,
                         cent_lon, cent_lat, overlap, sub_ids, sub_overlaps):
        """
        appends the newly detected mesocyclone with the provided attributes to the self cell
        """
        self.meso_datelist.append(nowdate)
        self.signature.append(signature)
        self.area.append(area)
        self.max_zeta.append(max_zeta)
        self.mean_zeta.append(mean_zeta)
        self.max_w.append(max_w)
        self.mean_w.append(mean_w)
        self.meso_max_hail.append(meso_max_hail)
        self.coordinates.append(coords)
        self.meso_lon.append(cent_lon)
        self.meso_lat.append(cent_lat)
        self.overlap.append(overlap)
        self.sub_ids.append(sub_ids)
        self.sub_overlaps.append(sub_overlaps)
    
    
    def add_raincell_attributes(self, cell_datelist, cell_lon, cell_lat, cell_max_rain, cell_max_hail, cell_max_wind):
        """
        creates new attributes for the supercell, related to the rain cell

        Parameters
        ----------
        cell_datelist : list of datetime
            list of datetime objects for each 5min-timestep in cell lifetime
        cell_lon : list of float
            list of mass center longitudes for each 5min-timestep in cell lifetime
        cell_lat : list of float
            list of mass center latitudes for each 5min-timestep in cell lifetime
        cell_max_rain : list of float
            list of maximum rain rate within the cell for each 5min-timestep in cell lifetime
        cell_max_hail : list of float
            list of maximum hail diameter within the cell for each 5min-timestep in cell lifetime
        cell_max_wind : list of float
            list of maximum 10m wind speed within the cell for each hourly timestep in cell lifetime
        """
        
        self.cell_datelist = cell_datelist
        self.cell_lon = cell_lon
        self.cell_lat = cell_lat
        self.cell_max_rain = cell_max_rain
        self.cell_max_hail = cell_max_hail
        self.cell_max_wind = cell_max_wind
    

    def to_dict(self):
        """
        returns a dictionary containing all cell object information

        out
        cell_dict: dictionary containing all cell object information, dict
        """
        cell_dict = {
            "rain_cell_id": self.rain_cell_id,
            "meso_datelist": [str(t)[:16] for t in self.meso_datelist],
            "signature": self.signature,
            "area": self.area, #[int(x) for x in self.area],
            "max_zeta": self.max_zeta,
            "mean_zeta": self.mean_zeta,
            "max_w": self.max_w,
            "mean_w": self.mean_w,
            "meso_max_hail" : self.meso_max_hail,
            "meso_lon": self.meso_lon,
            "meso_lat": self.meso_lat,
            "updraft_min_lifespan": int(self.updraft_min_lifespan / np.timedelta64(1, "h")),
            "overlap": self.overlap,
            "sub_ids": self.sub_ids,
            "sub_overlaps": self.sub_overlaps,
            "cell_datelist": [str(t)[:16] for t in self.cell_datelist],
            "cell_lon": self.cell_lon,
            "cell_lat": self.cell_lat,
            "cell_max_rain": self.cell_max_rain,
            "cell_max_hail": self.cell_max_hail,
            "cell_max_wind": self.cell_max_wind
        }
        return cell_dict



class NA_Vortex:
    """
    Class for single-detected vorticies which have not been assigned to any rain cell, and hence not considered as mesocyclones
    """
    
    def __init__(self, nowdate, signature, area, max_zeta, mean_zeta, max_w, mean_w, vx_max_hail, coord, cent_lon, cent_lat):
        self.date = nowdate
        self.signature = signature
        self.area = area
        self.max_zeta = max_zeta
        self.mean_zeta = mean_zeta
        self.max_w = max_w
        self.mean_w = mean_w
        self.vx_max_hail = vx_max_hail
        self.coordinates = coord
        self.vx_lon = cent_lon
        self.vx_lat = cent_lat
        
    
    def to_dict(self):
        """
        returns a dictionary containing all object information

        out
        cell_dict: dictionary containing all object information, dict
        """
        cell_dict = {
            "date": str(self.date)[:16],
            "signature": self.signature,
            "area": self.area,
            "max_zeta": self.max_zeta,
            "mean_zeta": self.mean_zeta,
            "max_w": self.max_w,
            "mean_w": self.mean_w,
            "vx_max_hail" : self.vx_max_hail,
            "vx_lon": self.vx_lon,
            "vx_lat": self.vx_lat
        }
        return cell_dict



def write_to_json(cells, na_vorticies, filename):
    """
    writes ascii file containing SuperCell and NA_Vortex objects information

    in
    cells: list of supercells, list
    na_vorticies: list of not assigned vorticies, list
    filename: filename, string

    out
    struct: dictionary containing all SuperCell and Missed_Mesocyclone objects information, dict
    """

    # figure out wether input is list or empty
    if len(cells) == 0:
        struct_SC = []
        data_structure = "no supercells found"
    elif isinstance(cells[0], SuperCell):  # single member
        struct_SC = [cell.to_dict() for cell in cells]
        data_structure = "supercell_data contains list of supercells, each supercell contains a dictionary of its parameters"

    if len(na_vorticies) == 0:
        struct_nav = []
        data_structure += " ; no not assigned vorticies found"   
    elif isinstance(na_vorticies[0], NA_Vortex):
        struct_nav = [nav.to_dict() for nav in na_vorticies]
        data_structure += " ; na_vortex_data contains list of vorticies not assigned to any rain cells, each of which contains a dictionary of its parameters"

    struct = {
        "info": "supercell track data generated using the second version of Supercell Detection and Tracking (SDT2)",
        "author": "Michael Blanc (mblanc@student.ethz.ch)",
        "data_structure": data_structure,
        "parameters": {
            "rain_cell_id": "id of the rain cell the supercell was assigned to",
            "meso_datelist": "list of datetimes corresponding to mesocyclone detection(s) within the supercell",
            "signature": "vorticity signature of the supercell (+1 for mesocyclones and -1 for mesoanticyclones)",
            "area": "mesocyclone area in gridpoints, after dilation and 2D footprint projection (gp)",
            "max_zeta": "maximum vorticity value within the mesocyclone (1/s)",
            "mean_zeta": "mean vorticity value within the mesocyclone (1/s)",
            "max_w": "maximum updraught velocity value within the mesocyclone (m/s)",
            "mean_w": "mean updraught velocity value within the mesocyclone (m/s)",
            "meso_max_hail": "maximum hail diameter within the mesocyclone (mm)",
            "meso_lon": "longitude of mesocyclone centroid (°E)",
            "meso_lat": "latitude of mesocyclone centroid (°N)",
            "updraft_min_lifespan": "minimum lifespan of mesocyclone in hours",
            "overlap": "overlap in gridpoints between the (dilated) mesocyclone and the rain cell (gp)",
            "sub_ids": "list of other rain cells the mesocyclone overlaps with",
            "sub_overlaps": "list of respective overlaps (gp)",
            "cell_datelist": "list of datetime objects for each 5min-timestep in cell lifetime",
            "cell_lon" : "list of mass center longitudes for each 5min-timestep in cell lifetime (°E)",
            "cell_lat" : "list of mass center latitudes for each 5min-timestep in cell lifetime (°N)",
            "cell_max_rain" : "list of maximum rain rate within the cell for each 5min-timestep in cell lifetime (mm/h)",
            "cell_max_hail": "list of maximum hail diameter within the cell for each 5min-timestep in cell lifetime (mm)",
            "cell_max_wind": "list of maximum 10m wind speed within the cell for each hourly timestep in cell lifetime (m/s)"
        },
        "supercell_data": struct_SC,
        "na_vortex_data": struct_nav
    }

    with open(filename, "w") as f:
        json.dump(struct, f)



def write_masks_to_netcdf(
    supercells,
    lons,
    lats,
    filename,
):
    """
    writes netcdf file containing mesocyclones binary masks

    in
    supercells: list of SuperCell objects, list
    lons: longitude at each gridpoint, 2D array
    lats: latitude at each gridpoint, 2D array
    filename: filename, string

    out
    ds: xarray dataset containing mesocyclones binary masks, xarray dataset
    """
    
    # create the datelist listing all hourly timesteps where a mesocyclone detection occured
    meso_datelist = []
    for cell in supercells:
        meso_datelist.extend(cell.meso_datelist)
    meso_datelist = np.unique(meso_datelist) # reduces the timesteps to unique ones and sorts them chronologically
    
    mask_array = np.zeros((len(meso_datelist), lats.shape[0], lons.shape[1]))
    mask_array[:] = False

    for cell in supercells: # loop over every supercell
        for i, t in enumerate(cell.meso_datelist): # loop over every mesocyclone detection
            date_index = np.where(meso_datelist == t)[0][0]
            mask_array[date_index, cell.coordinates[i][:,0], cell.coordinates[i][:,1]] = True

    coords = {
        "time": meso_datelist,
        "y": np.arange(lats.shape[0]),
        "x": np.arange(lons.shape[1]),
    }
    data_structure = {
        "meso_mask": (["time", "y", "x"], mask_array),
        "lat": (["y", "x"], lats),
        "lon": (["y", "x"], lons),
    }

    # create netcdf file
    ds = xr.Dataset(data_structure, coords=coords)

    # write to netcdf file
    # ds.to_netcdf(filename)
    ds.to_netcdf(filename, encoding={'meso_mask': {'zlib': True, 'complevel': 9}})

    # compress netcdf file
    # os.system("nczip " + filename)

    #return ds
