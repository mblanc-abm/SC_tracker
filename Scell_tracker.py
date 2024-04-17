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

import os
import sys
sys.path.append("../first_visu")
from CaseStudies import IUH

#=======================================================================================================================================

def track_Scells(day, timesteps, fnames_p, fnames_s, rain_masks_name, rain_tracks_name, threshold, sub_threshold, min_area, aura, fnames_h=None, path_meso_h=None):
    """
    description to be written

    Parameters
    ----------
    day: datetime
        considered day
    timesteps: list of datetime objects
        list of hourly datetimes, corresponding to the integer hours of the considered day where the IUH field is available
    fnames_p : list of str
        complete paths to the 3D pressure levels files containing the wind fields, for IUH
    fnames_s : list of str
        complete paths to the 2D surface pressure files, for IUH and max 10m wind speed
    rain_masks_name : str
        path to the netcdf rain mask: labeled rain cells (nan is background, cell labels start at 0), 2D arrays, concatenated with a 5 min resolution
    rain_tracks_name : str
        path to the output json file of the cell tracker algorithm 
    threshold : float
        threshold for minimum peak value within a supercell
    sub_threshold : float
        threshold for minimum values with all grid points of a supercell
    min_area : int
        minimum area (in gridpoints) for a supercell to be considered
    aura : int
        number of gridpoints to dilate labels, supercell post-dilation intervenes after min_area criteria is invoked
    fnames_h: list of str
        complete paths to the 5min-resolution 2D surface hail data files, for max hail value within the rain cell. Default is None
    path_meso_h: str
        path to the folder of the concatenated and compressed 5min-2D surface precipitation and hail fields files, for hail within the mesocyclone. Default is None.
        ex: "../lffd"
    
    Returns
    -------
    supercells: list of SuperCell objects
    missed_mesocyclones: list of Missed_Mesocyclone objects
    lons: longitude at each gridpoint, 2D array
    lats: latitude at each gridpoint, 2D array
    """
    
    # rain masks opening
    with xr.open_dataset(rain_masks_name) as dset:
        rain_masks = dset['cell_mask'] # 3D matrix
        times_5min = pd.to_datetime(dset['time'].values)
        lons = dset["lon"].values
        lats = dset["lat"].values
    
    # decrease temporal resolution to match IUH hourly one
    rain_masks_hourly = [mask for i, mask in enumerate(rain_masks) if times_5min[i] in timesteps]
    
    # rain tracks opening
    with open(rain_tracks_name, "r") as read_file:
        dset = json.load(read_file)
    rain_tracks = dset['cell_data'] # rain_tracks[j] contains all info about cell j
   
    active_cells = [] #in terms of supercellular activity
    active_cells_ids = [] #and their corresponding ids
    missed_mesocyclones = [] #objects of mesocyclones which did not overlap with any rain cell
    
    # if requested, determine the max hail diameter within the mesocyclone
    if path_meso_h is not None:
        
        path = os.path.join(path_meso_h, day.strftime("%Y%m%d") + "_0606.nz")
        with xr.open_dataset(path, engine="netcdf4") as ds:
            hail = ds["DHAIL_MX"][:-24].values
        
        day_before = day - pd.Timedelta(1,"d")
        path2 = os.path.join(path_meso_h, day_before.strftime("%Y%m%d") + "_0606.nz")
        with xr.open_dataset(path2, engine="netcdf4") as ds:
            hail_2 = ds["DHAIL_MX"][-24:].values
        
        hail_meso = np.concatenate([hail_2, hail])
    
    for i, nowdate in enumerate(timesteps): # loop over integer hours
        
        field = np.array(IUH(fnames_p[i], fnames_s[i]))
        labeled = label_above_threshold(field, threshold, sub_threshold, min_area)
        overlaps, no_overlaps = find_overlaps(field, labeled, rain_masks_hourly[i], aura)
        
        for j, SC_id in enumerate(overlaps['cell_id']): # loop over mesocyclones which will be assigned to a rain cell
            
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
            
            # determine the mesocyclone centroid
            coords = overlaps['coord'][j]
            cent_lon = float(np.mean(lons[coords[:,0], coords[:,1]]))
            cent_lat = float(np.mean(lats[coords[:,0], coords[:,1]]))
            
            # if requested, determine the max hail diameter within the mesocyclone
            if path_meso_h is not None:
                current_hail_meso = hail_meso[times_5min == nowdate] # 2D matrix
                max_hail = float(np.nanmax(current_hail_meso[coords[:,0], coords[:,1]]))
            else:
                max_hail = None
            
            # determine whether the rain cell is a new supercell or not
            if rain_cell_id in active_cells_ids:
                index = active_cells_ids.index(rain_cell_id)
                active_cells[index].append_candidate(nowdate, overlaps['signature'][j], overlaps['area'][j], float(overlaps['max_val'][j]), float(overlaps['mean_val'][j]), coords, round(cent_lon,2), round(cent_lat,2), overlap, max_hail, sub_ids, sub_overlaps)
            else:
                active_cells_ids.append(rain_cell_id)
                new_SC = SuperCell(rain_cell_id, nowdate, overlaps['signature'][j], overlaps['area'][j], float(overlaps['max_val'][j]), float(overlaps['mean_val'][j]), coords, round(cent_lon,2), round(cent_lat,2), overlap, max_hail, sub_ids, sub_overlaps)
                active_cells.append(new_SC)
        
        for j, sgn in enumerate(no_overlaps["signature"]): # loop over missed mesocyclones
            
            # determine the mesocyclone centroid
            coords = no_overlaps['coord'][j]
            cent_lon = float(np.mean(lons[coords[:,0], coords[:,1]]))
            cent_lat = float(np.mean(lats[coords[:,0], coords[:,1]]))
            
            missed_mesocyclones.append(Missed_Mesocyclone(nowdate, sgn, no_overlaps['area'][j], float(no_overlaps['max_val'][j]), float(no_overlaps['mean_val'][j]), coords, round(cent_lon,2), round(cent_lat,2)))
     
        
    # include the chosen rain cells attributes to the supercells
    for i, ID in enumerate(active_cells_ids):
        # extract rain cell centroid lon/lat coordinates, datelist and max rain rate values from rain tracks
        cell_tracks = rain_tracks[ID]
        cell_lon = [round(x,2) for x in cell_tracks['lon']]
        cell_lat = [round(x,2) for x in cell_tracks['lat']]
        cell_datelist = pd.to_datetime(cell_tracks['datelist'])
        cell_max_rain = cell_tracks['max_val']
        cell_max_rain = [round(el*12,1) for el in cell_max_rain] # conversion of rain rate in mm/h and round
        
        # if requested, determine maximum hail diameter within the rain cell for each 5min-timestep in cell lifetime
        if fnames_h is not None:
            
            cell_max_hail = []
            rain_masks_for_hail = [np.array(mask) for k, mask in enumerate(rain_masks) if times_5min[k] in cell_datelist]
            fnames_h_for_hail = [f for k, f in enumerate(fnames_h) if times_5min[k] in cell_datelist]
            for k, fname_h in enumerate(fnames_h_for_hail):
                with xr.open_dataset(fname_h) as dset:
                    hail_field = np.array(dset['DHAIL_MX'][0])
                hail_values = hail_field[rain_masks_for_hail[k] == ID]
                # during some merging/splitting processes, the cell changes its id, making it undetectable on the rain masks ...
                if len(hail_values) > 0: 
                    cell_max_hail.append(round(float(np.max(hail_values)), 1))
                else:
                    cell_max_hail.append(float(0.0))
        
        else: # None object when not requested
            cell_max_hail = None
        
        # maximum 10m wind speed within the rain cell for each hourly timestep in cell lifetime
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
                cell_max_wind.append(round(float(np.max(wind_values)), 1))
            else:
                cell_max_wind.append(float(0.0))
        
        active_cells[i].add_raincell_attributes(cell_datelist, cell_lon, cell_lat, cell_max_rain, cell_max_hail, cell_max_wind)
    
    _ = [cell.post_processing() for cell in active_cells]
    
    return active_cells, missed_mesocyclones, lons, lats



def label_above_threshold(field, threshold, sub_threshold, min_area):
    """
    Labels above-threshold and sufficiently large areas.
    Finds cohesive IUH regions above sub_threshold which contains at least one pixel above threshold.
    For sub_threshold = threshold, this is equivalent to searching for an uniform minimum threshold value.
    
    in
    field: 2D IUH field, array
    threshold: threshold for minimum peak value considered as within cell, float
    sub_theshold: threshold for minimum value considered as within cell, float
    min_area: minimum area (in gridpoints) for a cell to be considered, int

    out
    labeled: connected component labeled with unique label, array
    """

    # uses watershed method, fills adjacent cells so they touch at watershed line
    above_threshold = abs(field) >= sub_threshold
    labeled, _ = ndimage.label(above_threshold, structure=np.ones((3,3))) # 8-connectivity
    #labeled = watershed(field, markers, mask=above_threshold)
    
    #get rid of too small areas as well as areas which don't peak at threshold
    labels = np.unique(labeled).tolist()
    labels = [i for i in labels if i != 0]  # remove zero, corresponding to the background
    
    for label in labels:
        count = np.count_nonzero(labeled == label)
        max_value = np.max(abs(field[labeled == label]))
        if count < min_area or max_value < threshold:
            labeled[labeled == label] = 0
    
    return labeled



def find_overlaps(field, labeled, rain_mask, aura=0, printout=False):
    """
    finds overlaps between supercells and rain cells for SC -> rain cell association
    outputs SCs to rain cells association with overlap, SC signature, max value, mean value and area
    
    in
    labeled: labeled supercell areas (label number arbitrary, 0 is background), 2D array
    rain_mask: labeled rain cells (nan is background, cell ids start at 0), 2D array
    aura: number of gridpoints to dilate labels (post-dilation, called after min area is invoked !), int
    
    out
    overlaps: a set containing 6 columns; detected mesocyclones are classified according to their signature (RM of LM), max value,
              mean value, area, gridpoint coordinates, and assigned to a(several) rain cell(s) with their corresponding overlap (in gp), set
    no_overlaps: set containing information about signature, area, max and mean value, and gridpoint coordinates of missed mesocyclones
    """
    
    labels_SC = np.unique(labeled).tolist() # get a list of the SCs label numbers
    labels_SC = [i for i in labels_SC if i != 0] # remove zero, which corresponds to the background
    #cell_ids = np.unique(rain_mask)[:-1] # get a list of the cell ids and remove nans, corresponding to background
    
    #dilate labels,will be used only for searching overlaps !
    if aura:
        labeled_dil = expand_labels(labeled, distance=aura)
    else:
        labeled_dil = labeled
    
    #determine the overlap between mesocyclones and rain cells, and document not assigned mesocyclones
    overlaps = {"cell_id": [], "overlap": [], "signature": [], "area": [], "max_val": [], "mean_val": [], "coord": []}
    no_overlaps = {"signature": [], "area": [], "max_val": [], "mean_val": [], "coord": []}
    
    for SC_label in labels_SC:
        ovl_matrix = np.logical_and(labeled_dil == SC_label, np.logical_not(np.isnan(rain_mask))) #use the dilated labels for overlap
        ovl = np.count_nonzero(ovl_matrix)
        rain_mask = np.array(rain_mask)
        ovl_matrix = np.array(ovl_matrix)
        corr_cell_id = np.unique(rain_mask[ovl_matrix])
        corr_cell_id = corr_cell_id[~np.isnan(corr_cell_id)].astype(int) #remove nans and convert from float to int type
        
        coordinates = np.argwhere(labeled == SC_label) #find (gridpoints) coordinates of mesocyclone
        mean_val = round(np.mean(field[coordinates[:,0], coordinates[:,1]]), 1)
        sgn = np.sign(mean_val).astype(int)
        max_val = round(np.max(abs(field[coordinates[:,0], coordinates[:,1]])), 1)
        area = len(coordinates) # np.count_nonzero(labeled == SC_label)
                
        # if the current SC overlaps with an unique rain cell, append the corresponding cell id and overlap area
        if ovl > 0 and len(corr_cell_id) == 1:
            overlaps["cell_id"].append(int(corr_cell_id))
            overlaps["overlap"].append(int(ovl))
            overlaps["signature"].append(int(sgn))
            overlaps["mean_val"].append(mean_val)
            overlaps["max_val"].append(sgn*max_val)
            overlaps["area"].append(area)
            overlaps["coord"].append(coordinates)
        elif ovl > 0 and len(corr_cell_id) > 1:
            sub_ovl = [] #by construction at least one pixel of the SC must overlap with either of the the rain cells
            for cell_id in corr_cell_id:
                sub_ovl_matrix = np.logical_and(labeled_dil == SC_label, rain_mask == cell_id)
                sub_ovl.append(int(np.count_nonzero(sub_ovl_matrix)))
            overlaps["cell_id"].append(corr_cell_id.tolist())
            overlaps["overlap"].append(sub_ovl) #this time the new cell_id and overlap elements are both vectors
            overlaps["signature"].append(int(sgn))
            overlaps["mean_val"].append(mean_val)
            overlaps["max_val"].append(sgn*max_val)
            overlaps["area"].append(area)
            overlaps["coord"].append(coordinates)
        else: # no overlaps with any rain cell
            no_overlaps["signature"].append(int(sgn))
            no_overlaps["mean_val"].append(mean_val)
            no_overlaps["max_val"].append(sgn*max_val)
            no_overlaps["area"].append(area)
            no_overlaps["coord"].append(coordinates)
    
    
    if printout:
        print("Mesocyclones overlapping with rain cells:")
        print(overlaps)
        print("Missed mesocyclones:")
        print(no_overlaps)
        
    return overlaps, no_overlaps



class SuperCell:
    """
    supercell class with atributes for each cell
    """

    def __init__(self, rain_cell_id, nowdate, signature, area, max_val, mean_val, coord, cent_lon, cent_lat, overlap, max_hail, sub_ids, sub_overlaps):
        """
        inizialize cell object

        in
        rain_cell_id: rain cell id that has just been detected as supercell, unique, int
        nowdate: current date, datetime
        signature: signature of the supercell, +1 or -1, int
        area: area (in gridpoints) of the mesocyclone, int
        max_val: maximum IUH value of the mesocyclone, float
        mean_val: average IUH value of the mesocyclone, float
        coord: gridpoint coordinates of the mesocyclone
        cent_lon: longitude of mesocyclone centroid
        cent_lat: latitude of mesocyclone centroid
        overlap: overlap (in gridpoints) between the mesocyclone and the rain cell, int
        max_hail: maximum hail diameter within the mesocyclone, float (None if not requested)
        sub_ids: if the mesocyclone overlaps with more than 1 rain cell, list of the other rain cells ids, list of int
        sub_overlaps: if the mesocyclone overlaps with more than 1 rain cell, list of the respective overlaps (in gridpoints), list of int
        """
        self.rain_cell_id = rain_cell_id # ID of the rain cell the supercell was assigned to
        self.meso_datelist = [] # integer hours at which a mesocyclone was detected within the supercell
        self.meso_datelist.append(nowdate)
        self.signature = [] 
        self.signature.append(signature)
        self.area = []
        self.area.append(area)
        self.max_val = [] 
        self.max_val.append(max_val)
        self.mean_val = [] 
        self.mean_val.append(mean_val)
        self.overlap = [] # overlaps between the respective mesocyclones and the supercell
        self.overlap.append(overlap)
        self.updraft_min_lifespan = None
        self.coordinates = []
        self.coordinates.append(coord)
        self.meso_lon = []
        self.meso_lon.append(cent_lon)
        self.meso_lat = []
        self.meso_lat.append(cent_lat)
        self.max_hail = []
        self.max_hail.append(max_hail)
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
        
    
    def append_candidate(self, nowdate, signature, area, max_val, mean_val, coord, cent_lon, cent_lat, overlap, max_hail, sub_ids, sub_overlaps):
        """
        appends the newly detected mesocyclone with the provided attributes to the self cell
        """
        self.meso_datelist.append(nowdate)
        self.signature.append(signature)
        self.area.append(area)
        self.max_val.append(max_val)
        self.mean_val.append(mean_val)
        self.coordinates.append(coord)
        self.meso_lon.append(cent_lon)
        self.meso_lat.append(cent_lat)
        self.overlap.append(overlap)
        self.max_hail.append(max_hail)
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
            "max_val": self.max_val,
            "mean_val": self.mean_val,
            "meso_lon": self.meso_lon,
            "meso_lat": self.meso_lat,
            "updraft_min_lifespan": int(self.updraft_min_lifespan / np.timedelta64(1, "h")),
            "overlap": self.overlap,
            "meso_max_hail": self.max_hail,
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



class Missed_Mesocyclone:
    """
    Class for single-detected mesocyclones which have not been assigned to a rain cell, and hence not considered as supercells
    """
    
    def __init__(self, nowdate, signature, area, max_val, mean_val, coord, cent_lon, cent_lat):
        self.date = nowdate
        self.signature = signature
        self.area = area
        self.max_val = max_val
        self.mean_val = mean_val
        self.coordinates = coord
        self.meso_lon = cent_lon
        self.meso_lat = cent_lat
        
    
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
            "max_val": self.max_val,
            "mean_val": self.mean_val,
            "meso_lon": self.meso_lon,
            "meso_lat": self.meso_lat
        }
        return cell_dict



def write_to_json(cells, missed_mesos, filename):
    """
    writes ascii file containing SuperCell and Missed_Mesocyclone objects information

    in
    cells: list of supercells, list
    missed_mesos: list of missed mesocyclones, list
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

    if len(missed_mesos) == 0:
        struct_mm = []
        data_structure += " ; no missed mesocyclones found"
        
    elif isinstance(missed_mesos[0], Missed_Mesocyclone):
        struct_mm = [mm.to_dict() for mm in missed_mesos]
        data_structure += " ; missed_mesocyclone_data contains list of missed mesocyclones, each of which contains a dictionary of its parameters"

    struct = {
        "info": "supercell track data generated using Supercell Detection and Tracking (SDT)",
        "author": "Michael Blanc (mblanc@student.ethz.ch)",
        "data_structure": data_structure,
        "parameters": {
            "rain_cell_id": "id of the rain cell the supercell was assigned to",
            "meso_datelist": "list of datetimes corresponding to mesocyclone detection(s) within the supercell",
            "signature": "vorticity signature of the supercell (+1 for mesocyclones and -1 for mesoanticyclones)",
            "area": "mesocyclone area in gridpoints (gp)",
            "max_val": "maximum IUH value within the mesocyclone (m^2/s^2)",
            "mean_val": "mean IUH value within the mesocyclone (m^2/s^2)",
            "meso_lon": "longitude of mesocyclone centroid (°E)",
            "meso_lat": "latitude of mesocyclone centroid (°N)",
            "updraft_min_lifespan": "minimum lifespan of mesocyclone in hours",
            "overlap": "overlap in gridpoints between the (dilated) mesocyclone and the rain cell (gp)",
            "meso_max_hail": "maximum hail diameter within the mesocyclone at the mesocyclone detection (mm)",
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
        "missed_mesocyclone_data": struct_mm
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
