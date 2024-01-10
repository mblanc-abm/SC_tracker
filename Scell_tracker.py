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
from skimage.feature import peak_local_max
from skimage.segmentation import expand_labels, watershed
#from skimage.morphology import disk

import sys
sys.path.append("../first_visu")
from CaseStudies import IUH

#=======================================================================================================================================

def track_Scells(timesteps, fnames_p, fnames_s, rain_masks_name, rain_tracks_name, threshold, sub_threshold, min_area, aura):
    """
    description to be written

    Parameters
    ----------
    timesteps: list of datetime objects
        list of hourly datetimes, corresponding to the integer hours where the IUH field is available
    fnames_p : list of str
        complete paths to the 3D pressure levels files containing the wind fields, for IUH
    fnames_s : list of str
        complete paths to the 2D surface pressure files, for IUH
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
    
    Returns
    -------
    supercells:
        list of SuperCell objects
    """
    # rain masks opening
    with xr.open_dataset(rain_masks_name) as dset:
        rain_masks = dset['cell_mask'] # 3D matrix
        times_5min = pd.to_datetime(dset['time'].values)
    # select the integer hours contained in timesteps, to subsequently select the corresponding hourly rain masks
    #time_slices = np.logical_and(times_5min.strftime("%M") == '00', np.isin(times_5min, timesteps))
    
    #rain_masks = rain_masks[np.isin(times_5min, timesteps)] # decrease temporal resolution to match IUH hourly one
    rain_masks = [mask for i, mask in enumerate(rain_masks) if times_5min[i] in timesteps]
    
    # print(times_5min)
    # print(timesteps)
    # print(np.isin(times_5min, timesteps))
    
    # rain tracks opening
    # with open(rain_tracks_name, "r") as read_file:
    #     dset = json.load(read_file)
    # rain_tracks = dset['cell_data'] # rain_tracks[j] contains all info about cell j
    
    #n_rain_cells = len(rain_tracks)
    #all_rain_ids = [rain_masks[i]['cell_id'] for i in range(n_rain_cells)] #stores all the ids of detected the rain cells
    active_cells = [] #in terms of supercellular activity
    active_cells_ids = []
    
    for i, nowdate in enumerate(timesteps):
        field = np.array(IUH(fnames_p[i], fnames_s[i]))
        labeled = label_above_threshold(field, threshold, sub_threshold, min_area)
        overlaps = find_overlaps(field, labeled, rain_masks[i], aura)
        
        for j, SC_id in enumerate(overlaps['cell_id']):
            # determine the rain cell id of the supercell
            if np.size(SC_id) > 1: # choose the biggest overlap
                index = np.argmax(overlaps['overlap'][j])
                rain_cell_id = SC_id[index]
                overlap = overlaps['overlap'][j][index]
            else:
                rain_cell_id = SC_id
                overlap = overlaps['overlap'][j]
            # determine whether the rain cell is a new supercell or not
            if rain_cell_id in active_cells_ids:
                index = active_cells_ids.index(rain_cell_id)
                active_cells[index].append_candidate(nowdate, overlaps['signature'][j], overlaps['area'][j], overlaps['max_val'][j], overlaps['mean_val'][j], overlap)
            else:
                active_cells_ids.append(rain_cell_id)
                active_cells.append(SuperCell(rain_cell_id, nowdate, overlaps['signature'][j], overlaps['area'][j], overlaps['max_val'][j], overlaps['mean_val'][j], overlap))
            
    
    _ = [cell.post_processing() for cell in active_cells]
    
    return active_cells



#method of Killian, does not work here
def label_local_maxima(field, threshold, min_distance=1, aura=0):
    """
    labels areas of lokal peaks (separated by min_distance)

    in
    field: 2D IUH field, array
    threshold: threshold for minimum value considered as within cell, float
    min_distance: minimum distance (in gridpoints) between local maxima, float
    aura: number of gridpoints to dilate labels (post-dilation), int

    out
    labeled: connected component labeled with unique label, array
    above_threshold: above threshold binary area, array
    """
    
    #consider both neg and pos IUH peaks
    field[abs(field) < threshold] = 0
    peaks = peak_local_max(abs(field), min_distance=min_distance, threshold_abs=threshold)
    mask = np.zeros(field.shape, dtype=bool)
    mask[tuple(peaks.T)] = True
    # use watershed method, fills adjacent cells so they touch at watershed line
    above_threshold = abs(field) >= threshold
    markers, _ = ndimage.label(mask, structure=np.ones((3,3), dtype="int"))
    labeled = watershed(field, markers, mask=above_threshold)

    # dilate labels
    if aura:
        labeled = expand_labels(labeled, distance=aura)

    return labeled, above_threshold



def label_above_threshold(field, threshold, sub_threshold, min_area):
    """
    labels above-threshold and sufficiently large areas ; option to dilate labels.
    to capture small mesocyclones, finds cohesive IUH regions above sub_threshold which contains at least one pixel above threshold.
    for sub_threshold = threshold, this is equivalent to searching for an uniform minimum threshold value.
    
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
    labeled, _ = ndimage.label(above_threshold, structure=np.ones((3,3)))
    #labeled = watershed(field, markers, mask=above_threshold)
    
    #get rid of too smal areas as well as areas which don't peak at threshold
    labels = np.unique(labeled).tolist()
    labels = [i for i in labels if i != 0]  # remove zero, corresponding to the background
    
    for label in labels:
        count = np.count_nonzero(labeled == label)
        max_value = np.max(abs(field[labeled == label]))
        if count < min_area or max_value < threshold:
            labeled[labeled == label] = 0
    
    return labeled


def find_duplicates(lst):
    """
    finds duplicates in a list, ie elements that appear at least twice
    
    in
    lst: a list of elements, 1D array
    
    out
    duplicates: a list containing the duplicates
    """
    
    lst = np.array(lst)
    duplicates = []
    for el in lst:
        count = np.count_nonzero(lst == el)
        if count > 1:
            duplicates.append(el)
    
    return np.unique(duplicates)


def find_indices(lst, element):
    """
    finds all location indices of a given element within a list
    
    in
    lst: a list of elements, list
    element: the element of focus in the list, int
    
    out
    indices: a list containing location indices of element, list
    """
    
    indices = []
    current_index = -1

    try:
        while True:
            current_index = lst.index(element, current_index + 1)
            indices.append(current_index)
    except ValueError:
        pass

    return indices



def find_overlaps(field, labeled, rain_mask, aura=0, printout=False):
    """
    finds overlaps between supercells and rain cells for SC -> rain cell association
    outputs SCs to rain cells association with overlap, SC signature, max value, mean value and area
    
    in
    labeled: labeled supercell areas (label number arbitrary, 0 is background), 2D array
    rain_mask: labeled rain cells (nan is background, cell labels start at 0), 2D array
    aura: number of gridpoints to dilate labels (post-dilation, called after min area is invoked !), int
    
    out
    overlaps: a set containing 6 columns assigning each supercell, classified according to its signature (RM of LM), min value, mean value, area,
              and assigned to a(several) rain cell(s) with their corresponding overlap (in gp), set
    """
    
    labels_SC = np.unique(labeled).tolist() # get a list of the SCs label numbers
    labels_SC = [i for i in labels_SC if i != 0] # remove zero, which corresponds to the background
    #cell_ids = np.unique(rain_mask)[:-1] # get a list of the cell ids and remove nans, corresponding to background
    
    #dilate labels
    if aura:
        labeled_dil = expand_labels(labeled, distance=aura)
    else:
        labeled_dil = labeled
    
    #determine the overlap between supercells and rain cells
    overlaps = {"cell_id": [], "overlap": [], "signature": [], "area": [], "max_val": [], "mean_val":[]}
    
    for SC_label in labels_SC:
        ovl_matrix = np.logical_and(labeled_dil == SC_label, np.logical_not(np.isnan(rain_mask))) #use the dilated labels for overlap
        corr_cell_id = np.unique(rain_mask[np.where(ovl_matrix)])
        corr_cell_id = corr_cell_id[~np.isnan(corr_cell_id)].astype(int) #remove nans and convert from float to int type
        ovl = np.count_nonzero(ovl_matrix)
        
        mean_val = round(np.mean(field[labeled == SC_label]), 1)
        sgn = np.sign(mean_val).astype(int)
        max_val = round(np.max(abs(field[labeled == SC_label])), 1)
        area = np.count_nonzero(labeled == SC_label)
        
        # if the current SC overlaps with an unique rain cell, append the corresponding cell id and overlap area
        if ovl > 0 and len(corr_cell_id) == 1:
            overlaps["cell_id"].append(int(corr_cell_id))
            overlaps["overlap"].append(int(ovl))
            overlaps["signature"].append(int(sgn))
            overlaps["mean_val"].append(mean_val)
            overlaps["max_val"].append(sgn*max_val)
            overlaps["area"].append(int(area))
        elif ovl > 0 and len(corr_cell_id) > 1:
            sub_ovl = [] #by construction at least one pixel of the Sc must overlap with each of the the rain cells
            for cell_id in corr_cell_id:
                sub_ovl_matrix = np.logical_and(labeled_dil == SC_label, rain_mask == cell_id)
                sub_ovl.append(int(np.count_nonzero(sub_ovl_matrix)))
            overlaps["cell_id"].append(corr_cell_id.tolist())
            overlaps["overlap"].append(sub_ovl) #this time the new cell_id and overlap elements are both vectors
            overlaps["signature"].append(int(sgn))
            overlaps["mean_val"].append(mean_val)
            overlaps["max_val"].append(sgn*max_val)
            overlaps["area"].append(int(area))
    
    # check that all detected supercells is assigned to a rain cell
    # missed_SC = [sc for sc in labels_SC if sc not in overlaps["SC_label"]]
    # if len(missed_SC) > 0 and printout:
    #     print(str(len(missed_SC)) + " supercell(s) not assigned to an overlapping rain cell.")
    
    if printout:
        print(overlaps)
        
    # in case of duplicates (SCs overlapping with several rain cells), assign the SC to the largest overlapping rain cell
    # SC_duplicates = find_duplicates(overlaps["SC_label"])
    # if len(SC_duplicates) > 0:
    #     for sc in SC_duplicates:
    #         bool_indices = np.array(overlaps["SC_label"]) == sc
    #         corr_cell = np.array(overlaps["cell_id"])[bool_indices][np.argmax(np.array(overlaps["overlap"])[bool_indices])]
    #         indices = find_indices(overlaps["SC_label"], sc)
    #         # remove the SCs, cell_ids and overlaps for the duplicate, before assigning the duplicate SC to the max. overlap. cell_id
    #         overlaps["SC_label"] = [SC for index, SC in enumerate(overlaps["SC_label"]) if index not in indices]
    #         overlaps["cell_id"] = [cid for index, cid in enumerate(overlaps["cell_id"]) if index not in indices]
    #         overlaps["overlap"] = [ovl for index, ovl in enumerate(overlaps["overlap"]) if index not in indices]
    #         overlaps["SC_label"].append(sc)
    #         overlaps["cell_id"].append(corr_cell)
    #         overlaps["overlap"].append(np.max(np.array(overlaps["overlap"])[bool_indices]))
            
    return overlaps



class SuperCell:
    """
    supercell class with atributes for each cell
    """

    def __init__(self, rain_cell_id, nowdate, signature, area, max_val, mean_val, overlap):
        """
        inizialize cell object

        in
        cell_id: rain cell id, unique, int
        nowdate: current date, datetime
        signature: sign of the supercell, +1 or -1, int
        """
        self.rain_cell_id = rain_cell_id # ID of the rain cell the supercell was assigned to
        self.datelist = [] # is a vector of length the number of integer hours the cell was assigned supercells
        self.datelist.append(nowdate)
        self.signature = [] #vector of the same length as datelist on the first axis, and of the number of supercells on the second (at each time step)
        self.signature.append(signature)
        self.area = []
        self.area.append(area)
        self.max_val = [] # same
        self.max_val.append(max_val)
        self.mean_val = [] # same
        self.mean_val.append(mean_val)
        self.overlap = [] # overlap with the supercells (same for the size)
        self.overlap.append(overlap)
        self.min_lifespan = None


    def post_processing(self):
        """
        post process cells
        used on calculations that are performed after tracking
        """

        self.min_lifespan = len(np.unique(self.datelist)) - 1
        
    
    def append_candidate(self, nowdate, signature, area, max_val, mean_val, overlap):
        """
        appends the newly detected supercell with the provided attributes to the self cell
        """
        self.datelist.append(nowdate)
        self.signature.append(signature)
        self.area.append(area)
        self.max_val.append(max_val)
        self.mean_val.append(mean_val)
        self.overlap.append(overlap)


    # def append_associates(self, cells, field_static=None):
    #     """
    #     appends all associated cells to self cell

    #     in
    #     cells: list of cell objects, list
    #     field_static: static field, dict
    #     """
    #     cell_ids = [cell.cell_id for cell in cells]

    #     if self.parent is not None:
    #         if self.parent in cell_ids:
    #             self.append_cell(cells[self.parent], field_static)

    #     if self.child is not None:
    #         for child in self.child:
    #             if child in cell_ids:
    #                 self.append_cell(cells[child], field_static)

    #     if self.merged_to is not None:
    #         if self.merged_to in cell_ids:
    #             self.append_cell(cells[self.merged_to], field_static)

    # def append_cell(self, cell_to_append, field_static=None):
    #     """
    #     appends cell_to_append to self cell
    #     used to append short lived cells to parent
    #     todo: delta_x/y not recalculated

    #     in
    #     cell_to_append: cell to append, Cell
    #     field_static: static field, dict
    #     """

    #     for i, nowdate in enumerate(cell_to_append.datelist):
    #         # both cells exist simultaneously

    #         if nowdate in self.datelist:
    #             index = self.datelist.index(nowdate)

    #             if self.field is not None:
    #                 self.field[index] = np.append(
    #                     self.field[index], cell_to_append.field[i], axis=0
    #                 )

    #                 self.field[index] = np.unique(self.field[index], axis=0)

    #                 mass_center = np.mean(self.field[index], axis=0)

    #                 self.mass_center_x[index] = mass_center[0]
    #                 self.mass_center_y[index] = mass_center[1]

    #                 self.area_gp[index] = np.shape(self.field[index])[
    #                     0
    #                 ]  # area in gridpoints

    #             else:
    #                 # weighted mean approximation
    #                 self.mass_center_x[index] = cell_to_append.mass_center_x[
    #                     i
    #                 ] * cell_to_append.area_gp[i] / (
    #                     cell_to_append.area_gp[i] + self.area_gp[index]
    #                 ) + self.mass_center_x[
    #                     index
    #                 ] * self.area_gp[
    #                     index
    #                 ] / (
    #                     cell_to_append.area_gp[i] + self.area_gp[index]
    #                 )
    #                 self.mass_center_y[index] = cell_to_append.mass_center_y[
    #                     i
    #                 ] * cell_to_append.area_gp[i] / (
    #                     cell_to_append.area_gp[i] + self.area_gp[index]
    #                 ) + self.mass_center_y[
    #                     index
    #                 ] * self.area_gp[
    #                     index
    #                 ] / (
    #                     cell_to_append.area_gp[i] + self.area_gp[index]
    #                 )

    #                 # first or last timestep of cell_to_append has been added by add_split_timestep or add_merged_timestep so it doesnt need to be added.
    #                 if i != 0 and i != len(cell_to_append.datelist) - 1:
    #                     self.area_gp[index] += cell_to_append.area_gp[i]

    #                 # todo: update width and length

    #             if self.max_val[index] < cell_to_append.max_val[i]:
    #                 self.max_val[index] = cell_to_append.max_val[i]
    #                 self.max_x[index] = cell_to_append.max_x[i]
    #                 self.max_y[index] = cell_to_append.max_y[i]

    #             if field_static is not None:
    #                 self.lon[index] = field_static["lon"][
    #                     int(np.round(self.mass_center_x[index])),
    #                     int(np.round(self.mass_center_y[index])),
    #                 ]
    #                 self.lat[index] = field_static["lat"][
    #                     int(np.round(self.mass_center_x[index])),
    #                     int(np.round(self.mass_center_y[index])),
    #                 ]

    #         elif nowdate > self.datelist[-1]:
    #             index = -1
    #             if self.field is not None:
    #                 self.field.append(cell_to_append.field[i])

    #             self.datelist.append(nowdate)

    #             self.mass_center_x.append(cell_to_append.mass_center_x[i])
    #             self.mass_center_y.append(cell_to_append.mass_center_y[i])

    #             self.delta_x.append(cell_to_append.delta_x[i])
    #             self.delta_y.append(cell_to_append.delta_y[i])

    #             # area in gridpoints
    #             self.area_gp.append(cell_to_append.area_gp[i])
    #             # self.width_gp.append(cell_to_append.width_gp[i])
    #             # self.length_gp.append(cell_to_append.length_gp[i])
    #             # maximum value of field
    #             self.max_val.append(cell_to_append.max_val[i])

    #             self.max_x.append(cell_to_append.max_x[i])
    #             self.max_y.append(cell_to_append.max_y[i])

    #             self.lon.append(cell_to_append.lon[i])
    #             self.lat.append(cell_to_append.lat[i])

    #             self.label.append(-1)  # nan representation
    #             self.score.append(-1)  # nan representation
    #             self.overlap.append(-1)  # nan representation
    #             self.search_field.append(None)
    #             self.search_vector.append(None)

    #             self.lifespan = self.datelist[-1] - self.datelist[0]

    #         elif nowdate < self.datelist[0]:
    #             index = -1
    #             if self.field is not None:
    #                 self.field.insert(0, cell_to_append.field[i])
    #             self.datelist.insert(0, nowdate)

    #             self.mass_center_x.insert(0, cell_to_append.mass_center_x[i])
    #             self.mass_center_y.insert(0, cell_to_append.mass_center_y[i])

    #             self.delta_x.insert(0, cell_to_append.delta_x[i])
    #             self.delta_y.insert(0, cell_to_append.delta_y[i])

    #             # area in gridpoints
    #             self.area_gp.insert(0, cell_to_append.area_gp[i])
    #             # self.width_gp.insert(0, cell_to_append.width_gp[i])
    #             # self.length_gp.insert(0, cell_to_append.length_gp[i])
    #             self.max_val.insert(
    #                 0, cell_to_append.max_val[i]
    #             )  # maximum value of field

    #             self.max_x.insert(0, cell_to_append.max_x[i])
    #             self.max_y.insert(0, cell_to_append.max_y[i])

    #             self.lon.insert(0, cell_to_append.lon[i])
    #             self.lat.insert(0, cell_to_append.lat[i])

    #             self.label.insert(0, -1)  # nan representation
    #             self.score.insert(0, -1)  # nan representation
    #             self.overlap.insert(0, -1)  # na"label": "arbitrary supercell label",
    #        n representation
    #             self.search_field.insert(0, None)
    #             self.search_vector.insert(0, None)

    #             self.lifespan = self.datelist[-1] - self.datelist[0]

    #         else:
    #             print("other appending error")

    # def copy(self):
    #     """
    #     returns a copy of the cell object
    #     """
    #     return copy.deepcopy(self)

    def get_human_str(self):
        """
        generates human readable string of cell object for # logging

        out
        outstr: human readable string of cell object, string
        """
        outstr = "-" * 100 + "\n"
        outstr += f"date time:    {self.datelist}\n"
        outstr += f"signature:    {self.signature}\n"
        outstr += f"rain cell id:  {str(self.rain_cell_id).zfill(3)}\n"
        outstr += f"area in grid points:    {self.area}\n"
        outstr += f"maximum value:    {self.max_val}\n"
        outstr += f"mean value:    {self.mean_val}\n"
        outstr += f"overlap with rain cell:    {self.overlap}\n"
        outstr += f"minimum lifespan:    {self.min_lifespan}\n"

        outstr += "\n\n"

        return outstr


    def to_dict(self):
        """
        returns a dictionary containing all cell object information

        out
        cell_dict: dictionary containing all cell object information, dict
        """
        cell_dict = {
            "rain_cell_id": self.rain_cell_id,
            "datelist": [str(t)[:16] for t in self.datelist],
            "signature": self.signature,
            "area": self.area, #[int(x) for x in self.area],
            "max_val": self.max_val,
            "mean_val": self.mean_val,
            "overlap": self.overlap,
            "min_lifespan": self.min_lifespan
        }
        return cell_dict

    # def from_dict(self, cell_dict):
    #     """
    #     returns a cell object from a dictionary containing all cell object information

    #     in
    #     cell_dict: dictionary containing all cell object information, dict
    #     """
    #     self.cell_id = cell_dict["cell_id"]
    #     self.parent = cell_dict["parent"]
    #     self.child = cell_dict["child"]
    #     self.merged_to = cell_dict["merged_to"]
    #     self.died_of = cell_dict["died_of"]
    #     self.lifespan = cell_dict["lifespan"]
    #     self.datelist = [np.datetime64(t) for t in cell_dict["datelist"]]
    #     self.lon = cell_dict["lon"]
    #     self.lat = cell_dict["lat"]
    #     self.mass_center_x = cell_dict["mass_center_x"]
    #     self.mass_center_y = cell_dict["mass_center_y"]
    #     self.max_x = cell_dict["max_x"]
    #     self.max_y = cell_dict["max_y"]
    #     self.delta_x = cell_dict["delta_x"]
    #     self.delta_y = cell_dict["delta_y"]
    #     self.area_gp = cell_dict["area_gp"]
    #     # self.width_gp = cell_dict["width_gp"]
    #     # self.length_gp = cell_dict["length_gp"]
    #     self.max_val = cell_dict["max_val"]
    #     self.score = cell_dict["score"]

    #     self.overlap = [0] * len(self.datelist)
    #     self.search_field = [None] * len(self.datelist)
    #     self.search_vector = [[0, 0]] * len(self.datelist)
    #     self.alive = False
    #     self.field = None
    #     self.label = []
    #     self.swath = []



def write_to_json(cells, filename):
    """
    writes ascii file containing supercell object information

    in
    cells: list of supercells, list
    filename: filename, string

    out
    struct: dictionary containing all cell object information, dict
    """

    # figure out wether input is list or empty
    if len(cells) == 0:
        struct = []
        data_structure = "no supercells found"

    elif isinstance(cells[0], SuperCell):  # single member
        struct = [cell.to_dict() for cell in cells]
        data_structure = "SC_data contains list of supercells, each supercell contains a dictionary of supercell parameters"

    else:
        print(
            "input is neither list of supercells nor empty list, each supercell contains a a dictionary of supercell parameters"
        )
        return

    struct = {
        "info": "supercell track data generated using SDT",
        "author": "Michael Blanc (mblanc@student.ethz.ch)",
        "data_structure": data_structure,
        "parameters": {
            "rain_cell_id": "id(s) of the rain cell(s) the supercell was assigned to",
            "datelist": "list of datetimes for each timestep in supercell lifetime",
            "signature": "vorticity signature of the supercell (+1 for mesocyclones and -1 for mesoanticyclones)",
            "area": "area in gridpoints",
            "max_val": "maximum IUH value within the supercell",
            "mean_val": "mean IUH value within the supercell",
            "overlap": "overlap in gridpoints between the (dilated) supercell and the rain cell(s)",
            "min_lifespan": "minimum lifespan of supercell in hours",
        },
        "SC_data": struct,
    }

    with open(filename, "w") as f:
        json.dump(struct, f)