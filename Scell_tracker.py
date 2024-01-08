import json
import numpy as np
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

def track_Scells(fnames_p, fnames_s, rain_masks, rain_tracks, threshold, sub_threshold, min_area, aura):
    """
    description to be written

    Parameters
    ----------
    fnames_p : list of str
        complete paths to the 3D pressure levels files containing the wind fields, for IUH
    fnames_s : list of str
        complete paths to the 2D surface pressure files, for IUH
    rain_masks : array
        labeled rain cells (nan is background, cell labels start at 0), 2D arrays, concatenated with a 5 min resolution
    rain_tracks : json
        output json file of the cell tracker algorithm 
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
    Scells:
        list of SuperCell objects
    """
    
    



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



def label_above_threshold(field, threshold, sub_threshold, min_area, aura=0):
    """
    labels above-threshold and sufficiently large areas ; option to dilate labels
    to capture small mesocyclones, finds cohesive IUH regions above sub_threshold which contains at least one pixel above threshold
    for sub_threshold = threshold, this is equivalent to searching for an uniform minimum threshold value
    
    in
    field: 2D IUH field, array
    threshold: threshold for minimum peak value considered as within cell, float
    sub_theshold: threshold for minimum value considered as within cell, float
    min_area: minimum area (in gridpoints) for a cell to be considered, int
    aura: number of gridpoints to dilate labels (post-dilation, called after min area is invoked !), int

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

    #dilate labels
    if aura:
        labeled = expand_labels(labeled, distance=aura)
    
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



def find_overlaps(field, labeled, rain_mask, printout=False):
    """
    finds overlaps between supercells and rain cells for SC -> rain cell association
    
    in
    labeled: labeled supercell areas (label number arbitrary, 0 is background), 2D array
    rain_mask: labeled rain cells (nan is background, cell labels start at 0), 2D array
    
    out
    overlaps: a set containing 4 columns assigning each supercell, classified according to its signature (RM of LM), to a(several) rain cell(s) with their corresponding overlap (in gp), set
    missed_SC: number of supercells not assigned to a rain cell, int. for aura large enough, simply due to the abscence of a rain cell at the SC location
    """
    
    labels_SC = np.unique(labeled).tolist() # get a list of the SCs label numbers
    labels_SC = [i for i in labels_SC if i != 0] # remove zero, which corresponds to the background
    #cell_ids = np.unique(rain_mask)[:-1] # get a list of the cell ids and remove nans, corresponding to background
    
    #determine the overlap between supercells and rain cells
    overlaps = {"SC_label": [], "signature": [], "cell_id": [], "overlap": []}
    for SC_label in labels_SC:
        ovl_matrix = np.logical_and(labeled == SC_label, np.logical_not(np.isnan(rain_mask)))
        corr_cell_id = np.unique(rain_mask[np.where(ovl_matrix)])
        corr_cell_id = corr_cell_id[~np.isnan(corr_cell_id)].astype(int) #remove nans and convert from float to int type
        ovl = np.count_nonzero(ovl_matrix)
        sgn = np.sign(np.nanmean(field[labeled == SC_label])).astype(int)
        
        # if the current SC overlaps with an unique rain cell, append the corresponding cell id and overlap area
        if ovl > 0 and len(corr_cell_id) == 1: 
            overlaps["SC_label"].append(SC_label)
            overlaps["signature"].append(sgn)
            overlaps["cell_id"].append(corr_cell_id[0])
            overlaps["overlap"].append(ovl)
        elif ovl > 0 and len(corr_cell_id) > 1:
            sub_ovl = [] #by construction at least one pixel of the Sc must overlap with each of the the rain cells
            for cell_id in corr_cell_id:
                sub_ovl_matrix = np.logical_and(labeled == SC_label, rain_mask == cell_id)
                sub_ovl.append(np.count_nonzero(sub_ovl_matrix))
            overlaps["SC_label"].append(SC_label)
            overlaps["signature"].append(sgn)
            overlaps["cell_id"].append(corr_cell_id.tolist())
            overlaps["overlap"].append(sub_ovl) #this time the new cell_id and overlap elements are both vectors
    
    # check that all detected supercells is assigned to a rain cell
    missed_SC = [sc for sc in labels_SC if sc not in overlaps["SC_label"]]
    if len(missed_SC) > 0 and printout:
        print(str(len(missed_SC)) + " supercell(s) not assigned to an overlapping rain cell. You may want to dilate the supercells.")
    
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
            
    return overlaps, len(missed_SC)



class SuperCell:
    """
    supercell class with atributes for each cell
    """

    def __init__(self, cell_id: int, label: int, nowdate):
        """
        inizialize cell object

        in
        cell_id: cell id, int
        label: label of cell, int
        nowdate: current date, datetime
        parent: parent cell id, int
        """
        self.cell_id = cell_id # ID of the rain cell
        self.label = [] #
        self.label.append(label)
        self.datelist = []
        self.datelist.append(nowdate)
        self.alive = True
        self.parent = parent
        self.child = []
        self.merged_to = None

        self.mass_center_x = []  # list of x positions of mass center
        self.mass_center_y = []  # list of y positions of mass center

        self.delta_x = []  # movement vector of cell mass center
        self.delta_y = []  # movement vector of cell mass center

        self.lon = []
        self.lat = []

        self.area_gp = []  # area in gridpoints
        # self.width_gp = []  # width of cell footprint in gridpoints
        # self.length_gp = []  # length of cell footprint in gridpoints
        self.max_val = []
        self.max_x = []
        self.max_y = []

        self.lifespan = None
        self.score = [0]  # first assignment doesn't have a score
        self.overlap = [0]  # first assignment doesn't have a overlap

        self.died_of = None

        self.field = []  # list of coordinates that are inside the cell mask

        self.swath = None

        # field of area where cell is searched for (extrapolated from last step, first is None)
        self.search_field = [None]
        self.search_vector = [[0, 0]]

    def __str__(self):
        """
        default string representation of Cell object
        """
        return str(self.cell_id)

    def cal_spatial(self, field_static, coordinates, values):
        """
        calculate spatial stats (to be applied per timestep)
        used on calculations that rely on the field and labeled arrays only available
        at runtime

        in
        field_static: static field, dict
        coordinates: coordinates that lie within cell, array
        values: values of cell, array
        """
        self.field.append(coordinates)

        # max_pos = np.unravel_index(masked.argmax(),masked.shape)
        area_center = np.mean(coordinates, axis=0)
        max_pos = coordinates[np.argmax(values)]

        mass_center = np.mean((area_center, max_pos), 0)

        # max_pos = ndimage.measurements.maximum_position(masked)
        # mass_center = max_pos
        # mass_center = ndimage.measurements.center_of_mass(field,labeled,self.label[-1])

        self.mass_center_x.append(mass_center[0])
        self.mass_center_y.append(mass_center[1])

        if len(self.mass_center_x) < 2:
            self.delta_x.append(0)
            self.delta_y.append(0)

            # self.width_gp.append(0)
            # self.length_gp.append(0)
        else:
            self.delta_x.append(self.mass_center_x[-1] - self.mass_center_x[-2])
            self.delta_y.append(self.mass_center_y[-1] - self.mass_center_y[-2])

            # get width and length relative to movement vector
            # rotate the cell mask
            # angle = np.arctan2(self.delta_y[-1], self.delta_x[-1]) * 180 / np.pi

            # mask_size = np.max(coordinates, axis=0) - np.min(coordinates, axis=0) + 1

            # mask = np.zeros(mask_size)
            # mask[
            #     coordinates[:, 0] - np.min(coordinates, axis=0)[0],
            #     coordinates[:, 1] - np.min(coordinates, axis=0)[1],
            # ] = 1

            # cell_mask_r = rotate(mask, angle)

            # # threshold the rotated cell mask
            # cell_mask_r[cell_mask_r < 0.5] = 0
            # cell_mask_r[cell_mask_r > 0.5] = 1

            # width_gp = np.where(cell_mask_r)[1].max() - np.where(cell_mask_r)[1].min()
            # length_gp = np.where(cell_mask_r)[0].max() - np.where(cell_mask_r)[0].min()

            # self.width_gp.append(width_gp)
            # self.length_gp.append(length_gp)

        # self.area_gp.append(np.count_nonzero(masked)) # area in gridpoints
        self.max_val.append(np.max(values))  # maximum value of field

        self.max_x.append(max_pos[0])
        self.max_y.append(max_pos[1])

        self.lon.append(
            field_static["lon"][
                int(np.round(self.mass_center_x[-1])),
                int(np.round(self.mass_center_y[-1])),
            ]
        )
        self.lat.append(
            field_static["lat"][
                int(np.round(self.mass_center_x[-1])),
                int(np.round(self.mass_center_y[-1])),
            ]
        )

    def is_in_box(self, lat_lim, lon_lim):
        """
        check if any part of cell is in lat lon box

        in
        lat_lim: latitude limits, list
        lon_lim: longitude limits, list
        """

        if np.any(
            np.logical_and(
                np.array(self.lon) > lon_lim[0], np.array(self.lon) < lon_lim[1]
            )
        ):
            if np.any(
                np.logical_and(
                    np.array(self.lat) > lat_lim[0], np.array(self.lat) < lat_lim[1]
                )
            ):
                return True
        return False

    def post_processing(self):
        """
        post process cells
        used on calculations that are performed after tracking
        """

        self.lifespan = self.datelist[-1] - self.datelist[0]

    def insert_split_timestep(self, parent):
        """
        insert timestep from parent cell

        in
        parent: parent cell, Cell
        """

        if self.datelist[-1] not in parent.datelist: 
            # logging.warning(
            #     "parent datelist does not contain self datelist[-1], could not append split member"
            # )
            return

        split_idx = parent.datelist.index(self.datelist[0]) - 1
        self.label.insert(0, parent.label[split_idx])

        self.datelist.insert(0, parent.datelist[split_idx])

        self.mass_center_x.insert(0, parent.mass_center_x[split_idx])
        self.mass_center_y.insert(0, parent.mass_center_y[split_idx])

        self.delta_x.insert(0, parent.delta_x[split_idx])
        self.delta_y.insert(0, parent.delta_y[split_idx])

        self.lon.insert(0, parent.lon[split_idx])
        self.lat.insert(0, parent.lat[split_idx])

        self.area_gp.insert(0, parent.area_gp[split_idx])
        # self.width_gp.insert(0, parent.width_gp[split_idx])
        # self.length_gp.insert(0, parent.length_gp[split_idx])
        self.max_val.insert(0, parent.max_val[split_idx])
        self.max_x.insert(0, parent.max_x[split_idx])
        self.max_y.insert(0, parent.max_y[split_idx])

        self.score.insert(0, parent.score[split_idx])
        self.overlap.insert(0, parent.overlap[split_idx])

        self.field.insert(0, parent.field[split_idx])

        self.search_field.insert(0, None)
        self.search_vector.insert(0, [0, 0])

    def insert_merged_timestep(self, parent):
        """
        insert timestep from merged_to cell

        in
        parent: parent cell, Cell
        """
        # logging.info(
        # "inserting merged timestep from "
        #     + str(parent.cell_id)
        #     + " to "
        #     + str(self.cell_id)
        # )

        if self.datelist[-1] not in parent.datelist:
            # logging.warning(
            #     "parent datelist does not contain self datelist[-1], could not append merge member"
            # )
            return

        merge_idx = parent.datelist.index(self.datelist[-1]) #+ 1 # todo    isn't it + 1 ?
        self.label.append(parent.label[merge_idx])  # todo

        self.datelist.append(parent.datelist[merge_idx])

        self.mass_center_x.append(parent.mass_center_x[merge_idx])
        self.mass_center_y.append(parent.mass_center_y[merge_idx])

        self.delta_x.append(parent.delta_x[merge_idx])
        self.delta_y.append(parent.delta_y[merge_idx])

        self.lon.append(parent.lon[merge_idx])
        self.lat.append(parent.lat[merge_idx])

        self.area_gp.append(parent.area_gp[merge_idx])
        # self.width_gp.append(parent.width_gp[merge_idx])
        # self.length_gp.append(parent.length_gp[merge_idx])
        self.max_val.append(parent.max_val[merge_idx])
        self.max_x.append(parent.max_x[merge_idx])
        self.max_y.append(parent.max_y[merge_idx])

        self.score.append(parent.score[merge_idx])
        self.overlap.append(parent.overlap[merge_idx])

        self.field.append(parent.field[merge_idx])

        self.search_field.append(None)
        self.search_vector.append([0, 0])

    def terminate(self, reason=None):
        """
        terminates cell

        in
        reason: reason for termination, string
        """
        if self.alive:
            self.died_of = reason
            self.alive = False
            # logging.info("cell %d was terminated, reason is: %s", self.cell_id, reason)
        else:
            pass
            # logging.warning("cell %d is already dead!", self.cell_id)

    def print_summary(self):
        """
        diagnostics function
        """
        print(
            "cell no.",
            str(self.cell_id).zfill(3),
            " started at",
            np.round(self.lat[0], 2),
            "N",
            np.round(self.lon[0], 2),
            "E",
            self.datelist[0],
            "lasted",
            str(self.lifespan)[0:-3],
            "h",
        )
        print(
            "              max_area:",
            max(self.area_gp),
            "max_val:",
            round(np.max(self.max_val), 2),
        )
        print(
            "              parent:",
            self.parent,
            "childs:",
            self.child,
            "merged to:",
            self.merged_to,
        )

    def add_aura(self, kernel_size, field_static):
        """
        add aura to cell
        used for plotting and analysis

        in
        kernel_size: size of aura, int
        field_static: static field, dict
        """
        if kernel_size == 0:
            return

        kernel = disk(kernel_size, dtype=int) # make_aura_kernel(kernel_size)

        for i, field in enumerate(self.field):
            array = np.zeros_like(field_static["lat"])
            array[field[:, 0], field[:, 1]] = 1
            array = ndimage.binary_dilation(array, structure=kernel)
            self.field[i] = np.argwhere(array == 1)

    def add_genesis(self, n_timesteps, field_static):
        """
        add genesis timesteps to cell
        interpolated backwards in time to where the cell is initially detected
        mainly used for plotting and analysis

        in
        n_timesteps: number of timesteps to add, int
        field_static: static field, dict
        """
        x_0 = self.mass_center_x[0].copy()
        y_0 = self.mass_center_y[0].copy()
        field_0 = self.field[0].copy()

        dx = np.mean(self.delta_x[1:4])
        dy = np.mean(self.delta_y[1:4])

        self.delta_x[0] = dx
        self.delta_y[0] = dy

        for i in range(1, n_timesteps):
            x_offset = int(np.round(dx * i))
            y_offset = int(np.round(dy * i))

            self.delta_x.insert(0, self.delta_x[0])
            self.delta_y.insert(0, self.delta_y[0])

            self.datelist.insert(
                0, self.datelist[0] - (self.datelist[1] - self.datelist[0])
            )

            self.mass_center_x.insert(0, x_0 - x_offset)
            self.mass_center_y.insert(0, y_0 - y_offset)

            new_field = np.zeros_like(field_0)
            new_field[:, 0] = field_0[:, 0] - x_offset
            new_field[:, 1] = field_0[:, 1] - y_offset
            self.field.insert(0, new_field)

            self.lon.insert(
                0,
                field_static["lon"][
                    int(np.round(self.mass_center_x[0])),
                    int(np.round(self.mass_center_y[0])),
                ],
            )
            self.lat.insert(
                0,
                field_static["lat"][
                    int(np.round(self.mass_center_x[0])),
                    int(np.round(self.mass_center_y[0])),
                ],
            )

            self.area_gp.insert(0, self.area_gp[0])
            # self.width_gp.insert(0, self.width_gp[0])
            # self.length_gp.insert(0, self.length_gp[0])

            self.label.insert(0, None)
            self.max_val.insert(0, None)
            self.max_x.insert(0, None)
            self.max_y.insert(0, None)
            self.score.insert(0, None)
            self.overlap.insert(0, None)
            self.search_field.insert(0, None)
            self.search_vector.insert(0, [0, 0])

    def append_associates(self, cells, field_static=None):
        """
        appends all associated cells to self cell

        in
        cells: list of cell objects, list
        field_static: static field, dict
        """
        cell_ids = [cell.cell_id for cell in cells]

        if self.parent is not None:
            if self.parent in cell_ids:
                self.append_cell(cells[self.parent], field_static)

        if self.child is not None:
            for child in self.child:
                if child in cell_ids:
                    self.append_cell(cells[child], field_static)

        if self.merged_to is not None:
            if self.merged_to in cell_ids:
                self.append_cell(cells[self.merged_to], field_static)

    def append_cell(self, cell_to_append, field_static=None):
        """
        appends cell_to_append to self cell
        used to append short lived cells to parent
        todo: delta_x/y not recalculated

        in
        cell_to_append: cell to append, Cell
        field_static: static field, dict
        """

        for i, nowdate in enumerate(cell_to_append.datelist):
            # both cells exist simultaneously

            if nowdate in self.datelist:
                index = self.datelist.index(nowdate)

                if self.field is not None:
                    self.field[index] = np.append(
                        self.field[index], cell_to_append.field[i], axis=0
                    )

                    self.field[index] = np.unique(self.field[index], axis=0)

                    mass_center = np.mean(self.field[index], axis=0)

                    self.mass_center_x[index] = mass_center[0]
                    self.mass_center_y[index] = mass_center[1]

                    self.area_gp[index] = np.shape(self.field[index])[
                        0
                    ]  # area in gridpoints

                else:
                    # weighted mean approximation
                    self.mass_center_x[index] = cell_to_append.mass_center_x[
                        i
                    ] * cell_to_append.area_gp[i] / (
                        cell_to_append.area_gp[i] + self.area_gp[index]
                    ) + self.mass_center_x[
                        index
                    ] * self.area_gp[
                        index
                    ] / (
                        cell_to_append.area_gp[i] + self.area_gp[index]
                    )
                    self.mass_center_y[index] = cell_to_append.mass_center_y[
                        i
                    ] * cell_to_append.area_gp[i] / (
                        cell_to_append.area_gp[i] + self.area_gp[index]
                    ) + self.mass_center_y[
                        index
                    ] * self.area_gp[
                        index
                    ] / (
                        cell_to_append.area_gp[i] + self.area_gp[index]
                    )

                    # first or last timestep of cell_to_append has been added by add_split_timestep or add_merged_timestep so it doesnt need to be added.
                    if i != 0 and i != len(cell_to_append.datelist) - 1:
                        self.area_gp[index] += cell_to_append.area_gp[i]

                    # todo: update width and length

                if self.max_val[index] < cell_to_append.max_val[i]:
                    self.max_val[index] = cell_to_append.max_val[i]
                    self.max_x[index] = cell_to_append.max_x[i]
                    self.max_y[index] = cell_to_append.max_y[i]

                if field_static is not None:
                    self.lon[index] = field_static["lon"][
                        int(np.round(self.mass_center_x[index])),
                        int(np.round(self.mass_center_y[index])),
                    ]
                    self.lat[index] = field_static["lat"][
                        int(np.round(self.mass_center_x[index])),
                        int(np.round(self.mass_center_y[index])),
                    ]

            elif nowdate > self.datelist[-1]:
                index = -1
                if self.field is not None:
                    self.field.append(cell_to_append.field[i])

                self.datelist.append(nowdate)

                self.mass_center_x.append(cell_to_append.mass_center_x[i])
                self.mass_center_y.append(cell_to_append.mass_center_y[i])

                self.delta_x.append(cell_to_append.delta_x[i])
                self.delta_y.append(cell_to_append.delta_y[i])

                # area in gridpoints
                self.area_gp.append(cell_to_append.area_gp[i])
                # self.width_gp.append(cell_to_append.width_gp[i])
                # self.length_gp.append(cell_to_append.length_gp[i])
                # maximum value of field
                self.max_val.append(cell_to_append.max_val[i])

                self.max_x.append(cell_to_append.max_x[i])
                self.max_y.append(cell_to_append.max_y[i])

                self.lon.append(cell_to_append.lon[i])
                self.lat.append(cell_to_append.lat[i])

                self.label.append(-1)  # nan representation
                self.score.append(-1)  # nan representation
                self.overlap.append(-1)  # nan representation
                self.search_field.append(None)
                self.search_vector.append(None)

                self.lifespan = self.datelist[-1] - self.datelist[0]

            elif nowdate < self.datelist[0]:
                index = -1
                if self.field is not None:
                    self.field.insert(0, cell_to_append.field[i])
                self.datelist.insert(0, nowdate)

                self.mass_center_x.insert(0, cell_to_append.mass_center_x[i])
                self.mass_center_y.insert(0, cell_to_append.mass_center_y[i])

                self.delta_x.insert(0, cell_to_append.delta_x[i])
                self.delta_y.insert(0, cell_to_append.delta_y[i])

                # area in gridpoints
                self.area_gp.insert(0, cell_to_append.area_gp[i])
                # self.width_gp.insert(0, cell_to_append.width_gp[i])
                # self.length_gp.insert(0, cell_to_append.length_gp[i])
                self.max_val.insert(
                    0, cell_to_append.max_val[i]
                )  # maximum value of field

                self.max_x.insert(0, cell_to_append.max_x[i])
                self.max_y.insert(0, cell_to_append.max_y[i])

                self.lon.insert(0, cell_to_append.lon[i])
                self.lat.insert(0, cell_to_append.lat[i])

                self.label.insert(0, -1)  # nan representation
                self.score.insert(0, -1)  # nan representation
                self.overlap.insert(0, -1)  # nan representation
                self.search_field.insert(0, None)
                self.search_vector.insert(0, None)

                self.lifespan = self.datelist[-1] - self.datelist[0]

            else:
                print("other appending error")

    def copy(self):
        """
        returns a copy of the cell object
        """
        return copy.deepcopy(self)

    def get_human_str(self):
        """
        generates human readable string of cell object for # logging

        out
        outstr: human readable string of cell object, string
        """
        outstr = "-" * 100 + "\n"
        outstr += f"cell_id:  {str(self.cell_id).zfill(3)}\n"

        outstr += f"start:      {self.datelist[0]}\n"
        outstr += f"end:        {self.datelist[-1]}\n"
        outstr += f"lifespan:   {self.lifespan.astype('timedelta64[m]')}\n"
        outstr += f"parent:     {self.parent}\n"
        outstr += f"childs:     {self.child}\n"
        outstr += f"merged:     {self.merged_to}\n"
        outstr += f"died_of:    {self.died_of}\n"

        outstr += "\n"

        outstr += f"x:          {list(np.round(self.mass_center_x,2))}\n"
        outstr += f"y:          {list(np.round(self.mass_center_y,2))}\n"
        outstr += f"delta_x:    {list(np.round(self.delta_x,2))}\n"
        outstr += f"delta_y:    {list(np.round(self.delta_y,2))}\n"
        outstr += f"lon:        {list(np.round(self.lon,3))}\n"
        outstr += f"lat:        {list(np.round(self.lat,3))}\n"
        outstr += f"area_gp:    {self.area_gp}\n"
        outstr += f"max_val:    {list(np.round(self.max_val,2))}\n"
        outstr += f"score:      {list(np.round(self.score,3))}\n"
        outstr += f"overlap:    {self.overlap}\n"

        outstr += "\n\n"

        return outstr

    def check_consistency(self):
        """
        checks some cell parameters to see if they're cconsistent
        """
        if len(self.datelist) != len(self.mass_center_x):
            # logging.warning("cell %d has inconsistencies", self.cell_id)
            # logging.info("len(self.datelist): %d", len(self.datelist))
            # logging.info("len(self.mass_center_x): %d", len(self.mass_center_x))
            # logging.info("len(self.area_gp): %d", len(self.area_gp))
            # logging.debug(f"self.datelist {self.datelist}")
            # logging.debug(f"self.mass_center_x {self.mass_center_x}")
            # logging.debug(f"self.mass_center_y {self.mass_center_y}")
            # logging.debug(f"self.delta_x {self.delta_x}")
            # # logging.debug(f"self.delta_y {self.delta_y}")
            pass

    def to_dict(self):
        """
        returns a dictionary containing all cell object information

        out
        cell_dict: dictionary containing all cell object information, dict
        """
        cell_dict = {
            "cell_id": self.cell_id,
            "parent": self.parent,
            "child": self.child,
            "merged_to": self.merged_to,
            "died_of": self.died_of,
            "lifespan": self.lifespan / np.timedelta64(1, "m"),
            "datelist": [str(t)[:16] for t in self.datelist],
            "lon": [round(float(x), 4) for x in self.lon],
            "lat": [round(float(x), 4) for x in self.lat],
            "mass_center_x": [round(float(x), 2) for x in self.mass_center_x],
            "mass_center_y": [round(float(x), 2) for x in self.mass_center_y],
            "max_x": [int(x) for x in self.max_x],
            "max_y": [int(x) for x in self.max_y],
            "delta_x": [round(float(x), 2) for x in self.delta_x],
            "delta_y": [round(float(x), 2) for x in self.delta_y],
            "area_gp": [int(x) for x in self.area_gp],
            # "width_gp": [int(x) for x in self.width_gp],
            # "length_gp": [int(x) for x in self.length_gp],
            "max_val": [round(float(x), 2) for x in self.max_val],
            "score": [round(float(x), 2) for x in self.score],
        }
        return cell_dict

    def from_dict(self, cell_dict):
        """
        returns a cell object from a dictionary containing all cell object information

        in
        cell_dict: dictionary containing all cell object information, dict
        """
        self.cell_id = cell_dict["cell_id"]
        self.parent = cell_dict["parent"]
        self.child = cell_dict["child"]
        self.merged_to = cell_dict["merged_to"]
        self.died_of = cell_dict["died_of"]
        self.lifespan = cell_dict["lifespan"]
        self.datelist = [np.datetime64(t) for t in cell_dict["datelist"]]
        self.lon = cell_dict["lon"]
        self.lat = cell_dict["lat"]
        self.mass_center_x = cell_dict["mass_center_x"]
        self.mass_center_y = cell_dict["mass_center_y"]
        self.max_x = cell_dict["max_x"]
        self.max_y = cell_dict["max_y"]
        self.delta_x = cell_dict["delta_x"]
        self.delta_y = cell_dict["delta_y"]
        self.area_gp = cell_dict["area_gp"]
        # self.width_gp = cell_dict["width_gp"]
        # self.length_gp = cell_dict["length_gp"]
        self.max_val = cell_dict["max_val"]
        self.score = cell_dict["score"]

        self.overlap = [0] * len(self.datelist)
        self.search_field = [None] * len(self.datelist)
        self.search_vector = [[0, 0]] * len(self.datelist)
        self.alive = False
        self.field = None
        self.label = []
        self.swath = []
