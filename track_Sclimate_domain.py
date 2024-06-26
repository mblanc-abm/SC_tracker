"""
version intended for the whole domain
takes the starting and ending dates of the considered period as inputs
detects and assigns supercells to existing rain cells, saves supercells information into a json file sa well as mesocyclones masks into a netcdf file

input:
    outpath: path to output file, str
    start_day: starting day of the considered period, YYYYMMDD, str
    end_day: ending day of the considered period, YYYYMMDD, str
    climate: whether we are tracking on the current or future climate, "current" or "future", str

output:
    supercells_YYYYMMDD.json: json files containing supercells information
    meso_masks_YYYMMDD.nc: mesocyclones binary masks, netcdf
"""

from Scell_tracker import track_Scells, write_to_json, write_masks_to_netcdf
#import sys
import os
import argparse

import xarray as xr
import pandas as pd
#import numpy as np

#====================================================================================================================

# set the thresholds as default arguments before any domain simulation ! Also specify the output path further down !
def main(climate, start_day, end_day, zeta_th=5e-3, zeta_pk=None, w_th=5, two_meso_detections=False, min_area=3, aura=1):
    
    start_day = pd.to_datetime(start_day, format="%Y%m%d")
    end_day = pd.to_datetime(end_day, format="%Y%m%d")
    daylist = pd.date_range(start_day, end_day)
    
    # prepare hourly wind and surface pressure files path, and 5 min surface hail files path
    inpath = "/scratch/snx3000/mblanc/SDT/infiles/domain" # same path whatever the climate
    path_p = os.path.join(inpath, "1h_3D_plev")
    path_s = os.path.join(inpath, "1h_2D")
    path_h = os.path.join(inpath, "5min_2D")

    # make output directory
    outpath = "/scratch/snx3000/mblanc/SDT/SDT2_output/" + climate + "/domain/XPT_1MD_zetath5_wth5"
    os.makedirs(outpath, exist_ok=True)
    
    for i, day in enumerate(daylist):
        # print progress
        print("processing day ", i + 1, " of ", len(daylist))
        print("processing day: ", day.strftime("%Y/%m/%d"))
        # get the data for one day
        print("preparing data")
        
        # make timesteps: extract hourly timesteps from 4am of the current day to 3am of the subsequent day
        mask_path = "/scratch/snx3000/mblanc/CT2/" + climate + "/outfiles/" + "cell_masks_" + day.strftime("%Y%m%d") + ".nc"
        with xr.open_dataset(mask_path) as dset:
            times_5min = pd.to_datetime(dset['time'].values)
        timesteps = [t for t in times_5min if t.strftime("%M")=="00"]
    
        # make 1h_3D (pressure level), 1h_2D (surface pressure and max surface wind) filenames, and 5min_2D (max hail) path
        fnames_p = []
        fnames_s = []
        for dt in timesteps:
            fnames_p.append(path_p + "/lffd" + dt.strftime("%Y%m%d%H%M%S") + "p.nc")
            fnames_s.append(path_s + "/lffd" + dt.strftime("%Y%m%d%H%M%S") + ".nc")
        
        # make rain mask and tracks file names
        rain_masks_name = mask_path  
        #rain_tracks_name = "/project/pr133/mblanc/cell_tracker_output/" + climate + "_climate/cell_tracks_" + day.strftime("%Y%m%d") + ".json"
        rain_tracks_name = "/scratch/snx3000/mblanc/CT2/" + climate + "/outfiles/" + "cell_tracks_" + day.strftime("%Y%m%d") + ".json"
        
        # track supercells
        print("tracking supercells")
        supercells, na_vorticies, lons, lats = track_Scells(day, timesteps, fnames_p, fnames_s, path_h, rain_masks_name, rain_tracks_name, zeta_th,
                                                            zeta_pk, w_th, min_area, aura, two_meso_detections)
        
        print("writing data to file")
        outfile_json = os.path.join(outpath, "supercell_" + day.strftime("%Y%m%d") + ".json")
        outfile_nc = os.path.join(outpath, "meso_masks_" + day.strftime("%Y%m%d") + ".nc")
        write_to_json(supercells, na_vorticies, outfile_json)
        write_masks_to_netcdf(supercells, lons, lats, outfile_nc)
    
    print("finished tracking all days in queue")
    
    return

#====================================================================================================================

# climate = "current_climate"
# start_day = "20181031"
# end_day = "20181031"

# main(climate, start_day, end_day)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("climate", type=str, help="climate")
    p.add_argument("start_day", type=str, help="start day")
    p.add_argument("end_day", type=str, help="end day")
    args = p.parse_args()

    main(**vars(args))
