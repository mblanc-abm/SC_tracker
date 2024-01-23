"""
version intended for the whole domain
takes the date and hours of the considered case study as input
detects and assigns supercells to existing rain cells and saves supercellls information into a json file

input:
    outpath: path to output file, str
    start_day: starting day of the considered period, YYYYMMDD, str
    end_day: ending day of the considered period, YYYYMMDD, str
    climate: whether we are tracking on the current or future climate, "current" or "future", str

output:
    supercells_YYYYMMDD.json: json files containing supercells information
"""

from Scell_tracker import track_Scells, write_to_json, write_masks_to_netcdf
#import sys
import os
#import argparse

import xarray as xr
import pandas as pd
#import numpy as np

#====================================================================================================================

def main(outpath, start_day, end_day, climate):
    
    # set tracking parameters
    threshold = 75
    sub_threshold = 65
    min_area = 3
    aura = 2
    
    start_day = pd.to_datetime(start_day, format="%Y%m%d")
    end_day = pd.to_datetime(end_day, format="%Y%m%d")
    daylist = pd.date_range(start_day, end_day)

    # make output directory
    os.makedirs(outpath, exist_ok=True)
    
    for i, day in enumerate(daylist):
        # print progress
        print("processing day ", i + 1, " of ", len(daylist))
        print("processing day: ", day.strftime("%Y/%m/%d"))
        # get the data for one day
        print("preparing data")
        
        # make timesteps: extract hourly timesteps from 4am to 4am of the subsequent day
        mask_path = "/project/pr133/mblanc/cell_tracker_output/" + climate + "_climate/cell_masks_" + day.strftime("%Y%m%d") + ".nc"
        with xr.open_dataset(mask_path) as dset:
            times_5min = pd.to_datetime(dset['time'].values)
        timesteps = [t for t in times_5min if t.strftime("%M")=="00"]
    
        # make 1h_3D (pressure level), 1h_2D (surface pressure, hail amd max surface wind) files names
        fnames_p = []
        fnames_s = [] 
        if climate == "current":
            path_p = "/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_3D_plev/lffd"
            path_s = "/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_2D/lffd"
        else:
            path_p = "/project/pr133/irist/scClim/RUN_2km_cosmo6_climate_PGW_MPI_HR/output/lm_f/1h_3D_plev/lffd"
            path_s = "/project/pr133/irist/scClim/RUN_2km_cosmo6_climate_PGW_MPI_HR/output/lm_f/1h_2D/lffd"
        
        for dt in timesteps:
            fnames_p.append(path_p + dt.strftime("%Y%m%d%H%M%S") + "p.nc")
            fnames_s.append(path_s + dt.strftime("%Y%m%d%H%M%S") + ".nc")
        
        # make rain mask and tracks file names
        rain_masks_name = mask_path  
        rain_tracks_name = "/project/pr133/mblanc/cell_tracker_output/" + climate + "_climate/cell_tracks_" + day.strftime("%Y%m%d") + ".json"
        
        # track supercells
        print("tracking supercells")
        supercells, missed_mesocyclones, lons, lats = track_Scells(timesteps, fnames_p, fnames_s, rain_masks_name, rain_tracks_name, threshold, sub_threshold, min_area, aura)
        
        print("writing data to file")
        outfile_json = os.path.join(outpath, "supercell_" + day.strftime("%Y%m%d") + ".json")
        outfile_nc = os.path.join(outpath, "meso_masks_" + day.strftime("%Y%m%d") + ".nc")
        write_to_json(supercells, missed_mesocyclones, outfile_json)
        write_masks_to_netcdf(supercells, lons, lats, outfile_nc)
    
    print("finished tracking all days in queue")
    
    return

#====================================================================================================================

outpath = "/scratch/snx3000/mblanc/SDT_output/seasons/2021"
start_day = "20210401"
end_day = "20210930"
climate = "current"

main(outpath, start_day, end_day, climate)

# if __name__ == "__main__":
#     p = argparse.ArgumentParser(description="tracks cells")
#     p.add_argument("inpath", type=str, help="path to input files")
#     p.add_argument("outpath", type=str, help="path to output files")
#     p.add_argument("start_day", type=str, help="start day")
#     p.add_argument("end_day", type=str, help="end day")
#     args = p.parse_args()

#     main(**vars(args))
