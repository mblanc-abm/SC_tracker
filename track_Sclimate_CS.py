"""
version intended for the case studies
takes the date and hours of the considered case study as input
detects and assigns supercells to existing rain cells and saves supercellls information into a json file

input:
    outpath: path to output file, str
    day: day of the considered cases study YYYYMMDD, str
    hours: list of integer hours of the case study, list or range of ints
    cut: cut of the  case study, str

output:
    supercells_YYYYMMDD.json: json file containing cell tracks
"""

#import sys
import os
#import argparse

#import xarray as xr
import pandas as pd
#import numpy as np

from Scell_tracker import track_Scells, write_to_json

#====================================================================================================================

def main(outpath, day, hours, cut):
    
    # set tracking parameters
    threshold = 75
    sub_threshold = 65
    min_area = 3
    aura = 2
    #quiet = True

    # make output directory
    os.makedirs(outpath, exist_ok=True)
    
    day_obj = pd.to_datetime(day, format="%Y%m%d")
    print("processing day: ", day_obj.strftime("%Y/%m/%d"))
    # prepare the data for the day
    print("preparing data")
    
    # make timesteps
    timesteps = []
    for i,h in enumerate(hours):
        dtstr = day + str(h).zfill(2) + "0000" # YYYYmmddHHMMSS
        timesteps.append(pd.to_datetime(dtstr, format="%Y%m%d%H%M%S"))
    
    # make pressure level and surface pressure file names
    fnames_p = []
    fnames_s = []
    path_p = "/scratch/snx3000/mblanc/UHfiles/" + cut + "_lffd"
    path_s = "/scratch/snx3000/mblanc/UHfiles/" + cut + "_PSlffd"
    for i, dt in enumerate(timesteps):
        fnames_p.append(path_p + dt.strftime("%Y%m%d%H%M%S") + "p.nc")
        fnames_s.append(path_s + dt.strftime("%Y%m%d%H%M%S") + ".nc")
    
    # make rain mask and tracks file names
    rain_masks_name = "/scratch/snx3000/mblanc/cell_tracker/CaseStudies/outfiles/cell_masks_" + day + ".nc"
    rain_tracks_name = "random, useless"
    
    # track supercells
    print("tracking supercells")
    supercells, missed_mesocyclones = track_Scells(timesteps, fnames_p, fnames_s, rain_masks_name, rain_tracks_name, threshold, sub_threshold, min_area, aura)
    
    print("writing data to file")
    outfile_json = os.path.join(outpath, "supercell_" + day + ".json")
    _ = write_to_json(supercells, missed_mesocyclones, outfile_json)

    print("finished tracking day")
    
    return

#====================================================================================================================

outpath = "/scratch/snx3000/mblanc/SDT_output/CaseStudies"
day = "20210620"
hours = range(13,19)
cut = "swisscut"

main(outpath, day, hours, cut)

# if __name__ == "__main__":
#     p = argparse.ArgumentParser(description="tracks cells")
#     p.add_argument("inpath", type=str, help="path to input files")
#     p.add_argument("outpath", type=str, help="path to output files")
#     p.add_argument("start_day", type=str, help="start day")
#     p.add_argument("end_day", type=str, help="end day")
#     args = p.parse_args()

#     main(**vars(args))
