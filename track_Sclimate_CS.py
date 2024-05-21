"""
version intended for the case studies
takes the date and hours of the considered case study as inputs
detects and assigns supercells to existing rain cells, saves supercells information into a json file sa well as mesocyclones masks into a netcdf file

input:
    outpath: path to output files, str
    day: day of the considered cases study YYYYMMDD, str
    hours: list of integer hours of the case study, list or range of ints
    zeta_th: vorticity threshold for each pressure level, float
    w_th: updraught velocity threshold for each pressure level, float
    min_area: minimum horizontal area threshold for each pressure level in grid point, int
    aura: dilation radius for each pressure level in grid point, int

output:
    supercell_YYYYmmdd.json: json file containing supercell tracks
    meso_masks_YYYmmdd.nc: mesocyclone binary masks, netcdf
"""

#import sys
import os
#import argparse

#import xarray as xr
import pandas as pd
import numpy as np

from Scell_tracker import track_Scells, write_to_json, write_masks_to_netcdf

#====================================================================================================================

def main(outpath, day, hours, zeta_th, w_th, min_area=3, aura=1):

    # make output directory
    os.makedirs(outpath, exist_ok=True)
    
    day_obj = pd.to_datetime(day, format="%Y%m%d")
    print("processing day: ", day_obj.strftime("%Y/%m/%d"))
    
    # prepare the data for the day
    print("preparing data")
    
    # prepare hourly wind and surface pressure files path, and 5 min surface hail files path
    inpath = "/scratch/snx3000/mblanc/SDT/infiles/CaseStudies"
    path_p = os.path.join(inpath, "1h_3D_plev")
    path_s = os.path.join(inpath, "1h_2D")
    path_h = os.path.join(inpath, "5min_2D")
    
    # prepare corresponding file names
    fname_h = os.path.join(path_h, "cut_HAILlffd" + day + ".nc")
    fnames_p = []
    fnames_s = []
    
    # create hourly timesteps
    timesteps = []
    for i, h in enumerate(hours):
        dtstr = day + str(h).zfill(2) + "0000" # YYYYmmddHHMMSS
        fname_p = os.path.join(path_p, "cut_lffd" + dtstr + "p.nc")
        fname_s = os.path.join(path_s, "cut_PSlffd" + dtstr + ".nc")
        fnames_p.append(fname_p)
        fnames_s.append(fname_s)
        timesteps.append(pd.to_datetime(dtstr, format="%Y%m%d%H%M%S"))
    
    # prepare rain masks and tracks files names
    CT2_outpath = "/scratch/snx3000/mblanc/CT_CSs/outfiles_CT2PT"
    rain_masks_name = os.path.join(CT2_outpath, "cell_masks_" + day + ".nc")
    rain_tracks_name = os.path.join(CT2_outpath, "cell_tracks_" + day + ".json")
    
    # track supercells
    print("tracking supercells")
    supercells, na_vorticies, lons, lats = track_Scells(day_obj, timesteps, fnames_p, fnames_s, fname_h, rain_masks_name,
                                                        rain_tracks_name, zeta_th, w_th, min_area, aura)
    
    #write data to output files
    print("writing data to file")
    outfile_json = os.path.join(outpath, "supercell_" + day + ".json")
    outfile_nc = os.path.join(outpath, "meso_masks_" + day + ".nc")
    write_to_json(supercells, na_vorticies, outfile_json)
    write_masks_to_netcdf(supercells, lons, lats, outfile_nc)

    print("finished tracking day")
    
    return

#====================================================================================================================

outpath = "/scratch/snx3000/mblanc/SDT/SDT2_output/current_climate/CaseStudies"
CS_days = ['20120630', '20130727', '20130729', '20140625', '20170801', '20190610', '20190611', '20190613', '20190614',
            '20190820', '20210620', '20210628', '20210629', '20210708', '20210712', '20210713']
CS_ranges = [np.arange(14,24), np.arange(14,23), np.arange(7,16), np.arange(10,17), np.arange(18,24), np.arange(16,21),
              np.arange(9,17), np.arange(17,20), np.arange(18,24), np.arange(13,22), np.arange(13,19), np.arange(10,22),
              np.arange(11,21), np.arange(13,17), np.arange(17,20), np.arange(11,16)]
zeta_ths = [4,5,6]
w_ths = [5,6,7]

for zeta_th in zeta_ths:
    for w_th in w_ths:
        for i, day in enumerate(CS_days):
            main(outpath, day, CS_ranges[i], zeta_th, w_th)

# if __name__ == "__main__":
#     p = argparse.ArgumentParser(description="tracks cells")
#     p.add_argument("inpath", type=str, help="path to input files")
#     p.add_argument("outpath", type=str, help="path to output files")
#     p.add_argument("start_day", type=str, help="start day")
#     p.add_argument("end_day", type=str, help="end day")
#     args = p.parse_args()

#     main(**vars(args))
