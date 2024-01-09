import sys
import os
import argparse

import xarray as xr
import pandas as pd
import numpy as np

from Scell_tracker import SuperCell, track_Scells, write_to_json


def main(inpath, outpath, start_day, end_day):
    # set tracking parameters
    threshold = 75
    sub_threshold = 65
    min_area = 3
    aura = 1
    quiet = True

    start_day = pd.to_datetime(start_day, format="%Y-%m-%dT%H")
    end_day = pd.to_datetime(end_day, format="%Y-%m-%dT%H")

    daylist = pd.date_range(start_day, end_day)

    # make output directory
    os.makedirs(outpath, exist_ok=True)

    # iterate over each day in the dataset
    for i, day in enumerate(daylist):
        # print progress
        print("processing day ", i + 1, " of ", len(daylist))
        print("processing day: ", day.strftime("%Y%m%d"))
        # get the data for one day
        print("loading data")

        path = os.path.join(inpath, "lffd" + day.strftime("%Y%m%d") + "_0606.nz")

        ds = xr.open_dataset(path, engine="netcdf4")

        field_static = {}
        field_static["lat"] = ds["lat"].values
        field_static["lon"] = ds["lon"].values

        timesteps = ds["time"].values

        fields = ds["DHAIL_MX"].values

        # track cells
        print("tracking cells")
        cells = track_cells(
            fields,
            timesteps,
            field_static=field_static,
            quiet=quiet,
            min_distance=min_distance,
            dynamic_tracking=dynamic_tracking,
            v_limit=v_limit,
            min_area=min_area,
        )

        print("gap filling swaths")
        swath = ds["DHAIL_MX"].max(dim="time").values
        swath_gf = swath
        for cell in cells:
            swath_gf = np.max(np.dstack((cell.swath, swath_gf)), 2)

        # add empty dimension for time
        swath_gf = np.expand_dims(swath_gf, axis=0)

        swath_gf_nc = xr.Dataset(
            {"DHAIL_MX": (["time", "rlat", "rlon"], swath_gf)},
            coords={
                "time": np.atleast_1d(ds["time"][0].values),
                "rlat": ds["rlat"].values,
                "rlon": ds["rlon"].values,
            },
        )
        # add lat lon
        swath_gf_nc["lat"] = xr.DataArray(
            ds["lat"].values,
            dims=["rlat", "rlon"],
            coords={"rlat": ds["rlat"].values, "rlon": ds["rlon"].values},
        )
        swath_gf_nc["lon"] = xr.DataArray(
            ds["lon"].values,
            dims=["rlat", "rlon"],
            coords={"rlat": ds["rlat"].values, "rlon": ds["rlon"].values},
        )

        # add long name to swath
        swath_gf_nc.attrs["long_name"] = "gap filled swath using cell tracking"
        # add units to swath
        swath_gf_nc.attrs["units"] = "mm"

        print("writing data to file")
        outfile_json = os.path.join(
            outpath, "cell_tracks_" + day.strftime("%Y%m%d") + ".json"
        )
        outfile_nc = os.path.join(
            outpath, "cell_masks_" + day.strftime("%Y%m%d") + ".nc"
        )

        outfile_gapfilled = os.path.join(
            outpath, "gap_filled_" + day.strftime("%Y%m%d") + ".nc"
        )
        _ = write_to_json(cells, outfile_json)

        _ = write_masks_to_netcdf(cells, timesteps, field_static, outfile_nc)

        swath_gf_nc.to_netcdf(outfile_gapfilled)
        # os.system("nczip " + outfile_gapfilled)

    print("finished tracking all days in queue")
    return


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="tracks cells")
    p.add_argument("inpath", type=str, help="path to input files")
    p.add_argument("outpath", type=str, help="path to output files")
    p.add_argument("start_day", type=str, help="start day")
    p.add_argument("end_day", type=str, help="end day")
    args = p.parse_args()

    main(**vars(args))
