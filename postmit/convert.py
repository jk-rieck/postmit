"""convert

Collection of functions to convert binary output from the MITgcm to other
formats. Currently these formats are netCDF and zarr.
"""
import xmitgcm as xmit
import xarray as xr
import zarr
import numpy as np
import glob
from . import checks


def load_data(path, grid_dir, prefix, start, end, step, dt,
              tout='1m', chunk=True, geom="cartesian", cal="360_day"):
    """
    """
    if tout == '1y':
        ref_date = '0000-07-01 00:00:00'
        time_chunk = 20
    elif tout == '1m':
        ref_date = '0000-12-15 00:00:00'
        time_chunk = 24
    elif tout == '5d':
        ref_date = '0000-12-28 12:00:00'
        time_chunk = 36
    elif tout == '1d':
        ref_date = '0000-12-30 12:00:00'
        time_chunk = 30
    ds = xmit.open_mdsdataset(
        path, iters=list(np.arange(start, end + 1, step)), delta_t=dt,
        ignore_unknown_vars=True, grid_dir=grid_dir, ref_date=ref_date,
        geometry=geom, calendar=cal, prefix=prefix)
    if chunk:
        ds = ds.chunk({"time": time_chunk,
                       "Z": -1, "Zl": -1, "Zu": -1, "Zp1": -1,
                       "XC": -1, "XG": -1, "YC": -1, "YG": -1})
    return ds


def check_data(path, tout, prefix_in):
    has = False
    prefixes = []
    if glob.glob(path + '*' + tout + '*.data'):
        has = True
        for pf in prefix_in:
            if glob.glob(path + pf + '*' + tout + '*.data'):
                prefixes = prefixes + [pf + '_' + tout]
    return has, prefixes


def convert2nc(path, grid_path, in_path, out_path, ds_out, prefixes,
               starts, ends, steps, dt,
               chunk=True, geom="cartesian", cal="360_day"):
    """
    """
    has_1y, prefixes_1y = check_data(path, '1y', prefixes)
    has_1m, prefixes_1m = check_data(path, '1m', prefixes)
    has_5d, prefixes_5d = check_data(path, '5d', prefixes)
    has_1d, prefixes_1d = check_data(path, '1d', prefixes)
    if has_1y:
        start = starts['1y']
        end = ends['1y']
        step = steps['1y']
        print('    -- loading 1y-output from binary')
        ds_1y = xr.merge([load_data(path, grid_path, p, start, end, step,
                                    dt, tout='1y', chunk=True,
                                    geom="cartesian", cal="360_day")
                          for p in prefixes_1y])
        print('    -- running checks on 1ydata')
        ds_1y = checks.apply_all_checks(ds_1y, in_path)
        print('    -- writing 1y-output to netCDF')
        years, datasets = zip(*ds_1y.groupby("time.year"))
        paths = [out_path + ds_out + "_1y_%04d.nc" % y for y in years]
        xr.save_mfdataset(datasets, paths, engine="netcdf4", format="NETCDF4")
    if has_1m:
        start = starts['1m']
        end = ends['1m']
        step = steps['1m']
        print('    -- loading 1m-output from binary')
        ds_1m = xr.merge([load_data(path, grid_path, p, start, end, step,
                                    dt, tout='1m', chunk=True,
                                    geom="cartesian", cal="360_day")
                          for p in prefixes_1m])
        print('    -- running checks on 1m-data')
        ds_1m = checks.apply_all_checks(ds_1m, in_path)
        print('    -- writing 1m-output to netCDF')
        years, datasets = zip(*ds_1m.groupby("time.year"))
        paths = [out_path + ds_out + "_1m_%04d.nc" % y for y in years]
        xr.save_mfdataset(datasets, paths, engine="netcdf4", format="NETCDF4")
    if has_5d:
        start = starts['5d']
        end = ends['5d']
        step = steps['5d']
        print('    -- loading 5d-output from binary')
        ds_5d = xr.merge([load_data(path, grid_path, p, start, end, step,
                                    dt, tout='5d', chunk=True,
                                    geom="cartesian", cal="360_day")
                          for p in prefixes_5d])
        print('    -- running checks on 5d-data')
        ds_5d = checks.apply_all_checks(ds_5d, in_path)
        print('    -- writing 5d-output to netCDF')
        years, datasets = zip(*ds_5d.groupby("time.year"))
        paths = [out_path + ds_out + "_5d_%04d.nc" % y for y in years]
        xr.save_mfdataset(datasets, paths, engine="netcdf4", format="NETCDF4")
    if has_1d:
        start = starts['1d']
        end = ends['1d']
        step = steps['1d']
        print('    -- loading 1d-output from binary')
        ds_1d = xr.merge([load_data(path, grid_path, p, start, end, step,
                                    dt, tout='1d', chunk=True,
                                    geom="cartesian", cal="360_day")
                          for p in prefixes_1d])
        print('    -- running checks on 1d-data')
        ds_1d = checks.apply_all_checks(ds_1d, in_path)
        print('    -- writing 1d-output to netCDF')
        years, datasets = zip(*ds_1d.groupby("time.year"))
        paths = [out_path + ds_out + "_1d_%04d.nc" % y for y in years]
        xr.save_mfdataset(datasets, paths, engine="netcdf4", format="NETCDF4")
    return str("data saved to ") + str(out_path)

def convert2zarr(path, grid_path, in_path, out_path, ds_out, prefixes,
                 starts, ends, steps, dt,
                 chunk=True, geom="cartesian", cal="360_day"):
    """
    """
    has_1y, prefixes_1y = check_data(path, '1y', prefixes)
    has_1m, prefixes_1m = check_data(path, '1m', prefixes)
    has_5d, prefixes_5d = check_data(path, '5d', prefixes)
    has_1d, prefixes_1d = check_data(path, '1d', prefixes)
    if has_1y:
        start = starts['1y']
        end = ends['1y']
        step = steps['1y']
        print('    -- loading 1y-output from binary')
        ds_1y = xr.merge([load_data(path, grid_path, p, start, end, step,
                                    dt, tout='1y', chunk=True,
                                    geom="cartesian", cal="360_day")
                          for p in prefixes_1y])
        print('    -- running checks on 1y-data')
        ds_1y = checks.apply_all_checks(ds_1y, in_path)
        print('    -- writing 1y-output to zarr')
        ds_1y.to_zarr(out_path + ds_out + '.1y.zarr/',
                      mode='w', safe_chunks=True)
        del ds_1y
    if has_1m:
        start = starts['1m']
        end = ends['1m']
        step = steps['1m']
        print('    -- loading 1m-output from binary')
        ds_1m = xr.merge([load_data(path, grid_path, p, start, end, step,
                                    dt, tout='1m', chunk=True,
                                    geom="cartesian", cal="360_day")
                          for p in prefixes_1m])
        print('    -- running checks on 1m-data')
        ds_1m = checks.apply_all_checks(ds_1m, in_path)
        print('    -- writing 1m-output to zarr')
        ds_1m.to_zarr(out_path + ds_out + '.1m.zarr/',
                      mode='w', safe_chunks=True)
        del ds_1m
    if has_5d:
        start = starts['5d']
        end = ends['5d']
        step = steps['5d']
        print('    -- loading 5d-output from binary')
        ds_5d = xr.merge([load_data(path, grid_path, p, start, end, step,
                                    dt, tout='5d', chunk=True,
                                    geom="cartesian", cal="360_day")
                          for p in prefixes_5d])
        print('    -- running checks on 5d-data')
        ds_5d = checks.apply_all_checks(ds_5d, in_path)
        print('    -- writing 5d-output to zarr')
        ds_5d.to_zarr(out_path + ds_out + '.5d.zarr/',
                      mode='w', safe_chunks=True)
        del ds_5d
    if has_1d:
        start = starts['1d']
        end = ends['1d']
        step = steps['1d']
        print('    -- loading 1d-output from binary')
        ds_1d = xr.merge([load_data(path, grid_path, p, start, end, step,
                                    dt, tout='1d', chunk=True,
                                    geom="cartesian", cal="360_day")
                          for p in prefixes_1d])
        print('    -- running checks on 1d-data')
        ds_1d = checks.apply_all_checks(ds_1d, in_path)
        print('    -- writing 1d-output to zarr')
        ds_1d.to_zarr(out_path + ds_out + '.1d.zarr/',
                      mode='w', safe_chunks=True)
        del ds_1d
    return str("data saved to ") + str(out_path)
