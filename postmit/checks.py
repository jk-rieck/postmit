"""checks

Collection of functions that perform checks on the provided model output to verify that all required variables are present and, if not, try to construct them and add them to the dataset.
"""
import glob
import numpy as np

def fix_c_grid_axis_shift(ds):
    """Fix for when the attribute 'c_grid_axis_shift' is stored as a list rather than a float, which can occur when loading data with xmitgcm and then storing it as netcdf files.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset to perform the check on.

    Returns
    -------
    ds : xarray.Dataset
        Checked xarray Dataset.
    """
    for co in ds.coords:
        try:
            ds[co].attrs['c_grid_axis_shift'] = ds[co].c_grid_axis_shift[0]
        except:
            continue
    return ds


def check_z_distances(ds):
    """Check whether all z distances needed for xgcm are present in Dataset and try to construct the required metrics if they are not present.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset to perform the check on.

    Returns
    -------
    ds : xarray.Dataset
        Checked xarray Dataset.
    """
    if 'drF' in ds.coords:
        if (('drW' not in ds.coords)
            | ('drS' not in ds.coords)
            | ('drC' not in ds.coords)):
            ds['drW'] = ds.hFacW * ds.drF #vertical cell size at u point
            ds['drS'] = ds.hFacS * ds.drF #vertical cell size at v point
            ds['drC'] = ds.hFacC * ds.drF #vertical cell size at tracer point
            ds = ds.set_coords(["drW", "drS", "drC"])
    else:
        raise ValueError("no information on z distances found in Dataset")
    return ds


def get_isopycnals(path_to_input):
    """Get isopycnal level bounds from `data.layers` file located at `path`.

    Parameters
    ----------
    path_to_input : str
        Path to where `data.layers` file resides.

    Returns
    -------
    isos : np.array
        Array with the centers of the isopycnal layers.
    """
    with open(path_to_input + 'data.layers') as f:
        data_layers = f.readlines()
        linestart = 0
        while data_layers[linestart][1:14] != 'layers_bounds':
            linestart += 1
            continue
        lineend = linestart + 1
        while lineend > 0:
            try:
                tmp = int(data_layers[lineend].strip()[0])
                lineend += 1
            except:
                break
        isos = np.array([float(i) for i in data_layers[linestart].split('=')[1].split('\n')[0].split(',')[0:-1]])
        for j in range(linestart+1, lineend):
            isos = np.hstack((isos,
                              np.array([float(i) for i in data_layers[j].strip().split('\n')[0].split(',')[0:-1]])))
        return isos


def check_layers(ds, path_to_input):
    """If the model simulation used the Layers package, this function can be used to check whether the layer-coordinate (called `_UNKNOWN_` by default when loading the data with `xmitgcm`) is present and if so, it will be renamed to `layer_center` and populated with the layer centers derived from the `data.layers` file.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset to perform the check on.
    path_to_input : str
        Path to where the `data.layers` file resides.

    Returns
    -------
    ds : xarray.Dataset
        Checked xarray Dataset.
    """
    if glob.glob(path + 'data.layers'):
        if '_UNKNOWN_' in ds.dims:
            ds = ds.rename({'_UNKNOWN_': 'layer_center'})
            isopycnal_bounds = get_isopycnals(path_to_input)
            isopycnal_centers = (isopycnal_bounds[0:-1]
                                 + (isopycnal_bounds[1::]
                                    - isopycnal_bounds[0:-1]) / 2)
            ds["layer_center"] = isopycnal_centers
        ds = ds.chunk({'layer_center': -1})
    else: # if data.layers is not present, Layers package was prob. not used
        ds = ds
    return ds


def apply_all_checks(ds, path_to_input):
    """Apply all the checks contined in this file to a Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset to perform the checks on.
    path_to_input : str
        Path to where the `input` directory of the simulation.

    Returns
    -------
    ds : xarray.Dataset
        Checked xarray Dataset.
    """
    ds = check_z_distances(fix_c_grid_axis_shift(ds))
    return check_layers(ds, path_to_input)
