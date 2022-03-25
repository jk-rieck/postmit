"""checks

Collection of functions that perform checks on the provided model output to
verify that all required variables are present and, if not, try to construct
them and add them to the dataset.
"""
import glob
import numpy as np

def fix_c_grid_axis_shift(ds):
    """Fix for when the attribute 'c_grid_axis_shift' is stored as a list
    rather than a float, which can occur when loading data with xmitgcm and
    then storing it as netcdf files.

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
    """Check whether all z distances needed for xgcm are present in Dataset and
    try to construct the required metrics if they are not present.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset to perform the check on.

    Returns
    -------
    ds : xarray.Dataset
        Checked xarray Dataset.
    """
    # see if we can find `drF`, the vertical cell size
    if 'drF' in ds.coords:
        # if one of the vertical cell sizes is not present, we recompute all
        # of them by multiplying `drF` with the respective partial cell
        # factor `hFacC` etc.
        if (('drW' not in ds.coords)
            | ('drS' not in ds.coords)
            | ('drC' not in ds.coords)):
            ds['drW'] = ds.hFacW * ds.drF #vertical cell size at u point
            ds['drS'] = ds.hFacS * ds.drF #vertical cell size at v point
            ds['drC'] = ds.hFacC * ds.drF #vertical cell size at tracer point
            # we make the variables coordinates and add attributes
            ds = ds.set_coords(["drW", "drS", "drC"])
            ds["drW"].attrs = {"standard_name": "cell_z_size_at_u_location",
                               "long_name": "cell z size",
                               "unit": "m"}
            ds["drS"].attrs = {"standard_name": "cell_z_size_at_v_location",
                               "long_name": "cell z size",
                               "unit": "m"}
            ds["drC"].attrs = {"standard_name": "cell_z_size_at_t_location",
                               "long_name": "cell z size",
                               "unit": "m"}
    else:
        # if `drF` is not found, we cannot compute the other cell sizes
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
        # read all lines from `data.layers` to variable `data_layers`
        data_layers = f.readlines()
        linestart = 0
        # look for the line that includes `layers_bounds`, that will be the
        # first line to consider
        while data_layers[linestart][1:14] != 'layers_bounds':
            linestart += 1
            continue
        lineend = linestart + 1
        # in the lines following `linestart` we check whether the first entries
        # are numbers, if yes we assume that these entries still belong to the
        # definition of `layer_bounds`
        while lineend > 0:
            try:
                tmp = int(data_layers[lineend].strip()[0])
                lineend += 1
            except:
                break
        # now we take the lines from `linestart` to `lineend` and extract the
        # numbers, i.e. get rid of all the comma, spaces, apostrophes, etc.
        # to put the numbers into an array
        isos = np.array([float(i) for i in data_layers[linestart].split('=')[1].split('\n')[0].split(',')[0:-1]])
        for j in range(linestart+1, lineend):
            isos = np.hstack((isos,
                              np.array([float(i) for i in data_layers[j].strip().split('\n')[0].split(',')[0:-1]])))
        return isos


def check_layers(ds, path_to_input):
    """If the model simulation used the Layers package, this function can be
    used to check whether the layer-coordinate (called `_UNKNOWN_` by default
    when loading the data with `xmitgcm`) is present and if so, it will be
    renamed to `layer_center` and populated with the layer centers derived from
    the `data.layers` file.

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
    # check if file `data.layers` is found
    if glob.glob(path_to_input + 'data.layers'):
        # look for dimension `_UNKNOWN_`
        if '_UNKNOWN_' in ds.dims:
            ds = ds.rename({'_UNKNOWN_': 'layer_center'})
            # get layer coordinates bounds from `data.layers`
            isopycnal_bounds = get_isopycnals(path_to_input)
            # convert layer bounds to layer centers
            isopycnal_centers = (isopycnal_bounds[0:-1]
                                 + (isopycnal_bounds[1::]
                                    - isopycnal_bounds[0:-1]) / 2)
            # add to dataset and add attributes
            ds["layer_center"] = isopycnal_centers
            ds["layer_center"].attrs = {"long_name":
                                        "center of layers from Layers package"}
            ds = ds.chunk({'layer_center': -1})
    # if data.layers is not present, Layers package was prob. not used and
    # ds is left unchanged
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
