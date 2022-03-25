"""adds

Collection of functions that add some coordinates or variables to the dataset
that are required for later calculation or are just convenient
"""
import numpy as np

def add_lat_lon(ds, latmin, latmax, lonmin, lonmax):
    """Add latitude and longitude to a cartesian grid simulation. This is a bit
    arbitrary but can be helpful sometimes. E.g. when using the `gsw` python
    package that requires longitude and latitude for some conversions.
    If `latmin` and `latmax` (`lonmin` and `lonmax`) are the same value, an
    array of length `len(ds.YC)` (`len(ds.XC)`) with this value will be
    created. Otherwise an array of length `len(ds.YC)` (`len(ds.XC)`) will be
    created spanning the range from `latmin` to `latmax` (`lonmin` to `lonmax`).

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset to add latitude and longitude to.
    latmin : float
        Minimum latitude (range(-90, 90)).
    latmax : float
        Maximum latitude (range(-90, 90)), must be larger or equal latmin.
    lonmin : float
        Minimum longitude (range(-180, 180)).
    lonmax : float
        Maximum longitude (range(-180, 180)), must be larger or equal lonmin.

    Returns
    -------
    ds : xarray.Dataset
        Xarray dataset with added latitude and longitude variables.
    """
    if latmin == latmax:
        # if only one latitude is given, we construct 2D arrays containing this
        # latitude everywhere
        ds["latF"] = (("YC", "XC"), np.zeros((len(ds.YC), len(ds.XC))) + latmin)
        ds["latC"] = (("YG", "XC"), np.zeros((len(ds.YG), len(ds.XC))) + latmin)
        ds["latG"] = (("YC", "XG"), np.zeros((len(ds.YC), len(ds.XG))) + latmin)
        ds["latU"] = (("YG", "XG"), np.zeros((len(ds.YG), len(ds.XG))) + latmin)
    else:
        # compute the latitudes for each grid point given latmin and latmax and
        # the number of grid points (`len(ds.YC)`). Also make sure that each
        # grid point (t, u, v, f) gets the correct latitudes.
        dyC = (latmax - latmin) / len(ds.YC)
        dyG = (latmax - latmin) / len(ds.YG)
        ds["latF"] = (("YC", "XC"),
            np.repeat(np.linspace(latmin + (dyC / 2),
                                  latmax - (dyC / 2),
                                  len(ds.YC))[:, None],
                      len(ds.XC), axis=1))
        ds["latC"] = (("YG", "XC"),
            np.repeat(np.linspace(latmin,
                                  latmax - dyG,
                                  len(ds.YG))[:, None],
                      len(ds.XC), axis=1))
        ds["latG"] = (("YC", "XG"),
            np.repeat(np.linspace(latmin + (dyC / 2),
                                  latmax - (dyC / 2),
                                  len(ds.YC))[:, None],
                      len(ds.XG), axis=1))
        ds["latU"] = (("YG", "XG"),
            np.repeat(np.linspace(latmin,
                                  latmax - dyG,
                                  len(ds.YG))[:, None],
                      len(ds.XG), axis=1))
    if lonmin == lonmax:
        # if only one longitude is given, we construct 2D arrays containing this
        # longitude everywhere
        ds["lonF"] = (("YC", "XC"), np.zeros((len(ds.YC), len(ds.XC))) + lonmin)
        ds["lonC"] = (("YG", "XC"), np.zeros((len(ds.YG), len(ds.XC))) + lonmin)
        ds["lonG"] = (("YC", "XG"), np.zeros((len(ds.YC), len(ds.XG))) + lonmin)
        ds["lonU"] = (("YG", "XG"), np.zeros((len(ds.YG), len(ds.XG))) + lonmin)
    else:
        # compute the longitudes for each grid point given lonmin and lonmax and
        # the number of grid points (`len(ds.XC)`). Also make sure that each
        # grid point (t, u, v, f) gets the correct longitudes.
        dxC = (lonmax - lonmin) / len(ds.XC)
        dxG = (lonmax - lonmin) / len(ds.XG)
        ds["lonF"] = (("YC", "XC"),
            np.repeat(np.linspace(lonmin + (dxC / 2),
                                  lonmax - (dxC / 2),
                                  len(ds.XC))[None, :],
                      len(ds.YC), axis=0))
        ds["lonC"] = (("YG", "XC"),
            np.repeat(np.linspace(lonmin + (dxC / 2),
                                  lonmax - (dxC / 2),
                                  len(ds.XC))[None, :],
                      len(ds.YG), axis=0))
        ds["lonG"] = (("YC", "XG"),
            np.repeat(np.linspace(lonmin,
                                  lonmax - dxG,
                                  len(ds.XG))[None, :],
                      len(ds.YC), axis=0))
        ds["lonU"] = (("YG", "XG"),
            np.repeat(np.linspace(lonmin,
                                  lonmax - dxG,
                                  len(ds.XG))[:, None],
                      len(ds.YG), axis=1))
    return ds
