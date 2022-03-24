"""adds

Collection of functions that add some coordinates or variables to the dataset
that are required for later calculation or are just convenient
"""

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
        Xarray dataset with `lat` and `lon` added as variables.
    """
    if latmin == latmax:
        ds["lat"] = (("YC",), np.zeros(len(ds.YC)) + latmin)
    else:
        ds["lat"] = (("YC",), np.linspace(latmin, latmax, len(ds.YC)))
    if lonmin == lonmax:
        ds["lon"] = (("XC",), np.zeros(len(ds.XC)) + lonmin)
    else:
        ds["lon"] = (("XC",), np.linspace(lonmin, lonmax, len(ds.XC)))
    return ds
