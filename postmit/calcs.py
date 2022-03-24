"""calcs

Collection of functions that calculate additional variables that are not
directly available through the `diagnostics` of MITgcm or packages used.
"""
import xgcm
import gsw
import xarray as xr
from MITgcmutils import jmd95 as jmd

def vort(ds, grid=None):
    """
    """
    if grid == None:
        metrics = {
            ('X'): ['dxC', 'dxG', 'dxF', 'dxV'], # X distances
            ('Y'): ['dyC', 'dyG', 'dyF', 'dyU'], # Y distances
            ('Z'): ['drF', 'drW', 'drS', 'drC'], # Z distances
            ('X', 'Y'): ['rAw', 'rAs', 'rA', 'rAz'] # Areas in x-y plane
            }
        grid = xgcm.Grid(ds, periodic=["X", "Y"], metrics=metrics)
    ds["VORT"] = grid.derivative(ds.VVEL, "X") - grid.derivative(ds.UVEL, "Y")
    return ds


def rossby_num(ds, grid=None, path_to_input=None):
    """
    """
    if "fU" not in ds.variables:
        ds = get_const(ds, path_to_input)
    if "VORT" in ds.variables:
        ds["RosNum"] = abs(ds["VORT"] / ds["fU"])
    else:
        ds = vort(ds, grid)
        ds["RosNum"] = abs(ds["VORT"] / ds["fU"])
    return ds


def transports(ds, grid=None):
    """
    """
    if grid == None:
        metrics = {
            ('X'): ['dxC', 'dxG', 'dxF', 'dxV'], # X distances
            ('Y'): ['dyC', 'dyG', 'dyF', 'dyU'], # Y distances
            ('Z'): ['drF', 'drW', 'drS', 'drC'], # Z distances
            ('X', 'Y'): ['rAw', 'rAs', 'rA', 'rAz'] # Areas in x-y plane
            }
        grid = xgcm.Grid(ds, periodic=["X", "Y"], metrics=metrics)
    ds["UTRANS"] = grid.integrate(grid.integrate(ds.UVEL, "Y"), "Z").mean("XG")
    ds["VTRANS"] = grid.integrate(grid.integrate(ds.VVEL, "X"), "Z").mean("YG")
    Depthu = -(ds.drC).sum("Z")
    Depthv = -(ds.drS).sum("Z")
    ds["UVELbot"] = ds.UVEL.sel(Z=Depthu, method="nearest")
    ds["VVELbot"] = ds.VVEL.sel(Z=Depthv, method="nearest")
    ds["UTRANSbaro"] = grid.integrate(grid.integrate(
        (ds.UVEL - ds.UVELbot).where(ds.hFacW > 0), "Y"), "Z").mean("XG")
    ds["VTRANSbaro"] = grid.integrate(grid.integrate(
        (ds.VVEL - ds.VVELbot).where(ds.hFacW > 0), "X"), "Z").mean("YG")
    return ds


def sig0(ds):
    """
    """
    ds["SIG0"] = xr.apply_ufunc(jmd.dens, ds.SALT, ds.THETA, 0,
                                dask='parallelized',
                                output_dtypes=[ds.THETA.dtype])
    return ds


def sigi(ds, p):
    """
    """
    if p > 10:
        raise ValueError('Unrealistic pressure, `p` must be in range(0, 10)')
    press = p * 1000
    name = "SIG" + str(p)
    ds[name] = xr.apply_ufunc(jmd.dens, ds.SALT, ds.THETA, press,
                              dask='parallelized',
                              output_dtypes=[ds.THETA.dtype])
    return ds


def total_MOC(ds):
    """
    """
    return ds


def residual_MOC(ds):
    """
    """
    return ds


def eddy_MOC(ds):
    """
    """
    return ds


def get_const(ds, path_to_input):
    """Get constants from `data` (g, rhonil, cp) or use defaults.
    """
    with open(path_to_input + 'data') as f:
        data = f.readlines()
        #
        # gravity
        line = 0
        try:
            while data[line][1:8] != 'gravity':
                line += 1
                continue
            gravity = float(data[line].strip().split('=')[1].split(',')[0])
        except:
            gravity = 9.81
        ds["gravity"] = gravity
        #
        # rhonil
        line = 0
        try:
            while data[line][1:7] != 'rhonil':
                line += 1
                continue
            rhonil = float(data[line].strip().split('=')[1].split(',')[0])
        except:
            rhonil = 999.8
        ds["rhonil"] = rhonil
        #
        # rhoconst
        line = 0
        try:
            while data[line][1:9] != 'rhoconst':
                line += 1
                continue
            rhoconst = float(data[line].strip().split('=')[1].split(',')[0])
        except:
            rhoconst = rhonil
        ds["rhoconst"] = rhoconst
        #
        # heat capacity
        line = 0
        try:
            while data[line][1:16] != 'HeatCapacity_Cp':
                line += 1
                continue
            HeatCapacity_Cp =
                float(data[line].strip().split('=')[1].split(',')[0])
        except:
            HeatCapacity_Cp = 3994.
        ds["HeatCapacity_Cp"] = HeatCapacity_Cp
        #
        # ups (conversion factor from Sstart to SA)
        ds["ups"] = (35.16504 / 35
        #
        # f0
        line = 0
        try:
            while data[line][1:3] != 'f0':
                line += 1
                continue
            f0 = float(data[line].strip().split('=')[1].split(',')[0])
        except:
            f0 = 1.0E-4
        # beta
        line = 0
        try:
            while data[line][1:5] != 'beta':
                line += 1
                continue
            beta = float(data[line].strip().split('=')[1].split(',')[0])
        except:
            beta = 1.0E-11
        # f
        ds["fF"] = (f0 + (beta * ds.dyF).cumsum("YC"))
        ds["fG"] = (f0 + (beta * ds.dyG).cumsum("YC"))
        ds["fC"] = ((f0 + (beta * (ds.dyF[0, 0].values / 2)))
                    + (beta * ds.dyC).cumsum("YG"))
        ds["fU"] = ((f0 + (beta * (ds.dyG[0, 0].values / 2)))
                    + (beta * ds.dyU).cumsum("YG"))
    #
    # ice-ocean drag
    if glob.glob(path_to_input + 'data.seaice'):
        with open(path_to_input + 'data.seaice') as f:
            data = f.readlines()
            line = 0
            try:
                while data[line][1:17] != 'SEAICE_waterDrag':
                    line += 1
                    continue
                SEAICE_waterDrag =\
                    float(data[line].strip().split('=')[1].split(',')[0])
            except:
                SEAICE_waterDrag = 5.5E-3
            ds["SEAICE_waterDrag"] = SEAICE_waterDrag
    return ds


def buoy(ds, path_to_input=None, densvar="RHOAnoma"):
    """
    """
    if (("gravity" not in ds) | ("rhoconst" not in ds)):
        ds = get_const(ds, path_to_input)
    ds["BUOY"] = ((-ds["gravity"] / ds["rhoconst"]) * (ds.[densvar]))
    ds["BUOY"].attrs["units"] = 'm/s^2'
    return ds


def press(ds, path_to_input=None):
    """
    """
    if "rhonil" not in ds:
        ds = get_const(ds, path_to_input)
    ds["PRESS"] = (ds.PHIHYD + ds.PHrefC) * rhonil * 0.0001
    ds["PRESS"].attrs["units"] = 'dbar'
    return ds


def dens(ds, path_to_input=None):
    """
    """
    if "PRESS" not in ds:
        ds = press(ds, path_to_input)
    ds["DENS"] = xr.apply_ufunc(jmd.dens, ds.SALT, ds.THETA, ds.PRESS,
                                dask='parallelized',
                                output_dtypes=[ds.THETA.dtype])
    return ds

def SA(ds, path_to_input=None,
       latmin=None, latmax=None, lonmin=None, lonmax=None):
    """
    """
    if "ups" not in ds:
        ds = get_const(ds, path_to_input)
    if (("latF" not in ds) | ("lonF" not in ds)):
        ds = add_lat_lon(ds, latmin, latmax, lonmin, lonmax)
    ds["SA"] = xr.apply_ufunc(gsw.conversions.SA_from_Sstar,
                              ds.SALT * ups, ds.PRESS, ds.lonF, ds.latF,
                              dask='parallelized',
                              output_dtypes=[ds.SALT.dtype])
    return ds


def alpha(ds, path_to_input=None,
          latmin=None, latmax=None, lonmin=None, lonmax=None):
    """
    """
    if "SA" not in ds.variables:
        ds = SA(ds, path_to_input, latmin, latmax, lonmin, lonmax)
    if "PRESS" not in ds.variables:
        ds = press(ds, path_to_input)
    ds["alpha"] = xr.apply_ufunc(gsw.alpha, ds.SA, ds.THETA, ds.PRESS,
                                 dask='parallelized',
                                 output_dtypes=[ds.SA.dtype])
    ds["alpha"].attrs["units"] = '1/K'
    return ds


def beta(ds, path_to_input=None,
         latmin=None, latmax=None, lonmin=None, lonmax=None):
    """
    """
    if "SA" not in ds.variables:
        ds = SA(ds, path_to_input, latmin, latmax, lonmin, lonmax)
    if "PRESS" not in ds.variables:
        ds = press(ds, path_to_input)
    ds["beta"] = xr.apply_ufunc(gsw.beta, ds.SA, ds.THETA, ds.PRESS,
                                dask='parallelized', output_dtypes=[ds.SA.dtype])
    ds["beta"].attrs["units"] = 'kg/g'
    return ds


def surface_buoy_flux(ds, path_to_input=None,
         latmin=None, latmax=None, lonmin=None, lonmax=None):
    """
    """
    if (("gravity" not in ds)
        | ("rhoconst" not in ds)
        | ("HeatCapacity_Cp" not in ds)):
        ds = get_const(ds, path_to_input)
    if "SA" not in ds.variables:
        ds = SA(ds, path_to_input, latmin, latmax, lonmin, lonmax)
    ds["BFlx_SURF"] = ((ds.gravity / ds.rhoconst)
                        * ((ds.alpha.isel(Z=0) * ds.oceQnet
                            / ds.HeatCapacity_Cp)
                           - (ds.beta.isel(Z=0) * ds.isel(Z=0).SA *
                           -ds.oceFWflx)))
    return ds


def w_ekman(ds, grid=None, path_to_input=None,
            taux_name="EXFtaux", tauy_name="EXFtauy", out_name="WVELEk"):
    """
    tau fields should be on U and V points
    """
    if "rhoconst" not in ds:
        ds = get_const(ds, path_to_input)
    if grid == None:
        metrics = {
            ('X'): ['dxC', 'dxG', 'dxF', 'dxV'], # X distances
            ('Y'): ['dyC', 'dyG', 'dyF', 'dyU'], # Y distances
            ('Z'): ['drF', 'drW', 'drS', 'drC'], # Z distances
            ('X', 'Y'): ['rAw', 'rAs', 'rA', 'rAz'] # Areas in x-y plane
            }
        grid = xgcm.Grid(ds, periodic=["X", "Y"], metrics=metrics)
    if "VORT" not in ds.variables:
        ds = vort(ds, grid)
    if "XG" in ds[taux_name].dims:
        taux = ds[taux_name]
    else:
        taux = grid.interp(ds[taux_name], "X")
    if "YG" in ds[tauy_name].dims:
        tauy = ds[tauy_name]
    else:
        tauy = grid.interp(ds[tauy_name], "Y")
    ds[out_name] = ((1 / ds.rhoconst)
                    * grid.derivative(tauy, "X")
                       / ds.fU + ds["VORT"].isel(Z=0))
                       - grid.derivative(taux, "Y")
                       / ds.fU + ds["VORT"].isel(Z=0)))
    return ds


def ice_ocean_stress(ds, path_to_input=None,
                     thick_name="SIheff", fract_name="SIarea",
                     taux_name="SIOtaux", tauy_name="SIOtauy"):
    """
    """
    if (("rhoconst" not in ds) | ("SEAICE_waterDrag" not in ds)):
        ds = get_const(ds, path_to_input)
    ds[taux_name] = (ds.rhoconst * ds.SEAICE_waterDrag
        * (ds.SIuice.where(grid.interp(ds[thick_name], "X") > 0, other=0)
           - ds.UVEL.isel(Z=0))
        * abs(ds.SIuice.where(grid.interp(ds[thick_name], "X") > 0, other=0)
              - ds.UVEL.isel(Z=0))
        * grid.interp(ds[fract_name], "X"))
    ds[tauy_name] = (ds.rhoconst * ds.SEAICE_waterDrag
        * (ds.SIvice.where(grid.interp(ds[thick_name], "Y") > 0, other=0)
           - ds.VVEL.isel(Z=0))
        * abs(ds.SIvice.where(grid.interp(ds[thick_name], "Y") > 0, other=0)
              - ds.VVEL.isel(Z=0))
        * grid.interp(ds[fract_name], "Y"))
    return ds
