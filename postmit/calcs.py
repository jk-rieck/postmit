"""calcs

Collection of functions that calculate additional variables that are not
directly available through the `diagnostics` of MITgcm or packages used.
"""
import xgcm
import gsw
import xarray as xr
from MITgcmutils import jmd95 as jmd
from . import checks
from . import adds
import glob

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
    ds["VORT"].attrs["standard_name"] = "VORT"
    ds["VORT"].attrs["long_name"] = "vertical component of vorticity (1/s)"
    ds["VORT"].attrs["units"] = "1/s"
    return ds


def rossby_num(ds, grid=None, path_to_input=None):
    """
    """
    if "fU" not in ds.variables:
        ds = get_const(ds, path_to_input)
    if "VORT" not in ds.variables:
        ds = vort(ds, grid)
    ds["RosNum"] = abs(ds["VORT"] / ds["fU"])
    ds["RosNum"].attrs["standard_name"] = "RosNum"
    ds["RosNum"].attrs["long_name"] = "Rossby number abs(VORT/f)"
    ds["RosNum"].attrs["units"] = ""
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
    Depthu = (ds.drW).cumsum("Z")
    Depthv = (ds.drS).cumsum("Z")
    DepthW = grid.interp(ds.Depth, "X", to="left")
    DepthS = grid.interp(ds.Depth, "Y", to="left")
    ds["UVELbot"] = ds.UVEL.where(Depthu >= DepthW).where(
                        ds.maskW == 1).mean("Z", skipna=True, keep_attrs=True)
    ds["VVELbot"] = ds.VVEL.where(Depthv >= DepthS).where(
                        ds.maskS == 1).mean("Z", skipna=True, keep_attrs=True)
    ds["UTRANSbaro"] = grid.integrate(grid.integrate(
        (ds.UVEL - ds.UVELbot).where(ds.hFacW > 0), "Y"), "Z").mean("XG")
    ds["VTRANSbaro"] = grid.integrate(grid.integrate(
        (ds.VVEL - ds.VVELbot).where(ds.hFacS > 0), "X"), "Z").mean("YG")
    ds["UTRANS"].attrs["standard_name"] = "UTRANS"
    ds["UTRANS"].attrs["long_name"] =\
        "mean zonal transport integrated over Y and Z (m^3/s)"
    ds["UTRANS"].attrs["units"] = "m^3/s"
    ds["VTRANS"].attrs["standard_name"] = "VTRANS"
    ds["VTRANS"].attrs["long_name"] =\
        "mean meridional transport integrated over X and Z (m^3/s)"
    ds["VTRANS"].attrs["units"] = "m^3/s"
    ds["UVELbot"].attrs["standard_name"] = "UVELbot"
    ds["UVELbot"].attrs["long_name"] =\
        "zonal velocity at the bottom (deepest wet grid cell) (m/s)"
    ds["UVELbot"].attrs["units"] = "m/s"
    ds["VVELbot"].attrs["standard_name"] = "VVELbot"
    ds["VVELbot"].attrs["long_name"] =\
        "meridional velocity at the bottom (deepest wet grid cell) (m/s)"
    ds["VVELbot"].attrs["units"] = "m/s"
    ds["UTRANSbaro"].attrs["standard_name"] = "UTRANSbaro"
    ds["UTRANSbaro"].attrs["long_name"] =\
        "mean zonal baroclinic transport integrated over Y and Z (m^3/s)"
    ds["UTRANSbaro"].attrs["units"] = "m^3/s"
    ds["VTRANSbaro"].attrs["standard_name"] = "VTRANSbaro"
    ds["VTRANSbaro"].attrs["long_name"] =\
        "mean meridional baroclinic transport integrated over Y and Z (m^3/s)"
    ds["VTRANSbaro"].attrs["units"] = "m^3/s"
    return ds


def sig0(ds):
    """
    """
    ds["SIG0"] = xr.apply_ufunc(jmd.dens, ds.SALT, ds.THETA, 0,
                                dask='parallelized',
                                output_dtypes=[ds.THETA.dtype], keep_attrs=True)
    ds["SIG0"].attrs["standard_name"] = "SIG0"
    ds["SIG0"].attrs["long_name"] =\
        "potential density referenced to the surface (0 dbar) (kg/m^3)"
    ds["SIG0"].attrs["units"] = "kg/m^3"
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
                              output_dtypes=[ds.THETA.dtype], keep_attrs=True)
    ds[name].attrs["standard_name"] = name
    ds[name].attrs["long_name"] =\
        "potential density referenced to " + str(p * 1000) + " dbar (kg/m^3)"
    ds[name].attrs["units"] = "kg/m^3"
    return ds


def total_MOC(ds, grid=None):
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
    ds["MOC"] = grid.integrate((ds.VVEL * ds.drS).cumsum(dim="Z"), "X")
    ds["MOC"].attrs["standard_name"] = "meridional_overturning"
    ds["MOC"].attrs["long_name"] = "meridional overturning (m^3/s)"
    ds["MOC"].attrs["units"] = "m^3/s"
    return ds


def residual_MOC(ds, grid=None, path_to_input=None):
    """
    """
    dw = ds.copy()
    if grid == None:
        metrics = {
            ('X'): ['dxC', 'dxG', 'dxF', 'dxV'], # X distances
            ('Y'): ['dyC', 'dyG', 'dyF', 'dyU'], # Y distances
            ('Z'): ['drF', 'drW', 'drS', 'drC'], # Z distances
            ('X', 'Y'): ['rAw', 'rAs', 'rA', 'rAz'] # Areas in x-y plane
            }
        grid = xgcm.Grid(dw, periodic=["X", "Y"], metrics=metrics)
    if "layer_center" not in ds.dims:
        dw = checks.check_layers(dw, path_to_input)
    dw["MOC_res"] = grid.integrate(
        dw.LaVH1RHO.sortby("layer_center",
        ascending=True).cumsum(dim="layer_center"), "X")
    dw["layer_depths"] = -dw.LaHs1RHO.sortby(
        "layer_center", ascending=True).cumsum(dim="layer_center").mean("XC")
    tmp = xr.merge([dw.MOC_res, dw.layer_depths, dw.LaHs1RHO])
    tmp = tmp.assign_coords(dw.coords).drop_dims(["Z", "Zp1", "Zl", "Zu"])
    tmp["layer_center"].attrs["axis"] = "Z"
    tmp["layer_bounds"] = checks.get_isopycnals(path_to_input)
    tmp["layer_bounds"].attrs["axis"] = "Z"
    tmp["layer_bounds"].attrs["c_grid_axis_shift"] = -0.5
    metrics_tmp = {
        ('Y'): ['dyC', 'dyG', 'dyF', 'dyU'], # Y distances
        ('Z'): ['LaHs1RHO'], # Z distances
        ('X', 'Y'): ['rAw', 'rAs', 'rA', 'rAz'] # Areas in x-y plane
        }
    grid2 = xgcm.Grid(tmp, periodic=["Y"], metrics=metrics_tmp)
    dw["MOC_res_z"] = grid2.transform(tmp.MOC_res, "Z", dw.Z,
                                      target_data=tmp.layer_depths)
    dw["MOC_res_z"] = dw["MOC_res_z"].transpose("time", "Z", "YG")
    dw["time"].attrs = ds["time"].attrs
    dw["Z"].attrs = ds["Z"].attrs
    dw["YG"].attrs = ds["YG"].attrs
    dw["MOC_res"].attrs["standard_name"] = "residual_overturning"
    dw["MOC_res"].attrs["long_name"] = "residual overturning (m^3/s)"
    dw["MOC_res"].attrs["units"] = "m^3/s"
    dw["MOC_res_z"].attrs["standard_name"] =\
        "residual_overturning_on_z_levels"
    dw["MOC_res_z"].attrs["long_name"] =\
        "residual overturning on depth levels (m^3/s)"
    dw["MOC_res_z"].attrs["units"] = "m^3/s"
    dw["layer_depths"].attrs["standard_name"] =\
        "layer_depths_of_isopycnal_layers"
    dw["layer_depths"].attrs["long_name"] =\
        "layer depths of isopycnal layers from LaVH1RHO"
    dw["layer_depths"].attrs["units"] = "m"
    return dw


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
        ds["gravity"].attrs["standard_name"] = "gravity"
        ds["gravity"].attrs["long_name"] = "gravitational acceleration (m/s^2)"
        ds["gravity"].attrs["units"] = "m/s^2"
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
        ds["rhonil"].attrs["standard_name"] = "rhonil"
        ds["rhonil"].attrs["long_name"] = "reference density (kg/m^3)"
        ds["rhonil"].attrs["units"] = "kg/m^3"
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
        ds["rhoconst"].attrs["standard_name"] = "rhoconst"
        ds["rhoconst"].attrs["long_name"] =\
            "vertically constant reference density (Boussinesq) (kg/m^3)"
        ds["rhoconst"].attrs["units"] = "kg/m^3"
        #
        # heat capacity
        line = 0
        try:
            while data[line][1:16] != 'HeatCapacity_Cp':
                line += 1
                continue
            HeatCapacity_Cp =\
                float(data[line].strip().split('=')[1].split(',')[0])
        except:
            HeatCapacity_Cp = 3994.
        ds["HeatCapacity_Cp"] = HeatCapacity_Cp
        ds["HeatCapacity_Cp"].attrs["standard_name"] = "HeatCapacity_Cp"
        ds["HeatCapacity_Cp"].attrs["long_name"] =\
            "specific heat capacity Cp (ocean) (J/kg/K)"
        ds["HeatCapacity_Cp"].attrs["units"] = "J/kg/K"
        #
        # ups (conversion factor from Sstart to SA)
        ds["ups"] = (35.16504 / 35)
        ds["ups"].attrs["standard_name"] = "ups"
        ds["ups"].attrs["long_name"] =\
            "scale factor to convert model salinity to preformed"
        ds["ups"].attrs["units"] = ""
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
        ds["fF"].attrs["standard_name"] = "fF"
        ds["fF"].attrs["long_name"] =\
            "coriolis parameter at t location (1/s)"
        ds["fF"].attrs["units"] = "1/s"
        ds["fG"].attrs["standard_name"] = "fG"
        ds["fG"].attrs["long_name"] =\
            "coriolis parameter at v location (1/s)"
        ds["fG"].attrs["units"] = "1/s"
        ds["fC"].attrs["standard_name"] = "fC"
        ds["fC"].attrs["long_name"] =\
            "coriolis parameter at u location (1/s)"
        ds["fC"].attrs["units"] = "1/s"
        ds["fU"].attrs["standard_name"] = "fU"
        ds["fU"].attrs["long_name"] =\
            "coriolis parameter at f location (1/s)"
        ds["fU"].attrs["units"] = "1/s"
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
        ds["SEAICE_waterDrag"].attrs["standard_name"] = "SEAICE_waterDrag"
        ds["SEAICE_waterDrag"].attrs["long_name"] =\
            "water-ice drag coefficient"
        ds["SEAICE_waterDrag"].attrs["units"] = ""
    return ds


def buoy(ds, path_to_input=None, densvar="RHOAnoma"):
    """
    """
    if (("gravity" not in ds) | ("rhoconst" not in ds)):
        ds = get_const(ds, path_to_input)
    ds["BUOY"] = ((-ds["gravity"] / ds["rhoconst"]) * (ds[densvar]))
    ds["BUOY"].attrs["standard_name"] = "BUOY"
    ds["BUOY"].attrs["long_name"] = "buoyancy (m/s^2)"
    ds["BUOY"].attrs["units"] = 'm/s^2'
    return ds


def press(ds, path_to_input=None):
    """
    """
    if "rhonil" not in ds:
        ds = get_const(ds, path_to_input)
    ds["PRESS"] = (ds.PHIHYD + ds.PHrefC) * ds.rhonil * 0.0001
    ds["PRESS"].attrs["standard_name"] = "PRESS"
    ds["PRESS"].attrs["long_name"] = "pressure (dbar)"
    ds["PRESS"].attrs["units"] = 'dbar'
    return ds


def dens(ds, path_to_input=None):
    """
    """
    if "PRESS" not in ds:
        ds = press(ds, path_to_input)
    ds["DENS"] = xr.apply_ufunc(jmd.dens, ds.SALT, ds.THETA, ds.PRESS,
                                dask='parallelized',
                                output_dtypes=[ds.THETA.dtype], keep_attrs=True)
    ds["DENS"].attrs["standard_name"] = "DENS"
    ds["DENS"].attrs["long_name"] = "in-situ density (kg/m^3)"
    ds["DENS"].attrs["units"] = 'kg/m^3'
    return ds

def SA(ds, path_to_input=None,
       latmin=None, latmax=None, lonmin=None, lonmax=None):
    """
    """
    if "ups" not in ds:
        ds = get_const(ds, path_to_input)
    if (("latF" not in ds) | ("lonF" not in ds)):
        ds = adds.add_lat_lon(ds, latmin, latmax, lonmin, lonmax)
    if "PRESS" not in ds:
        ds = press(ds, path_to_input=path_to_input)
    ds["SA"] = xr.apply_ufunc(gsw.conversions.SA_from_Sstar,
                              ds.SALT * ds.ups, ds.PRESS, ds.lonF, ds.latF,
                              dask='parallelized',
                              output_dtypes=[ds.SALT.dtype], keep_attrs=True)
    ds["SA"].attrs["standard_name"] = "SA"
    ds["SA"].attrs["long_name"] = "absolute salinity (g/kg)"
    ds["SA"].attrs["units"] = 'g/kg'
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
                                 output_dtypes=[ds.SA.dtype], keep_attrs=True)
    ds["alpha"].attrs["standard_name"] = "alpha"
    ds["alpha"].attrs["long_name"] = "thermal expansion coefficient (1/K)"
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
                                dask='parallelized', output_dtypes=[ds.SA.dtype], keep_attrs=True)
    ds["beta"].attrs["standard_name"] = "beta"
    ds["beta"].attrs["long_name"] = "haline contraction coefficient (kg/g)"
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
    if "alpha" not in ds.variables:
        ds = alpha(ds, path_to_input, latmin, latmax, lonmin, lonmax)
    if "beta" not in ds.variables:
        ds = beta(ds, path_to_input, latmin, latmax, lonmin, lonmax)
    ds["BFlx_SURF"] = ((ds.gravity / ds.rhoconst)
                        * ((ds.alpha.isel(Z=0) * ds.oceQnet
                            / ds.HeatCapacity_Cp)
                           - (ds.rhoconst * ds.beta.isel(Z=0)
                              * ds.isel(Z=0).SA * -ds.oceFWflx)))
    ds["BFlx_SURF"].attrs["standard_name"] = "BFlx_SURF"
    ds["BFlx_SURF"].attrs["long_name"] = "air-sea buyancy flux)"
    ds["BFlx_SURF"].attrs["units"] = ''
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
                       / ds.fU + ds["VORT"].isel(Z=0)
                       - grid.derivative(taux, "Y")
                       / ds.fU + ds["VORT"].isel(Z=0))
    ds[out_name].attrs["standard_name"] = out_name
    ds[out_name].attrs["long_name"] = "Ekman vertical velocity (m/s)"
    ds[out_name].attrs["units"] = 'm/s'
    return ds


def ice_ocean_stress(ds, grid=None, path_to_input=None,
                     thick_name="SIheff", fract_name="SIarea",
                     taux_name="SIOtaux", tauy_name="SIOtauy"):
    """
    """
    if (("rhoconst" not in ds) | ("SEAICE_waterDrag" not in ds)):
        ds = get_const(ds, path_to_input)
    if grid == None:
        metrics = {
            ('X'): ['dxC', 'dxG', 'dxF', 'dxV'], # X distances
            ('Y'): ['dyC', 'dyG', 'dyF', 'dyU'], # Y distances
            ('Z'): ['drF', 'drW', 'drS', 'drC'], # Z distances
            ('X', 'Y'): ['rAw', 'rAs', 'rA', 'rAz'] # Areas in x-y plane
            }
        grid = xgcm.Grid(ds, periodic=["X", "Y"], metrics=metrics)
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
    ds[taux_name].attrs["standard_name"] = taux_name
    ds[taux_name].attrs["long_name"] = "zonal ice-ocean stress (N/m^2)"
    ds[taux_name].attrs["units"] = 'N/m^2'
    ds[tauy_name].attrs["standard_name"] = tauy_name
    ds[tauy_name].attrs["long_name"] = "meridional ice-ocean stress (N/m^2)"
    ds[tauy_name].attrs["units"] = 'N/m^2'
    return ds
