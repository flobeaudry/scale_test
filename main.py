import libs.visualization as vis
import numpy as np
from libs.constants import *

# ----------------------------------------------------------------------
# User input for the location of the files
# ----------------------------------------------------------------------

dataset10 = vis.Arctic(
    directory="output10_2002",
    time="2002-01-01-00-00",
    expno="05",
    datatype="u",
    fig_shape="round",
    save=1,
    resolution=10,
    fig_name_supp="_02",
    fig_type="png",
    trans=True,
)

dataset10Dadv2 = vis.Arctic(
    directory="output10Dadv_2002_2",
    time="2002-01-01-00-00",
    expno="07",
    datatype="u",
    fig_shape="round",
    save=1,
    resolution=10,
    fig_name_supp="Dadv_02_2",
    fig_type="png",
    trans=True,
)

dataset10Dadv = vis.Arctic(
    directory="output10Dadv_2002",
    time="2002-01-01-00-00",
    expno="06",
    datatype="u",
    fig_shape="round",
    save=1,
    resolution=10,
    fig_name_supp="Dadv_02",
    fig_type="png",
    trans=True,
)

dataset83 = vis.Arctic(
    directory="output83",
    time="2002-01-01-00-00",
    expno="83",
    datatype="u",
    fig_shape="round",
    save=1,
    resolution=10,
    fig_name_supp="83",
    fig_type="png",
    trans=True,
)

dataset85 = vis.Arctic(
    directory="output85",
    time="2002-01-01-00-00",
    expno="85",
    datatype="u",
    fig_shape="round",
    save=1,
    resolution=10,
    fig_name_supp="85",
    fig_type="png",
    trans=True,
)

# dataset93 = vis.Arctic(
#     directory="output93",
#     time="2002-01-01-00-00",
#     expno="93",
#     datatype="u",
#     fig_shape="round",
#     save=1,
#     resolution=10,
#     fig_name_supp="93",
#     fig_type="png",
#     trans=True,
# )

dataset95 = vis.Arctic(
    directory="output95",
    time="2002-01-01-00-00",
    expno="95",
    datatype="u",
    fig_shape="round",
    save=1,
    resolution=10,
    fig_name_supp="95",
    fig_type="png",
    trans=True,
)

dataset_RGPS = vis.Arctic(
    fig_shape="round",
    save=1,
    resolution=12.5,
    fig_name_supp="_02_RGPS",
    fig_type="png",
    trans=True,
)

# ----------------------------------------------------------------------

dt = "00-06-00"
time_end = "2002-01-31-18-00"
datasets = np.array(
    [dataset10]
    #     , dataset10Dadv, dataset10Dadv2, dataset83, dataset85, dataset95]
)
arctic_plots = 0

# ----------------------------------------------------------------------
# dedt plots
# ----------------------------------------------------------------------

if arctic_plots == 1:
    for dataset in datasets:
        dedt_plot = dataset.multi_load(
            datatype="dedt", time_end="2002-01-03-18-00", dt=dt
        )
        dedt_plot_Dadv = dataset10Dadv.multi_load(
            datatype="dedt", time_end="2002-01-03-18-00", dt=dt
        )
        dedt_plot_Dadv2 = dataset10Dadv2.multi_load(
            datatype="dedt", time_end="2002-01-03-18-00", dt=dt
        )
        dedt_plot_ta = dataset._time_average(dedt_plot, dt)
        dataset.arctic_plot(dedt_plot_ta[..., 0])

    dudx = np.load("RGPS_derivatives/DUDX.npy")
    dudy = np.load("RGPS_derivatives/DUDY.npy")
    dvdx = np.load("RGPS_derivatives/DVDX.npy")
    dvdy = np.load("RGPS_derivatives/DVDY.npy")
    du80_RGPS = np.stack((dudx, dudy, dvdx, dvdy), axis=-1)
    deps_RGPS_plot = dataset_RGPS._deformation(du80_RGPS, 0)

    dataset_RGPS.arctic_plot_RGPS(deps_RGPS_plot[..., 0], "dedt", "_02_")

    quit()

# ----------------------------------------------------------------------
# load everthing, compute du, mask du with RGPS80
# ----------------------------------------------------------------------

# mask80
# mask80 = dataset_RGPS.mask80("RGPS_data", ti=-1, tf=88)
mask80 = np.load("RGPS_mask/mask80JFM.npy")

# RGPS data
# deps, div, shear = dataset_RGPS.nc_load("RGPS_data/w0102n_3dys.nc", tf=29)
# load derivatives and mask it
dudx = dataset_RGPS.mask80_times_RGPS(
    np.load("RGPS_derivatives/DUDX.npy"), mask80
)
dudy = dataset_RGPS.mask80_times_RGPS(
    np.load("RGPS_derivatives/DUDY.npy"), mask80
)
dvdx = dataset_RGPS.mask80_times_RGPS(
    np.load("RGPS_derivatives/DVDX.npy"), mask80
)
dvdy = dataset_RGPS.mask80_times_RGPS(
    np.load("RGPS_derivatives/DVDY.npy"), mask80
)

# stack them
du80_RGPS = np.stack((dudx, dudy, dvdx, dvdy), axis=-1)

# ----------------------------------------------------------------------
# Scaling spatial
# ----------------------------------------------------------------------
(
    deps_RGPS_du,
    shear_RGPS_du,
    div_RGPS_du,
    scale_RGPS_du,
) = dataset_RGPS.spatial_mean_du(du80_RGPS, L_RGPS)

# ----------------------------------------------------------------------
# Scaling temporal
# ----------------------------------------------------------------------
(
    deps_RGPS_du_T,
    shear_RGPS_du_T,
    div_RGPS_du_T,
    scale_RGPS_du_T,
) = dataset_RGPS.temporal_mean_du(du80_RGPS, T10)

# ----------------------------------------------------------------------
# Means (only for spatial)
# ----------------------------------------------------------------------
(mean_deps_RGPS_du, mean_scale_RGPS_du, __,) = dataset_RGPS.scale_plot_vect(
    deps_RGPS_du,
    scale_RGPS_du,
    L_RGPS,
    save=0,
    fig_name_supp="_dedt_02_RGPS_du",
)
# ----------------------------------------------------------------------
# multifractality
# ----------------------------------------------------------------------
# spatial
parametersRGPS, coeffRGPS = dataset_RGPS.multifractal_spatial(
    3, deps_RGPS_du, scale_RGPS_du, RGPS=1,
)

# temporal
parametersRGPS_T, coeffRGPS_T = dataset_RGPS.multifractal_temporal(
    3, du80_RGPS
)

# ----------------------------------------------------------------------
# initialize stack of deformations and means with RGPS
# ----------------------------------------------------------------------
du80_stack = [du80_RGPS]

mean_deps_stack = np.empty((mean_deps_RGPS_du.shape[0], len(datasets) + 1))
mean_scale_stack = np.empty((mean_scale_RGPS_du.shape[0], len(datasets) + 1))

mean_deps_stack_T = np.empty((deps_RGPS_du_T.shape[0], len(datasets) + 1))
mean_scale_stack_T = np.empty((scale_RGPS_du_T.shape[0], len(datasets) + 1))

param_stack = np.empty((parametersRGPS.shape[0], len(datasets) + 1))
coeff_stack = np.empty((coeffRGPS.shape[0], len(datasets) + 1))

param_stack_T = np.empty((parametersRGPS_T.shape[0], len(datasets) + 1))
coeff_stack_T = np.empty((coeffRGPS_T.shape[0], len(datasets) + 1))

mean_deps_stack[:, 0] = mean_deps_RGPS_du
mean_scale_stack[:, 0] = mean_scale_RGPS_du

mean_deps_stack_T[:, 0] = deps_RGPS_du_T
mean_scale_stack_T[:, 0] = scale_RGPS_du_T

param_stack[:, 0] = parametersRGPS
coeff_stack[:, 0] = coeffRGPS

param_stack_T[:, 0] = parametersRGPS_T
coeff_stack_T[:, 0] = coeffRGPS_T

# ----------------------------------------------------------------------
# loop on all sim datasets
# ----------------------------------------------------------------------
j = 1
for dataset in datasets:
    # calcul time averaged
    u_v = dataset.multi_load(dt, time_end)
    u_v = np.where(u_v == 0, np.NaN, u_v)
    u_v_ta = dataset._time_average(u_v, dt)

    # calcul du
    du = dataset._derivative(u_v_ta[:, :, 0, :], u_v_ta[:, :, 1, :])

    # mask the data
    du80 = du
    du80[..., 0] = dataset.mask80_times(du[..., 0], mask80)[0]
    du80[..., 1] = dataset.mask80_times(du[..., 1], mask80)[0]
    du80[..., 2] = dataset.mask80_times(du[..., 2], mask80)[0]
    du80[..., 3] = dataset.mask80_times(du[..., 3], mask80)[0]

    # ------------------------------------------------------------------
    #       Scaling
    # ------------------------------------------------------------------

    # spatial
    deps, shear, div, scale = dataset.spatial_mean_du(du80, L10)

    # temporal
    deps_T, shear_T, div_T, scale_T = dataset.temporal_mean_du(du80, T10)

    # means (only spatial)
    mean_deps, mean_scale, __ = dataset.scale_plot_vect(
        deps, scale, L10, save=0, fig_name_supp="_dedt_02"
    )

    # ------------------------------------------------------------------
    #       Multifractality
    # ------------------------------------------------------------------

    # spatial
    parameters, coeff = dataset.multifractal_spatial(3, deps, scale,)

    # temporal
    parameters_T, coeff_T = dataset.multifractal_temporal(3, du80)

    # ------------------------------------------------------------------
    # append to lists
    # ------------------------------------------------------------------
    du80_stack.append(du80)

    mean_deps_stack[:, j] = mean_deps
    mean_scale_stack[:, j] = mean_scale

    mean_deps_stack_T[:, j] = deps_T
    mean_scale_stack_T[:, j] = scale_T

    param_stack[:, j] = parameters
    coeff_stack[:, j] = coeff

    param_stack_T[:, j] = parameters_T
    coeff_stack_T[:, j] = coeff_T

    j += 1

# ----------------------------------------------------------------------
# plot PDF and CDF
# ----------------------------------------------------------------------

dataset10.pdf_du(du80_stack, save=1, fig_name_supp="_02")
dataset10.cdf_du(du80_stack, save=1, fig_name_supp="_02")

# ----------------------------------------------------------------------
# multiplot scaling
# ----------------------------------------------------------------------
dataset10.multi_plot_spatial(
    mean_deps_stack, mean_scale_stack, fig_name_supp="_dedt_02"
)

dataset10.multi_plot_temporal(
    mean_deps_stack_T, mean_scale_stack_T, fig_name_supp="_dedt_02"
)

# ----------------------------------------------------------------------
# multifractality plots
# ----------------------------------------------------------------------
dataset10.multifractal_plot(param_stack, coeff_stack, 3, 1, "_param_02")

dataset10.multifractal_plot(
    param_stack_T, coeff_stack_T, 3, 1, "T_param_02", temp=1
)
