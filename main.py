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
)

dataset10D = vis.Arctic(
    directory="output10D_1997",
    time="1997-01-01-00-00",
    expno="01",
    datatype="u",
    fig_shape="round",
    save=1,
    resolution=10,
    fig_name_supp="D_97",
    fig_type="png",
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
)


dataset_RGPS = vis.Arctic(
    fig_shape="round",
    save=1,
    resolution=12.5,
    fig_name_supp="_02_RGPS",
    fig_type="png",
)

# ----------------------------------------------------------------------

dt = "00-06-00"
time_end = "1997-01-31-18-00"
time_end_02 = "2002-01-31-18-00"
arctic_plots = 0

# ----------------------------------------------------------------------
# dedt plots
# ----------------------------------------------------------------------

if arctic_plots == 1:
    dedt_plot = dataset10.multi_load(
        datatype="dedt", time_end="2002-01-03-18-00", dt=dt
    )
    dedt_plot_Dadv = dataset10Dadv.multi_load(
        datatype="dedt", time_end="2002-01-03-18-00", dt=dt
    )
    dudx = np.load("RGPS_derivatives/DUDX.npy")
    dudy = np.load("RGPS_derivatives/DUDY.npy")
    dvdx = np.load("RGPS_derivatives/DVDX.npy")
    dvdy = np.load("RGPS_derivatives/DVDY.npy")
    du80_RGPS = np.stack((dudx, dudy, dvdx, dvdy), axis=-1)
    deps_RGPS_plot = dataset_RGPS._deformation(du80_RGPS, 0)

    dedt_plot_ta = dataset10D._time_average(dedt_plot, dt)
    dedt_plot_ta_Dadv = dataset10Dadv._time_average(dedt_plot_Dadv, dt)

    dataset_RGPS.arctic_plot_RGPS(deps_RGPS_plot[..., 0], "dedt", "_02_")
    dataset10.arctic_plot(dedt_plot_ta[..., 0])
    dataset10Dadv.arctic_plot(dedt_plot_ta_Dadv[..., 0])

# ----------------------------------------------------------------------
# load everthing, compute du, mask du with RGPS80
# ----------------------------------------------------------------------

# mask80
# mask80 = dataset_RGPS.mask80("RGPS_data", ti=-1, tf=88)
mask80 = np.load("RGPS_mask/mask80JFM.npy")

# calcul time averaged
u_v = dataset10.multi_load(dt, time_end_02)
u_v_D = dataset10D.multi_load(dt, time_end)
u_v_Dadv = dataset10Dadv.multi_load(dt, time_end_02)
u_v = np.where(u_v == 0, np.NaN, u_v)
u_v_D = np.where(u_v_D == 0, np.NaN, u_v_D)
u_v_Dadv = np.where(u_v_Dadv == 0, np.NaN, u_v_Dadv)
u_v_ta = dataset10._time_average(u_v, dt)
u_v_ta_D = dataset10D._time_average(u_v_D, dt)
u_v_ta_Dadv = dataset10Dadv._time_average(u_v_Dadv, dt)

# calcul du
du = dataset10._derivative(u_v_ta[:, :, 0, :], u_v_ta[:, :, 1, :])
du_D = dataset10D._derivative(u_v_ta_D[:, :, 0, :], u_v_ta_D[:, :, 1, :])
du_Dadv = dataset10Dadv._derivative(
    u_v_ta_Dadv[:, :, 0, :], u_v_ta_Dadv[:, :, 1, :]
)
# mask the data
du80 = du
du80_D = du_D
du80_Dadv = du_Dadv
du80[..., 0], mask = dataset10.mask80_times(du[..., 0], mask80)
du80[..., 1] = dataset10.mask80_times(du[..., 1], mask80)[0]
du80[..., 2] = dataset10.mask80_times(du[..., 2], mask80)[0]
du80[..., 3] = dataset10.mask80_times(du[..., 3], mask80)[0]
du80_D[..., 0] = dataset10D.mask80_times(du_D[..., 0], mask80)[0]
du80_D[..., 1] = dataset10D.mask80_times(du_D[..., 1], mask80)[0]
du80_D[..., 2] = dataset10D.mask80_times(du_D[..., 2], mask80)[0]
du80_D[..., 3] = dataset10D.mask80_times(du_D[..., 3], mask80)[0]
du80_Dadv[..., 0] = dataset10Dadv.mask80_times(du_Dadv[..., 0], mask80)[0]
du80_Dadv[..., 1] = dataset10Dadv.mask80_times(du_Dadv[..., 1], mask80)[0]
du80_Dadv[..., 2] = dataset10Dadv.mask80_times(du_Dadv[..., 2], mask80)[0]
du80_Dadv[..., 3] = dataset10Dadv.mask80_times(du_Dadv[..., 3], mask80)[0]

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
# plot PDF and CDF
# ----------------------------------------------------------------------

du80_stack = [du80_RGPS, du80, du80_Dadv]

dataset10.pdf_du(du80_stack, save=1, fig_name_supp="_02")
dataset10.cdf_du(du80_stack, save=1, fig_name_supp="_02")

# ----------------------------------------------------------------------
# SCALING
# ----------------------------------------------------------------------

# calcul du sclaing spatial
deps10, shear10, div10, scale10 = dataset10.spatial_mean_du(du80, L10)

deps10D, shear10D, div10D, scale10D = dataset10D.spatial_mean_du(du80_D, L10)

(
    deps10Dadv,
    shear10Dadv,
    div10Dadv,
    scale10Dadv,
) = dataset10Dadv.spatial_mean_du(du80_Dadv, L10)

(
    deps_RGPS_du,
    shear_RGPS_du,
    div_RGPS_du,
    scale_RGPS_du,
) = dataset_RGPS.spatial_mean_du(du80_RGPS, L_RGPS)

# meme chose mais pour le scaling temporel
deps10_T, shear10_T, div10_T, scale10_T = dataset10.temporal_mean_du(du80, T10)

deps10D_T, shear10D_T, div10D_T, scale10D_T = dataset10D.temporal_mean_du(
    du80_D, T10
)

(
    deps10Dadv_T,
    shear10Dadv_T,
    div10Dadv_T,
    scale10Dadv_T,
) = dataset10Dadv.temporal_mean_du(du80_Dadv, T10)

(
    deps_RGPS_du_T,
    shear_RGPS_du_T,
    div_RGPS_du_T,
    scale_RGPS_du_T,
) = dataset_RGPS.temporal_mean_du(du80_RGPS, T10)

# ----------------------------------------------------------------------
# plots at 10 km
# ----------------------------------------------------------------------

mean_deps, mean_scale, __ = dataset10.scale_plot_vect(
    deps10, scale10, L10, save=0, fig_name_supp="_dedt_02"
)
mean_depsD, mean_scaleD, __ = dataset10D.scale_plot_vect(
    deps10D, scale10D, L10, save=0, fig_name_supp="D_dedt_97"
)
mean_depsDadv, mean_scaleDadv, __ = dataset10Dadv.scale_plot_vect(
    deps10Dadv, scale10Dadv, L10, save=0, fig_name_supp="Dadv_dedt_02"
)

# ----------------------------------------------------------------------
# plots RGPS 12.5 km
# ----------------------------------------------------------------------

(mean_deps_RGPS_du, mean_scale_RGPS_du, __,) = dataset_RGPS.scale_plot_vect(
    deps_RGPS_du,
    scale_RGPS_du,
    L_RGPS,
    save=0,
    fig_name_supp="_dedt_02_RGPS_du",
)

# ----------------------------------------------------------------------
# multiplot scaling
# ----------------------------------------------------------------------

mean_deps_stack = np.stack(
    (mean_deps_RGPS_du, mean_deps, mean_depsDadv), axis=1,
)
mean_scale_stack = np.stack(
    (mean_scale_RGPS_du, mean_scale, mean_scaleDadv), axis=1,
)

mean_deps_stack_T = np.stack((deps_RGPS_du_T, deps10_T, deps10Dadv_T), axis=1,)
mean_scale_stack_T = np.stack(
    (scale_RGPS_du_T, scale10_T, scale10Dadv_T), axis=1,
)

dataset10.multi_plot_spatial(
    mean_deps_stack, mean_scale_stack, fig_name_supp="_dedt_02"
)

dataset10.multi_plot_temporal(
    mean_deps_stack_T, mean_scale_stack_T, fig_name_supp="_dedt_02"
)

# ----------------------------------------------------------------------
# multifractality
# ----------------------------------------------------------------------

# spatial
parameters10, coeff10 = dataset10.multifractal_spatial(3, deps10, scale10,)

parameters10D, coeff10D = dataset10D.multifractal_spatial(
    3, deps10D, scale10D,
)

parameters10Dadv, coeff10Dadv = dataset10Dadv.multifractal_spatial(
    3, deps10Dadv, scale10Dadv,
)

parametersRGPS, coeffRGPS = dataset_RGPS.multifractal_spatial(
    3, deps_RGPS_du, scale_RGPS_du, RGPS=1,
)

param_stack = np.stack(
    (parametersRGPS, parameters10, parameters10Dadv), axis=1,
)

coeff_stack = np.stack((coeffRGPS, coeff10, coeff10Dadv), axis=1,)

dataset10.multifractal_plot(param_stack, coeff_stack, 3, 1, "_param_02")

# temporal
parameters10_T, coeff10_T = dataset10.multifractal_temporal(3, du80)

parameters10D_T, coeff10D_T = dataset10D.multifractal_temporal(3, du80_D)

parameters10Dadv_T, coeff10Dadv_T = dataset10Dadv.multifractal_temporal(
    3, du80_Dadv
)

parametersRGPS_T, coeffRGPS_T = dataset_RGPS.multifractal_temporal(
    3, du80_RGPS
)

param_stack_T = np.stack(
    (parametersRGPS_T, parameters10_T, parameters10Dadv_T), axis=1,
)

coeff_stack_T = np.stack((coeffRGPS_T, coeff10_T, coeff10Dadv_T), axis=1,)

dataset10.multifractal_plot(
    param_stack_T, coeff_stack_T, 3, 1, "T_param_02", temp=1
)

