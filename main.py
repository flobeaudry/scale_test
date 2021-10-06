import libs.visualization as vis
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# User input for the location of the files
# ----------------------------------------------------------------------

dataset10 = vis.Arctic(
    directory="output10_1997",
    time="1997-01-01-00-00",
    expno="12",
    datatype="u",
    fig_shape="round",
    save=1,
    resolution=10,
    fig_name_supp="_97",
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
)

dataset_RGPS = vis.Arctic(
    fig_shape="round", save=1, resolution=12.5, fig_name_supp="_02_RGPS",
)

# dataset20 = vis.Arctic(
#     directory="output20",
#     time="1997-01-01-00-00",
#     expno="01",
#     datatype="u",
#     fig_shape="round",
#     save=1,
#     resolution=20,
#     fig_name_supp="_1997",
# )

# dataset40 = vis.Arctic(
#     directory="output40_1997",
#     time="1997-01-01-00-00",
#     expno="01",
#     datatype="u",
#     fig_shape="round",
#     save=1,
#     resolution=40,
#     fig_name_supp="_1997",
# )

# ----------------------------------------------------------------------

# dataset10.arctic_plot(dataset10.load())
# dataset.multi_load("01-00-00", "1997-03-31-00-00")

L_RGPS = [12.5, 25, 50, 100, 200, 400]
L10 = [10, 20, 40, 80, 160, 320, 640]
L20 = [20, 40, 80, 160, 320, 640]
L40 = [40, 80, 160, 320, 640]
dt = "00-06-00"
time_end = "1997-03-31-18-00"

# ----------------------------------------------------------------------

# h = dataset10.load(datatype="h")
# A = dataset10.load(datatype="A")
# zeta_load = dataset10.load(datatype="viscosity")
# zeta_load = zeta_load[1:-1, 1:-1]

# uv = dataset10.load(datatype="u")
# du, dv = dataset10._derivative(uv[..., 0], uv[..., 1])

# delta = dataset10._delta(du, dv)
# p = dataset10._pressure(h, A)

# zeta, zeta_max = dataset10._viscosity(p, delta)

# import matplotlib.colors

# fig = plt.figure(dpi=300)
# ax = plt.gca()
# fig.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.95, wspace=0.02)
# cf = ax.scatter(
#     p.flatten() / (2 * zeta_max.flatten() * delta.flatten()),
#     zeta_load.flatten() / zeta_max.flatten(),
#     c=p.flatten(),
#     s=0.5,
#     norm=matplotlib.colors.LogNorm(),
# )
# ax.set_yscale("log")
# ax.set_xscale("log")
# ax.set_xlabel(r"$\frac{P}{2\Delta\zeta}$")
# ax.set_ylabel(r"$1-\frac{\zeta}{\zeta_{max}}$", rotation=0)
# ax.set_ylim(0.2, 1.1)
# cbar = fig.colorbar(cf)
# cbar.set_label(
#     "$P$ [N$\cdot$m$^{-1}$]",
#     rotation=-90,
#     va="bottom",
# )
# fig.savefig("images/zeta_vs_delta10.png")

# ----------------------------------------------------------------------
# compute all mean deformations in boxes
# ----------------------------------------------------------------------

# deps10, shear10, div10, scale10 = dataset10.spatial_mean_vect(
#     dataset10.multi_load(dt, time_end), L10, dt
# )
# deps10D, shear10D, div10D, scale10D = dataset10D.spatial_mean_vect(
#     dataset10D.multi_load(dt, time_end), L10, dt
# )

# data_box20, data_box20_visc = dataset20.spatial_mean_box(
#     dataset20.multi_load(dt, time_end), L20, dt, time_end, from_velocity=1
# )
# def20, scale20 = dataset20.spatial_mean_vect(
#     dataset20.multi_load(dt, time_end), L20, dt
# )

# data_box40, data_box40_visc = dataset40.spatial_mean_box(
#     dataset40.multi_load(dt, time_end), L40, dt, time_end, from_velocity=1
# )
# def40, scale40 = dataset40.spatial_mean_vect(
#     dataset40.multi_load(dt, time_end), L40, dt
# )

# mask80
mask80 = dataset_RGPS.mask80("RGPS_data", tt=30)

# RGPS data
# load everything
deps, div, shear = dataset_RGPS.nc_load("RGPS_data/w0102n_3dys.nc", tt=30)
# plot initial values everything
# dataset_RGPS.arctic_plot_RGPS(mask80, "mask", mask=1)
# dataset_RGPS.arctic_plot_RGPS(div[..., 0], "div", "_02_")
# dataset_RGPS.arctic_plot_RGPS(deps[..., 0], "dedt", "_02_")
# dataset_RGPS.arctic_plot_RGPS(deps[..., 0], "shear", "_02_")
# mask it using mask80
shear80 = dataset_RGPS.mask80_times_RGPS(shear, mask80)
div80 = dataset_RGPS.mask80_times_RGPS(div, mask80)
# split the divergence in -/+
ndiv80 = np.where(div80 < 0, div80, np.NaN)
pdiv80 = np.where(div80 > 0, div80, np.NaN)
# remove nans
shear80_cut = shear80[~np.isnan(shear80)]
ndiv80_cut = ndiv80[~np.isnan(ndiv80)]
pdiv80_cut = pdiv80[~np.isnan(pdiv80)]

# # my data
# # load everything
# shear10 = dataset10.multi_load(dt, "1997-01-31-18-00", datatype="shear")
# div10 = dataset10.multi_load(dt, "1997-01-31-18-00", datatype="divergence")
# # time average it
# shear10 = dataset10._time_average(shear10, dt)
# div10 = dataset10._time_average(div10, dt)
# # mask it using mask 80
# shear10 = dataset10.mask80_times(shear10, mask80)
# div10 = dataset10.mask80_times(div10, mask80)
# # split the divergence in -/+
# ndiv10 = np.where(div10 < 0, div10, np.NaN)
# pdiv10 = np.where(div10 > 0, div10, np.NaN)
# # get rid of nans
# shear10_cut = shear10[~np.isnan(shear10)]
# ndiv10_cut = ndiv10[~np.isnan(ndiv10)]
# pdiv10_cut = pdiv10[~np.isnan(pdiv10)]

# # damage data
# # load everything
# shear10D = dataset10D.multi_load(dt, "1997-01-31-18-00", datatype="shear")
# div10D = dataset10D.multi_load(dt, "1997-01-31-18-00", datatype="divergence")
# # time average it
# shear10D = dataset10D._time_average(shear10D, dt)
# div10D = dataset10D._time_average(div10D, dt)
# # mask it using mask 80
# shear10D = dataset10D.mask80_times(shear10D, mask80)
# div10D = dataset10D.mask80_times(div10D, mask80)
# # split the divergence in -/+
# ndiv10D = np.where(div10D < 0, div10D, np.NaN)
# pdiv10D = np.where(div10D > 0, div10D, np.NaN)
# # get rid of nans
# shear10D_cut = shear10D[~np.isnan(shear10D)]
# ndiv10D_cut = ndiv10D[~np.isnan(ndiv10D)]
# pdiv10D_cut = pdiv10D[~np.isnan(pdiv10D)]
# print(shear10D_cut.shape, shear80_cut.shape)

# make the pdf plots for each of them
dataset_RGPS.pdf_plot_vect(shear80_cut, -ndiv80_cut, pdiv80_cut)
# dataset10.pdf_plot_vect(shear10_cut, -ndiv10_cut, pdiv10_cut)
# dataset10D.pdf_plot_vect(shear10D_cut, -ndiv10D_cut, pdiv10D_cut)

# compute scaling of RGPS
(
    deps_RGPS,
    shear_RGPS,
    div_RGPS,
    deps_scale_RGPS,
    shear_scale_RGPS,
    div_scale_RGPS,
) = dataset_RGPS.spatial_mean_RGPS(shear80, div80, L_RGPS)

# ----------------------------------------------------------------------
# save data in file
# ----------------------------------------------------------------------

# np.save("processed_data/deps10D.npy", deps10D)
# np.save("processed_data/shear10D.npy", shear10D)
# np.save("processed_data/div10D.npy", div10D)
# np.save("processed_data/scale10D.npy", scale10D)
# np.save("processed_data/deps10.npy", deps10)
# np.save("processed_data/shear10.npy", shear10)
# np.save("processed_data/div10.npy", div10)
# np.save("processed_data/scale10.npy", scale10)
# np.save("data10_visc.npy", data_box10_visc)

# np.save("data20.npy", data_box20)
# np.save("data20_visc.npy", data_box20_visc)
# np.save("processed_data/def20.npy", def20)
# np.save("processed_data/scale20.npy", scale20)

# np.save("data40.npy", data_box40)
# np.save("data40_visc.npy", data_box40_visc)
# np.save("processed_data/def40.npy", def40)
# np.save("processed_data/scale40.npy", scale40)

# np.save("processed_data/deps_RGPS.npy", deps_RGPS)
# np.save("processed_data/shear_RGPS.npy", shear_RGPS)
# np.save("processed_data/div_RGPS.npy", div_RGPS)
# np.save("processed_data/deps_scale_RGPS.npy", deps_scale_RGPS)
# np.save("processed_data/shear_scale_RGPS.npy", shear_scale_RGPS)
# np.save("processed_data/div_scale_RGPS.npy", div_scale_RGPS)

# ----------------------------------------------------------------------
# load data if previously saved
# ----------------------------------------------------------------------

deps10 = np.load("processed_data/deps10.npy", allow_pickle=True)
shear10 = np.load("processed_data/shear10.npy", allow_pickle=True)
div10 = np.load("processed_data/div10.npy", allow_pickle=True)
scale10 = np.load("processed_data/scale10.npy", allow_pickle=True)
deps10D = np.load("processed_data/deps10D.npy", allow_pickle=True)
shear10D = np.load("processed_data/shear10D.npy", allow_pickle=True)
div10D = np.load("processed_data/div10D.npy", allow_pickle=True)
scale10D = np.load("processed_data/scale10D.npy", allow_pickle=True)
# data_box10_visc = np.load("data10_visc.npy")

# data_box20 = np.load("data20.npy")
# data_box20_visc = np.load("data20_visc.npy")
# def20 = np.load("processed_data/def20.npy", allow_pickle=True)
# scale20 = np.load("processed_data/scale20.npy", allow_pickle=True)

# data_box40 = np.load("data40.npy")
# data_box40_visc = np.load("data40_visc.npy")
# def40 = np.load("processed_data/def40.npy", allow_pickle=True)
# scale40 = np.load("processed_data/scale40.npy", allow_pickle=True)

# deps_RGPS = np.load("processed_data/deps_RGPS.npy", allow_pickle=True)
# shear_RGPS = np.load("processed_data/shear_RGPS.npy", allow_pickle=True)
# div_RGPS = np.load("processed_data/div_RGPS.npy", allow_pickle=True)
# deps_scale_RGPS = np.load(
#     "processed_data/deps_scale_RGPS.npy", allow_pickle=True
# )
# shear_scale_RGPS = np.load(
#     "processed_data/shear_scale_RGPS.npy", allow_pickle=True
# )
# div_scale_RGPS = np.load(
#     "processed_data/div_scale_RGPS.npy", allow_pickle=True
# )

# ----------------------------------------------------------------------
# plots at 10 km
# ----------------------------------------------------------------------

# dataset10.pdf_plot_vect(def10, L10)
# dataset10.cdf_plot(data_box10)
mean_deps, mean_scale = dataset10.scale_plot_vect(
    deps10, scale10, L10, save=0, fig_name_supp="_dedt_97"
)
mean_depsD, mean_scaleD = dataset10D.scale_plot_vect(
    deps10D, scale10D, L10, save=0, fig_name_supp="D_dedt_97"
)

# dataset10.arctic_plot(dataset10.load())
# dataset10D.arctic_plot(dataset10D.load())

# ----------------------------------------------------------------------
# plots at 20 km
# ----------------------------------------------------------------------

# dataset20.pdf_plot(data_box20)
# dataset20.cdf_plot(data_box20)
# dataset20.scale_plot(data_box20, L20, data_box20_visc)
# dataset20.scale_plot_vect(def20, scale20, L20)

# ----------------------------------------------------------------------
# plots at 40 km
# ----------------------------------------------------------------------

# dataset40.pdf_plot(data_box40)
# dataset40.cdf_plot(data_box40)
# dataset40.scale_plot(data_box40, L40, data_box40_visc)
# mean_def, mean_scale = dataset40.scale_plot_vect(def40, scale40, L40)

# ----------------------------------------------------------------------
# plots RGPS 12.5 km
# ----------------------------------------------------------------------

mean_deps_RGPS, mean_scale_RGPS = dataset_RGPS.scale_plot_vect(
    deps_RGPS, deps_scale_RGPS, L_RGPS, save=1, fig_name_supp="_dedt_02_RGPS",
)

# ----------------------------------------------------------------------
# multiplot
# ----------------------------------------------------------------------

mean_deps_stack = np.stack(
    (mean_deps[0:6], mean_depsD[0:6], mean_deps_RGPS), axis=1
)
mean_scale_stack = np.stack(
    (mean_scale[0:6], mean_scaleD[0:6], mean_scale_RGPS), axis=1
)

dataset10.multi_plot(
    mean_deps_stack, mean_scale_stack, fig_name_supp="_dedt_97"
)
