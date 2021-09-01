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
    fig_name_supp="_1997",
)

dataset10D = vis.Arctic(
    directory="output10D_1997",
    time="1997-01-01-00-00",
    expno="01",
    datatype="u",
    fig_shape="round",
    save=1,
    resolution=10,
    fig_name_supp="D_1997",
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

L10 = [20, 40, 80, 160, 320, 640]
L20 = [40, 80, 160, 320, 640]
L40 = [80, 160, 320, 640]
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

div, shear, lat, lon, xgrid, ygrid, mask, time = dataset10.nc_load(
    "RGPS12_3dg_1997/w9697n_3dys.nc"
)
print(lat.shape)
dataset10.arctic_plot_RGPS(div[0], xgrid, ygrid)

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

# ----------------------------------------------------------------------
# load data if previously saved
# ----------------------------------------------------------------------

# deps10 = np.load("processed_data/deps10.npy", allow_pickle=True)
# shear10 = np.load("processed_data/shear10.npy", allow_pickle=True)
# div10 = np.load("processed_data/div10.npy", allow_pickle=True)
# scale10 = np.load("processed_data/scale10.npy", allow_pickle=True)
# deps10D = np.load("processed_data/deps10D.npy", allow_pickle=True)
# shear10D = np.load("processed_data/shear10D.npy", allow_pickle=True)
# div10D = np.load("processed_data/div10D.npy", allow_pickle=True)
# scale10D = np.load("processed_data/scale10D.npy", allow_pickle=True)
# data_box10_visc = np.load("data10_visc.npy")

# data_box20 = np.load("data20.npy")
# data_box20_visc = np.load("data20_visc.npy")
# def20 = np.load("processed_data/def20.npy", allow_pickle=True)
# scale20 = np.load("processed_data/scale20.npy", allow_pickle=True)

# data_box40 = np.load("data40.npy")
# data_box40_visc = np.load("data40_visc.npy")
# def40 = np.load("processed_data/def40.npy", allow_pickle=True)
# scale40 = np.load("processed_data/scale40.npy", allow_pickle=True)

# ----------------------------------------------------------------------
# plots at 10 km
# ----------------------------------------------------------------------

# dataset10.pdf_plot_vect(def10, L10)
# dataset10.cdf_plot(data_box10)
# mean_deps, mean_scale = dataset10.scale_plot_vect(
#     deps10, scale10, L10, save=False, fig_name_supp="_dedt_1997"
# )
# mean_depsD, mean_scaleD = dataset10D.scale_plot_vect(
#     deps10D, scale10D, L10, save=False, fig_name_supp="D_dedt_1997"
# )
# mean_deps_stack = np.stack((mean_deps, mean_depsD), axis=1)
# mean_scale_stack = np.stack((mean_scale, mean_scaleD), axis=1)

# dataset10.multi_plot(
#     mean_deps_stack, mean_scale_stack, fig_name_supp="_dedt_1997"
# )

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
