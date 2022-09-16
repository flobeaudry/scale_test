from cmath import isnan
import numpy as np
import matplotlib.pyplot as plt
import libs.visualization as vis


# dataset_RGPS = vis.Arctic(
#     fig_shape="round", save=1, resolution=12.5, fig_name_supp="_02_RGPS",
# )

# L_RGPS = [12.5, 25, 50, 100, 200, 400, 800]
# L10 = [10, 20, 40, 80, 160, 320, 640]
# dt = "00-06-00"
# time_end = "1997-03-31-18-00"


# mask80 = np.load("RGPS_mask/mask80JFM.npy")

# # load derivatives and mask it
# dudx = dataset_RGPS.mask80_times_RGPS(
#     np.load("RGPS_derivatives/DUDX.npy"), mask80
# )
# dudy = dataset_RGPS.mask80_times_RGPS(
#     np.load("RGPS_derivatives/DUDY.npy"), mask80
# )
# dvdx = dataset_RGPS.mask80_times_RGPS(
#     np.load("RGPS_derivatives/DVDX.npy"), mask80
# )
# dvdy = dataset_RGPS.mask80_times_RGPS(
#     np.load("RGPS_derivatives/DVDY.npy"), mask80
# )

dudx = np.load("RGPS_derivatives/DUDX.npy")
dudy = np.load("RGPS_derivatives/DUDY.npy")
dvdx = np.load("RGPS_derivatives/DVDX.npy")
dvdy = np.load("RGPS_derivatives/DVDY.npy")

# stack them
du_RGPS = np.stack((dudx, dudy, dvdx, dvdy), axis=-1)

shear = np.sqrt((dudx - dvdy) ** 2 + (dudy + dvdx) ** 2)
div = dudx + dvdy
deps = np.sqrt(shear ** 2 + div ** 2)
index = np.where(~np.isnan(deps))
deps = deps[index]

print(np.min(deps), np.max(deps))
n = np.logspace(np.log10(5e-3), 0, num=25)
p, x = np.histogram(deps.flatten(), bins=n, density=1)
x = (x[:-1] + x[1:]) / 2

plt.loglog(x, p)
plt.show()

# (
#     deps_RGPS_du,
#     shear_RGPS_du,
#     div_RGPS_du,
#     scale_RGPS_du,
# ) = dataset_RGPS.spatial_mean_du(du_RGPS, L_RGPS)

# (
#     deps_RGPS_du2,
#     shear_RGPS_du2,
#     div_RGPS_du2,
#     scale_RGPS_du2,
# ) = dataset_RGPS.spatial_mean_RGPS_du(du_RGPS, L_RGPS)

# mean_deps_RGPS_du, mean_scale_RGPS_du = dataset_RGPS.scale_plot_vect(
#     deps_RGPS_du,
#     scale_RGPS_du,
#     L_RGPS,
#     save=0,
#     fig_name_supp="_dedt_02_SRGPS_du_nomask",
# )

# mean_deps_RGPS_du2, mean_scale_RGPS_du2 = dataset_RGPS.scale_plot_vect(
#     deps_RGPS_du2,
#     scale_RGPS_du2,
#     L_RGPS,
#     save=0,
#     fig_name_supp="_dedt_02_SRGPS_du_nomask",
# )

# print(
#     np.nanmean(np.nanmean(deps, axis=-1)),
#     mean_deps_RGPS_du,
#     mean_deps_RGPS_du2,
# )

# plt.pcolormesh(np.nanmean(deps, axis=-1), vmin=0, vmax=0.1)
# plt.colorbar()
# plt.show()
