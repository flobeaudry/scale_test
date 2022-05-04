import numpy as np
import matplotlib.pyplot as plt
import libs.visualization as vis
from scipy.io import loadmat

L_RGPS = [12.5, 25, 50, 100, 200, 400]

dataset_RGPS = vis.Arctic(
    fig_shape="round", save=0, resolution=12.5, fig_name_supp="_TEST_RGPS",
)

shear = np.load("RGPS_deformations/SHR.npy")
div = np.load("RGPS_deformations/DIV.npy")

ds_J = loadmat("RGPS_binary_deformations/Shr_RGPS_all_January.mat")
ds_F = loadmat("RGPS_binary_deformations/Shr_RGPS_all_February.mat")
ds_M = loadmat("RGPS_binary_deformations/Shr_RGPS_all_March.mat")
shear_J = ds_J["Shr"]
shear_F = ds_F["Shr"]
shear_M = ds_M["Shr"]
shear_JFM = np.concatenate((shear_J, shear_F, shear_M), axis=-1)
# swap continents and no data by NaNs
shear_JFM = np.where(shear_JFM >= 1e10, np.NaN, shear_JFM)
shear_JFM = np.transpose(shear_JFM[:248, :, :], (1, 0, 2))

shear_bool = ~np.isnan(shear_JFM)
shear_sum = np.nansum(shear_bool, axis=-1) / shear_JFM.shape[-1]
mask = np.where(shear_sum < 0, -1, shear_sum)
mask80 = np.where(mask >= 0.8, 1, np.NaN)

np.save("RGPS_mask/mask80JFM.npy", mask80)

fig = plt.figure(dpi=300, figsize=(4, 4))
ax = plt.subplot(1, 1, 1)
cf = ax.pcolormesh(shear_sum)
ax.contour(shear_sum, levels=np.array([0.8]))
fig.colorbar(cf)
plt.show()

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
plt.pcolormesh(dudx[..., 0])
plt.show()
du_RGPS = np.stack((dudx, dudy, dvdx, dvdy), axis=-1)

(
    deps_RGPS_du,
    shear_RGPS_du,
    div_RGPS_du,
    scale_RGPS_du,
) = dataset_RGPS.spatial_mean_du(du_RGPS, L_RGPS)

mean_deps_RGPS_du, mean_scale_RGPS_du = dataset_RGPS.scale_plot_vect(
    deps_RGPS_du,
    scale_RGPS_du,
    L_RGPS,
    save=0,
    fig_name_supp="_dedt_02_RGPS_du",
)
plt.show()
