import numpy as np
import matplotlib.pyplot as plt
import libs.visualization as vis
from scipy.io import loadmat
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.path as mpath

L_RGPS = [12.5, 25, 50, 100, 200, 400]

dataset_RGPS = vis.Arctic(fig_shape="round", save=1, resolution=12.5,)

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

# figure

dataset_RGPS.arctic_plot_RGPS(shear_sum, "mask", "80")

# fig = plt.figure(dpi=300)
# ax = plt.subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
# fig.subplots_adjust(bottom=0.05, top=0.95, left=0.04, right=0.95, wspace=0.02)
# # theta = np.linspace(0, 2 * np.pi, 100)
# # center, radius = [0.5, 0.5], 0.5
# # verts = np.vstack([np.sin(theta), np.cos(theta)]).T
# # circle = mpath.Path(verts * radius + center)
# # ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
# # ax.set_boundary(circle, transform=ax.transAxes)
# cf = ax.pcolormesh(shear_sum)
# ax.contour(shear_sum, levels=np.array([0.8]))
# fig.colorbar(cf)
# ax.add_feature(cfeature.LAND, zorder=3)
# ax.coastlines(resolution="50m", zorder=4)
# plt.show()
