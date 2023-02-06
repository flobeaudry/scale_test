import numpy as np
import matplotlib.pyplot as plt
import libs.visualization as vis
from scipy.io import loadmat
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cmocean
import matplotlib.cm as cm
import matplotlib.colors as colors
from libs.constants import *

L_RGPS = [12.5, 25, 50, 100, 200, 400]

dataset_RGPS = vis.Arctic(
    fig_shape="round",
    save=1,
    resolution=12.5,
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
continents = np.where(
    np.where(shear_JFM >= 1e20, 1e9, shear_JFM) < 1e10, np.NaN, 1
)
shear_JFM = np.where(shear_JFM >= 1e10, np.NaN, shear_JFM)
shear_JFM = np.transpose(shear_JFM[:248, :, :], (1, 0, 2))
continents = np.transpose(continents[:248, :, :], (1, 0, 2))
continents = continents[..., 0]

shear_bool = ~np.isnan(shear_JFM)
shear_sum = np.nansum(shear_bool, axis=-1) / shear_JFM.shape[-1]
mask = np.where(shear_sum < 0, -1, shear_sum)
mask80 = np.where(mask >= 0.8, 1, np.NaN)

np.save("RGPS_mask/mask80JFM.npy", mask80)

# figure

# dataset_RGPS.arctic_plot_RGPS(shear_sum, "mask", "80")

fig = plt.figure(dpi=300, figsize=(4, 3.2))
left, width = 0.14, 0.75
bottom, height = 0.14, 0.75
rect_scatter = [
    left,
    bottom,
    width,
    height,
]
ax = fig.add_axes(rect_scatter)

cf = ax.pcolormesh(shear_sum * 100, cmap="summer", vmin=0, vmax=100)
ax.pcolormesh(continents, cmap="cmo.topo", vmin=0, vmax=1)
ax.contour(shear_sum * 100, levels=np.array([80]))
cbar = fig.colorbar(cf)
cbar.ax.set_ylabel("Temporal presence [%]", rotation=-90, va="bottom")

fig.savefig("images/rgps80mask.png")
fig.savefig("images/rgps80mask.pdf")
