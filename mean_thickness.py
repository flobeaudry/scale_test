import numpy as np
from libs.constants import *
from libs.namelist import *
from libs.datasets import *
import matplotlib.pyplot as plt
import netCDF4 as nc
import scipy.interpolate as sci


datasets = np.array(
    [
        dataset29,
        dataset66,
        dataset10Dadv,
        dataset23,
        dataset25,
        dataset65,
        dataset44,
        dataset45,
        dataset67,
        dataset68,
        dataset33,
        dataset35,
    ]
)

datasets_name = np.array(
    [
        # "RGPS",
        "Control",
        "VP(0.7)",
        "VPd(2,1,30)",
        "VPd(2,3,30)",
        "VPd(2,5,30)",
        "VPd(0.7,5,30)",
        "VPd(2,5,30,0.5)",
        "VPd(2,5,30,2)",
        "VPd(2,5,30,35e3)",
        "VPd(2,5,30,55e3)",
        "VPd(2,3,2)",
        "VPd(2,5,2)",
    ]
)

# massage obs
fn = "SMOS_Icethickness_v3.3_north_20110130.nc"
temp = nc.Dataset(fn)
obs = temp["sea_ice_thickness"][0]
interpy = np.arange(520)
interpx = np.arange(440)
interp = sci.RectBivariateSpline(
    np.arange(obs.shape[0]), np.arange(obs.shape[1]), obs
)
obs_interp = interp(interpx, interpy)

# plots
xplot = np.arange(0, datasets.shape[0])
mean_thick = np.zeros((datasets.shape[0]))
thickness = np.zeros((datasets.shape[0], 440, 520))

mask80 = np.load("RGPS_mask/mask80JFM.npy")
fig, axss = dataset10.multi_fig_precond(x, y, total)

# multi plot of diff thickness
for i, dataset in enumerate(datasets):
    h_raw = dataset.multi_load(dt, time_end, datatype="h")
    # h_mask = dataset.mask80_times(h_raw[:, :, :], mask80)[0]
    h = np.where(h_raw == 0, np.NaN, h_raw)
    thickness[i] = np.nanmean(h, axis=-1)
    mean_thick[i] = np.nanmean(h)
    if i == 0:
        dataset.arctic_plot(
            thickness[0],
            title=datasets_name[i],
            ax=axss[i],
            cmap="viridis",
            bot=0,
            top=4,
        )
    else:
        cf = dataset.arctic_plot(
            thickness[i] - thickness[0],
            title=datasets_name[i],
            ax=axss[i],
            cmap="coolwarm",
            bot=-2,
            top=2,
        )

# plot of means
fig2, ax2 = plt.subplots()

fig2.canvas.draw()

labels = [item.get_text() for item in ax2.get_xticklabels()]
labels = datasets_name
ax2.set_xticklabels(labels, rotation=45)
plt.xticks(xplot)

ax2.plot(xplot, mean_thick)

dataset10.multi_fig(fig, cf, save=1, label="Ice thickness [m]")
