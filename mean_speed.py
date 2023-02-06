import numpy as np
from libs.constants import *
from libs.namelist import *
from libs.datasets import *
import matplotlib.pyplot as plt


datasets = np.array(
    [
        dataset10,
        dataset66,
        dataset10Dadv,
        dataset23,
        dataset25,
        dataset65,
        dataset44,
        dataset45,
        dataset29,
        dataset33,
        dataset35,
    ]
)

datasets_name = np.array(
    [
        "RGPS",
        "VP(2)",
        "VP(0.7)",
        "VPd(2,1,30)",
        "VPd(2,3,30)",
        "VPd(2,5,30)",
        "VPd(0.7,5,30)",
        "VPd(2,5,30,0.5)",
        "VPd(2,5,30,2)",
        "VPd(2,50,30)",
        "VPd(2,3,2)",
        "VPd(2,5,2)",
    ]
)

datasets_color = np.array(
    [
        "black",
        "xkcd:dark mauve",
        "xkcd:sandy",
        "xkcd:blue green",
        "xkcd:kelly green",
        "xkcd:light teal",
        "xkcd:goldenrod",
        "xkcd:powder pink",
        "xkcd:deep rose",
        "xkcd:light mauve",
        "xkcd:azure",
        "xkcd:pastel blue",
    ]
)

t = np.arange(0, 31, 0.25)
x = np.arange(0, datasets.shape[0])
speed = np.zeros((t.shape[0], datasets.shape[0]))

mask80 = np.load("RGPS_mask/mask80JFM.npy")

for i, dataset in enumerate(datasets):
    u_v_raw = dataset.multi_load(dt, time_end, datatype="u")
    u_raw = dataset.mask80_times(u_v_raw[:, :, 0, :], mask80)[0]
    v_raw = dataset.mask80_times(u_v_raw[:, :, 1, :], mask80)[0]
    u_v_mask = np.stack((u_raw, v_raw), axis=-2)
    u_v = np.where(u_v_mask == 0, np.NaN, u_v_mask)
    for j, __ in enumerate(t):
        speed[j, i] = np.nanmean(
            np.sqrt(u_v[:, :, 0, j] ** 2 + u_v[:, :, 1, j] ** 2), axis=(0, 1)
        )

fig, ax = plt.subplots()

fig.canvas.draw()

labels = [item.get_text() for item in ax.get_xticklabels()]
labels = datasets_name[1:]
ax.set_xticklabels(labels, rotation=90)
plt.xticks(x)

ax.plot(x, np.nanmean(speed, axis=0))
plt.show()
