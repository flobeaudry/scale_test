import netCDF4 as nc
import numpy as np
import scipy.interpolate as sci
import matplotlib.pyplot as plt
import matplotlib.colors as colors

fn40 = "40/geowinds.nc"
fn80 = "80/geowinds.nc"
ds40 = nc.Dataset(fn40)
ds80 = nc.Dataset(fn80)

x80 = np.arange(54 + 1) * 80 - 2500
y80 = np.arange(64 + 1) * 80 - 2250

x40 = np.arange(109 + 1) * 40 - 2500
y40 = np.arange(129 + 1) * 40 - 2250

x80 = (x80[:-1] + x80[1:]) / 2
y80 = (y80[:-1] + y80[1:]) / 2

x40 = (x40[:-1] + x40[1:]) / 2
y40 = (y40[:-1] + y40[1:]) / 2

high = ds40["uwnd"][-1]
low = ds80["uwnd"][-1]

high_mean = []
for j in range(high.shape[1] // 2):
    for i in range(high.shape[0] // 2):
        high_mean.append(np.mean(high[2 * i : 2 * i + 3, 2 * j : 2 * j + 3]))

x = np.reshape(high_mean, low.shape)

print(x.mean(), low.mean(), high.mean())

# f = sci.RectBivariateSpline(x80, y80, ds80["uwnd"][-1, :])
# interp = f(x40, y40)

# print(np.mean(interp), np.mean(ds40["uwnd"][-1]), np.mean(ds80["uwnd"][-1]))
plt.imshow(
    (x - low),
    cmap="bwr",
    # norm=colors.Normalize(vmin=-25, vmax=25),
)
plt.colorbar()
plt.savefig("diff.png")