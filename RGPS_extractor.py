import numpy as np
import netCDF4 as nc
import libs.visualization as vis
import matplotlib.pyplot as plt

fn97 = "RGPS_from_sirex/RGPS_composite_deformation_1997.nc"
fn08 = "RGPS_from_sirex/RGPS_composite_deformation_2008.nc"

ds97 = nc.Dataset(fn97)
ds08 = nc.Dataset(fn08)

dudx97 = ds97["dudx"][:]
dudy97 = ds97["dudy"][:]
dvdx97 = ds97["dvdx"][:]
dvdy97 = ds97["dvdy"][:]

dataset_RGPS = vis.Arctic(
    fig_shape="round",
    save=1,
    resolution=12.5,
    fig_name_supp="_02_RGPS",
    fig_type="png",
)

mask80 = np.load("RGPS_mask/mask80JFM.npy")
# dudx97 = dataset_RGPS.mask80_times_RGPS(dudx97, mask80)
# dudy97 = dataset_RGPS.mask80_times_RGPS(dudy97, mask80)
# dvdx97 = dataset_RGPS.mask80_times_RGPS(dvdx97, mask80)
# dvdy97 = dataset_RGPS.mask80_times_RGPS(dvdy97, mask80)

du80_RGPS = np.stack((dudx97, dudy97, dvdx97, dvdy97), axis=-1)

dedt = dataset_RGPS._deformation(du80_RGPS, 0)

plt.pcolormesh(dedt[5])
plt.show()
