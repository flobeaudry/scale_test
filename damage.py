import numpy as np
import matplotlib.pyplot as plt
import libs.visualization as vis
from scipy.special import factorial

dataset = vis.Arctic(
    directory="output99",
    time="2002-01-01-06-00",
    expno="99",
    datatype="damage",
    fig_shape="round",
    save=0,
    resolution=10,
    fig_name_supp="_02",
)

dt = "00-06-00"
time_end = "2002-01-01-06-00"

dam = dataset.multi_load(dt, time_end)
dam = np.where(dam > 0, dam, np.NaN)

plt.pcolormesh(dam[..., -1])
plt.show()

dam_flat = dam.flatten()
n = plt.hist(dam_flat, bins="auto")
l = n[0][-1]
plt.savefig("images/damage.png")
plt.show()
# plt.plot((l ** n[0]) / factorial(n[0]) * np.exp(-l))
# print((l ** n[0]) / factorial(n[0]) * np.exp(-l))
# plt.show()
