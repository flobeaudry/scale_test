import numpy as np
import matplotlib.pyplot as plt
import libs.visualization as vis
from scipy.special import factorial

dataset = vis.Arctic(
    directory="output10D_1997",
    time="1997-01-01-00-00",
    expno="01",
    datatype="damage",
    fig_shape="round",
    save=0,
    resolution=10,
    fig_name_supp="_97",
)

dt = "00-06-00"
time_end = "1997-01-03-18-00"

dam = dataset.multi_load(dt, time_end)

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
