import libs.visualization as vis
import numpy as np
import matplotlib.pyplot as plt

dataset = vis.Arctic(
    directory="output10_2002",
    time="2002-01-01-00-00",
    expno="05",
    datatype="h",
    fig_shape="round",
    save=1,
    resolution=10,
    fig_name_supp="_TEST_2002",
)

dam = dataset.load(datatype="h")

plt.pcolor(dam, vmax=5, cmap="jet")
plt.colorbar()
plt.show()

