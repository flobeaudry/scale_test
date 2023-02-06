import libs.visualization as vis
import numpy as np
import matplotlib.pyplot as plt

dataset = vis.Arctic(
    directory="output25",
    time="2002-01-31-00-00",
    expno="25",
    datatype="h",
    fig_shape="round",
    save=0,
    resolution=10,
    fig_name_supp="_TEST_2002",
)

dam = dataset.load(datatype="h")

plt.pcolor(dam, vmin=50, vmax=200, cmap="jet")
plt.colorbar()
plt.show()
