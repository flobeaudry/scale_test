import libs.visualization as vis
import numpy as np
import matplotlib.pyplot as plt

dataset = vis.Arctic(
    directory="output10Dadv_2002",
    time="2002-01-15-00-00",
    expno="06",
    datatype="dedt",
    fig_shape="round",
    save=1,
    resolution=10,
    fig_name_supp="_TEST_2002",
)

dam = dataset.load(datatype="dam")

plt.pcolor(dam)
plt.colorbar()
plt.show()

