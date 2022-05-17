import libs.visualization as vis
import numpy as np
import matplotlib.pyplot as plt

dataset10 = vis.Arctic(
    directory="output10Dadvection_1997",
    time="1997-03-31-18-00",
    expno="03",
    datatype="dedt",
    fig_shape="round",
    save=1,
    resolution=10,
    fig_name_supp="_97_advection_rk2_imex2_bdf1",
)

# dataset10.arctic_plot(dataset10.load(datatype="dedt"))
# dataset10.arctic_plot(dataset10.load(datatype="shear"))
# dataset10.arctic_plot(dataset10.load(datatype="divergence"))
# dataset10.arctic_plot(dataset10.load(datatype="h"))
# dataset10.arctic_plot(dataset10.load(datatype="A"))

h = dataset10.load(datatype="h")

plt.pcolor(h)
plt.colorbar()
plt.show()
np.where(h > 398)
print(np.max(h), np.where(h > 398))

