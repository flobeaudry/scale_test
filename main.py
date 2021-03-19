import libs.visualization as vis
import matplotlib.pyplot as plt

# User input for the location of the files
dataset = vis.Arctic(
    directory="spin_up10",
    time="1986-01-01-00-00",
    expno="03",
    datatype="dedt",
    fig_shape="round",
    save=1,
)

print("\nWelcome in the data loader routine! To cancel selection, enter exit or quit.")

dataset.arctic_plot(dataset.load())
# dataset.multi_load("01-00-00", "1997-03-31-00-00")

# L = [80, 160, 320, 640]
# dt = "01-00-00"
# time_end = "1997-03-31-00-00"
# data_box = dataset.spatial_mean_box(dataset.multi_load(dt, time_end), L, dt)
# dataset.scale_plot(data_box, L)
# plt.show()