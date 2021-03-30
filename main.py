import libs.visualization as vis
import numpy as np

# User input for the location of the files
# dataset40 = vis.Arctic(
#     directory="output40",
#     time="1997-01-02-00-00",
#     expno="11",
#     datatype="dedt",
#     fig_shape="round",
#     save=1,
#     resolution=40,
# )

dataset10 = vis.Arctic(
    directory="spin_up10",
    time="1986-01-01-00-00",
    expno="03",
    datatype="dedt",
    fig_shape="round",
    save=1,
    resolution=10,
)

# dataset.arctic_plot(dataset.load())
# dataset.multi_load("01-00-00", "1997-03-31-00-00")

L10 = [20, 40, 80, 160, 320, 640]
L40 = [80, 160, 320, 640]
dt = "01-00-00"
time_end = "1997-03-31-00-00"

# compute all mean deformations in boxes
# data_box40 = dataset40.spatial_mean_box(dataset10.multi_load(dt, time_end), L, dt)
# data_box10 = dataset10.spatial_mean_box(dataset10.load(), L, dt)

# save data in file
# np.save("data10.npy", data_box10)

# load data if previously saved
data_box10 = np.load("data10.npy")
# data_box40 = np.load("data.npy")

# plots at 10 km
dataset10.pdf_plot(data_box10)
dataset10.cdf_plot(data_box10)
# dataset10.scale_plot(data_box10, L10)

# plots at 40 km
# dataset40.pdf_plot(data_box40)
# dataset40.cdf_plot(data_box40)
# dataset40.scale_plot(data_box40, L40)
