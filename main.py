import libs.visualization as vis
import numpy as np

# User input for the location of the files

dataset10 = vis.Arctic(
    directory="output10",
    time="1977-01-01-00-00",
    expno="07",
    datatype="dedt",
    fig_shape="round",
    save=1,
    resolution=10,
)

dataset20 = vis.Arctic(
    directory="output20",
    time="1997-01-01-00-00",
    expno="01",
    datatype="dedt",
    fig_shape="round",
    save=1,
    resolution=20,
)

dataset40 = vis.Arctic(
    directory="output40",
    time="1997-01-02-00-00",
    expno="11",
    datatype="dedt",
    fig_shape="round",
    save=1,
    resolution=40,
)

# dataset10.arctic_plot(dataset20.load())
# dataset.multi_load("01-00-00", "1997-03-31-00-00")

L10 = [20, 40, 80, 160, 320, 640]
L20 = [40, 80, 160, 320, 640]
L40 = [80, 160, 320, 640]
dt = "00-06-00"
time_end = "1997-03-31-18-00"

# compute all mean deformations in boxes
# data_box10, data_box10_visc = dataset10.spatial_mean_box(dataset10.load(), L10, dt)
data_box20, data_box20_visc = dataset20.spatial_mean_box(
    dataset20.multi_load(dt, time_end), L20, dt, time_end
)
# data_box40, data_box40_visc = dataset40.spatial_mean_box(
#     dataset40.multi_load(dt, time_end), L40, dt, time_end
# )

# save data in file
# np.save("data10.npy", data_box10)
# np.save("data10_visc.npy", data_box10_visc)
np.save("data20.npy", data_box20)
np.save("data20_visc.npy", data_box20_visc)
# np.save("data40.npy", data_box40)
# np.save("data40_visc.npy", data_box40_visc)

# load data if previously saved
# data_box10 = np.load("data10.npy")
# data_box10_visc = np.load("data10_visc.npy")
# data_box20 = np.load("data20.npy")
# data_box20_visc = np.load("data20_visc.npy")
# data_box40 = np.load("data40.npy")
# data_box40_visc = np.load("data40_visc.npy")

# plots at 10 km
# dataset10.pdf_plot(data_box10)
# dataset10.cdf_plot(data_box10)
# dataset10.scale_plot(data_box10, L10, data_box10_visc)

# plots at 20 km
# dataset20.pdf_plot(data_box20)
# dataset20.cdf_plot(data_box20)
# dataset20.scale_plot(data_box20, L20, data_box20_visc)

# plots at 40 km
# dataset40.pdf_plot(data_box40)
# dataset40.cdf_plot(data_box40)
# dataset40.scale_plot(data_box40, L40, data_box40_visc)
