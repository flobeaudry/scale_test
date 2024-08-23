# time definition
start_time = "2002-01-01-00-00"
#start_time2 = "2022-01-01-00-00"
dt = "00-06-00"
time_end = "2002-01-31-18-00"
#time_end2 = "2022-01-31-18-00"

# other parameters of datasets
datatype = "u"
#datatype = "dedt"
trans = False
fig_type = "png"
fig_shape = "square"
save = 0

# if you want to plot maps of the Arctic
arctic_plots = 1
x = 3
y = 4
total = 11
remove = [6]

# if you want to plot the deformation statistics plots
deformation_plots = 1
# if you want to use data that has already been processed to do plots
load = 0
# name of the file where you save your processed data
namefile = "../artificial_fields/try.npz"  # test with my own "artificial" file
#namefile = "massaged_data/try.npz"  # antoine's original file that I don't know
