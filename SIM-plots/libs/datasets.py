import libs.visualization as vis
from libs.namelist import *

# Test dataset artificial
dataset29 = vis.Arctic(
    directory="output29",
    time=start_time,
    expno="29",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="29",
    fig_type=fig_type,
    trans=trans,
)

dataset10 = vis.Arctic(
    directory="output10",
    time=start_time,
    expno="10",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="10",
    fig_type=fig_type,
    trans=trans,
)

# Only random backgroud between 0 and 1
dataset11 = vis.Arctic(
    directory="output11",
    time=start_time,
    expno="11",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="11",
    fig_type=fig_type,
    trans=trans,
)

# Random backgroud between 0 and 1 + generally diverging field in x-dir
dataset12 = vis.Arctic(
    directory="output12",
    time=start_time,
    expno="12",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="12",
    fig_type=fig_type,
    trans=trans,
)

# Random backgroud between 0 and 1 + generally converging field in x-dir
dataset13 = vis.Arctic(
    directory="output13",
    time=start_time,
    expno="13",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="13",
    fig_type=fig_type,
    trans=trans,
)

# Div field with big diverging diagonal
dataset15 = vis.Arctic(
    directory="output15",
    time=start_time,
    expno="15",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="15",
    fig_type=fig_type,
    trans=trans,
)

# Div field with vertical lines of div (+1, -1, +1, -1, etc)
dataset16 = vis.Arctic(
    directory="output16",
    time=start_time,
    expno="16",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="16",
    fig_type=fig_type,
    trans=trans,
)

# Div field with vertical lines of div (+1, +1, -1, -1, +1, etc)
dataset17 = vis.Arctic(
    directory="output17",
    time=start_time,
    expno="17",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="17",
    fig_type=fig_type,
    trans=trans,
)

# -------------------------------------------------------------------------
# All 0 with one vertical line of div (+1)
dataset20 = vis.Arctic(
    directory="output20",
    time=start_time,
    expno="20",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="20",
    fig_type=fig_type,
    trans=trans,
)
# All +1 div
dataset21 = vis.Arctic(
    directory="output21",
    time=start_time,
    expno="21",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="21",
    fig_type=fig_type,
    trans=trans,
)
# Lines of div +1 0 +1 0 etc
dataset22 = vis.Arctic(
    directory="output22",
    time=start_time,
    expno="22",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="22",
    fig_type=fig_type,
    trans=trans,
)
# Lines of div +1 0 0 +1 0 0 etc
dataset23 = vis.Arctic(
    directory="output23",
    time=start_time,
    expno="23",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="23",
    fig_type=fig_type,
    trans=trans,
)
# Lines of div +1 0 +2 0 +1 0 +2 etc
dataset24 = vis.Arctic(
    directory="output24",
    time=start_time,
    expno="24",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="24",
    fig_type=fig_type,
    trans=trans,
)
# Lines of div +1 0 0 0 0 0 0 0 +1
dataset25 = vis.Arctic(
    directory="output25",
    time=start_time,
    expno="25",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="25",
    fig_type=fig_type,
    trans=trans,
)

# Lines of div +1 0 0 0 0 0 0 0 +1
dataset26 = vis.Arctic(
    directory="output26",
    time=start_time,
    expno="26",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="26",
    fig_type=fig_type,
    trans=trans,
)

# test
dataset60 = vis.Arctic(
    directory="output60",
    time=start_time,
    expno="60",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="60",
    fig_type=fig_type,
    trans=trans,
)
# test
dataset61 = vis.Arctic(
    directory="output61",
    time=start_time,
    expno="61",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="61",
    fig_type=fig_type,
    trans=trans,
)


dataset10_bad = vis.Arctic(
    directory="output10_2002",
    time=start_time,
    expno="05",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="_02",
    fig_type=fig_type,
    trans=trans,
)

dataset10Dadv2 = vis.Arctic(
    directory="output10Dadv_2002_2",
    time=start_time,
    expno="07",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="Dadv_02_2",
    fig_type=fig_type,
    trans=trans,
)

dataset10Dadv = vis.Arctic(
    directory="output10Dadv_2002",
    time=start_time,
    expno="06",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="Dadv_02",
    fig_type=fig_type,
    trans=trans,
)

dataset83 = vis.Arctic(
    directory="output83",
    time=start_time,
    expno="83",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="83",
    fig_type=fig_type,
    trans=trans,
)

dataset85 = vis.Arctic(
    directory="output85",
    time=start_time,
    expno="85",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="85",
    fig_type=fig_type,
    trans=trans,
)

dataset93 = vis.Arctic(
    directory="output93",
    time=start_time,
    expno="93",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="93",
    fig_type=fig_type,
    trans=trans,
)

dataset95 = vis.Arctic(
    directory="output95",
    time=start_time,
    expno="95",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="95",
    fig_type=fig_type,
    trans=trans,
)

dataset23 = vis.Arctic(
    directory="output23",
    time=start_time,
    expno="23",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="23",
    fig_type=fig_type,
    trans=trans,
)

dataset25 = vis.Arctic(
    directory="output25",
    time=start_time,
    expno="25",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="25",
    fig_type=fig_type,
    trans=trans,
)

dataset29 = vis.Arctic(
    directory="output29",
    time=start_time,
    expno="29",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="29",
    fig_type=fig_type,
    trans=trans,
)

dataset30 = vis.Arctic(
    directory="output30",
    time=start_time,
    expno="30",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="30",
    fig_type=fig_type,
    trans=trans,
)

dataset31 = vis.Arctic(
    directory="output31",
    time=start_time,
    expno="31",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="31",
    fig_type=fig_type,
    trans=trans,
)

dataset33 = vis.Arctic(
    directory="output33",
    time=start_time,
    expno="33",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="33",
    fig_type=fig_type,
    trans=trans,
)

dataset34 = vis.Arctic(
    directory="output34",
    time=start_time,
    expno="34",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="34",
    fig_type=fig_type,
    trans=trans,
)

dataset35 = vis.Arctic(
    directory="output35",
    time=start_time,
    expno="35",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="35",
    fig_type=fig_type,
    trans=trans,
)

dataset53 = vis.Arctic(
    directory="output53",
    time=start_time,
    expno="53",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="53",
    fig_type=fig_type,
    trans=trans,
)

dataset52 = vis.Arctic(
    directory="output52",
    time=start_time,
    expno="52",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="52",
    fig_type=fig_type,
    trans=trans,
)

dataset50 = vis.Arctic(
    directory="output50",
    time=start_time,
    expno="50",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="50",
    fig_type=fig_type,
    trans=trans,
)

dataset65 = vis.Arctic(
    directory="output65",
    time=start_time,
    expno="65",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="65",
    fig_type=fig_type,
    trans=trans,
)

dataset66 = vis.Arctic(
    directory="output66",
    time=start_time,
    expno="66",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="66",
    fig_type=fig_type,
    trans=trans,
)

dataset44 = vis.Arctic(
    directory="output44",
    time=start_time,
    expno="44",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="44",
    fig_type=fig_type,
    trans=trans,
)

dataset45 = vis.Arctic(
    directory="output45",
    time=start_time,
    expno="45",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="45",
    fig_type=fig_type,
    trans=trans,
)

dataset67 = vis.Arctic(
    directory="output67",
    time=start_time,
    expno="67",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="67",
    fig_type=fig_type,
    trans=trans,
)

dataset68 = vis.Arctic(
    directory="output68",
    time=start_time,
    expno="68",
    datatype=datatype,
    save=save,
    resolution=10,
    fig_shape=fig_shape,
    fig_name_supp="68",
    fig_type=fig_type,
    trans=trans,
)

dataset_RGPS = vis.Arctic(
    fig_shape=fig_shape,
    save=save,
    resolution=12.5,
    fig_name_supp="_02_RGPS",
    fig_type=fig_type,
    trans=trans,
)
