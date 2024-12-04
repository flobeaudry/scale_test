# scale_test
Creation of artificial velocity gradient fields and analysis with SIM-plots.


# SIM-plots
Plotting tools for the McGill Sea Ice Model.

Simply run main.py with the modules you want to use. You will have to run file_converter.py if you want to convert binary files from Ron Kwok into numpy arrays.

Modules that can be useful:<br/>
<br/>
-load: loads data file into variable<br/>
-multi_load: loads multiple data files into variable<br/>
-spatial_mean_box: computes the spatial mean in all provided box sizes<br/>
<br/>
-arctic_plot: plots dataset over the arctic<br/>
-scale_plot: plots the scaling graph from the data from boxes<br/>
-pdf_plot: plots PDF of the data from boxes<br/>
-cdf_plot: plots CDF of the data from boxes<br/>

Dependencies:<br/>
-Needs cartopy;<br/>
-Needs cmocean;<br/>
-Works in anaconda3.<br/>
