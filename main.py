import numpy as np
from libs.constants import *
from libs.namelist import *

# ----------------------------------------------------------------------
# User input for the location of the files
# ----------------------------------------------------------------------

print("Welcome in the data processing routine!")

from libs.datasets import *

# ----------------------------------------------------------------------

datasets = np.array(
    [
        #dataset11,  # Background
        #dataset12,  # divergence in x-dir
        #dataset13,  # convergence in x-dir
        dataset29,  # SIM control
        #dataset15,  # Random noise + one line diagonal of deformation
        #dataset16,  # Vertical thin lines of convergence and divergence (1, -1, 1, -1)
        dataset11, # just noize
        #dataset17, # Vertical thicker lines of convergence and divergence (1, 1, -1, -1, 1, 1)
        #dataset20, # One vertical line of div (+1)
        #dataset21, # All div (+1 everywhere)
        #dataset22, # Spaced out vertical div lines (+1, 0, +1, 0)
        #dataset23, # More spaced out vertical div lines (+1, 0, 0, +1)
        #dataset24, # More spaced out vertical div lines (+1, 0, 0, +1)
        #dataset25, 
        #dataset26,
        #
        dataset60,
        #
        #dataset29,  # Control
        #dataset66,  # VP(0.7)
        #dataset10Dadv,  # VPd(2,1,30,27.5)
        #dataset23,  # VPd(2,3,30,27.5)
        #dataset25,  # VPd(2,5,30,27.5)
        #dataset33,  # VPd(2,3,2,27.5)
        #dataset35,  # VPd(2,5,2,27.5)
        #dataset65,  # VPd(0.7,5,30,27.5)
        ## dataset44,  # VPd(2,5,30,0.5)
        ## dataset45,  # VPd(2,5,30,2)
        #dataset67,  # VPd(2,5,30,35)
        #dataset68,  # VPd(2,5,30,35)
    ]
)

datasets_name = np.array(
    [
        "RGPS",
        #"U-div",
        #"U-conv",
        #"Background",
        "SIM",
        #"div diagonal",
        #"div-conv lines",
        "random noise",
        #"0 0 +1 0",
        #"+1 +1 +1 +1",
        #"+1 0 +1 0",
        #"+1 0 0 +1",
        #"+1 0 +2 0",
        #" +1 0000000 +1",
        #"1 000000000000000000000",
        #
        'test',
        #
        #"div-conv thick lines",
        #"VP(0.7)",
        #"VPd(2,1,30,27.5)",
        #"VPd(2,3,30,27.5)",
        #"VPd(2,5,30,27.5)",
        #"VPd(2,3,2,27.5)",
        #"VPd(2,5,2,27.5)",
        #"VPd(0.7,5,30,27.5)",
        ## "VPd(2,5,30,0.5)",
        ## "VPd(2,5,30,2)",
        #"VPd(2,5,30,35)",
        #"VPd(2,5,30,55)",
    ]
)

datasets_color = np.array(
    [
        "black",  # RGPS,
        #"xkcd:teal",  # Background
        #"xkcd:magenta", # x-dir convergence
        "tab:gray",  # SIM
        #"blue",  # x-dir divergence
        #"green",  # x-dir divergence
        "tab:red",
        'tab:blue', # 20
        #"tab:orange", #21
        #"tab:green", #22
        #"tab:purple", # 23
        #"tab:cyan", # 24
        #"tab:olive", #25
        #"tab:pink",
        #"xkcd:kelly green",
        #
        #"xkcd:dark mauve",  # Control
        #"xkcd:sandy",  # VP(0.7)
        #"xkcd:blue green",  # VPd(2,1,30,27.5)
        #"xkcd:kelly green",  # VPd(2,3,30,27.5)
        #"xkcd:light teal",  # VPd(2,5,30,27.5)
        #"xkcd:azure",  # VPd(2,3,2,27.5)
        #"xkcd:pastel blue",  # VPd(2,5,2,27.5)
        #"xkcd:goldenrod",  # VPd(0.7,5,30,27.5)
        #"xkcd:powder pink",  # VPd(2,5,30,35)
        #"xkcd:deep rose",  # VPd(2,5,30,35)
        ## "xkcd:light mauve",   # VPd(2,5,30,0.5)
    ]
)

# ----------------------------------------------------------------------
# dedt plots
# ----------------------------------------------------------------------

if arctic_plots == 1:
    fig, axss = dataset29.multi_fig_precond(x, y, total, remove)

    dudx = np.load("RGPS_derivatives/DUDX.npy")
    dudy = np.load("RGPS_derivatives/DUDY.npy")
    dvdx = np.load("RGPS_derivatives/DVDX.npy")
    dvdy = np.load("RGPS_derivatives/DVDY.npy")
    du80_RGPS = np.stack((dudx, dudy, dvdx, dvdy), axis=-1)
    deps_RGPS_plot = dataset_RGPS._deformation(du80_RGPS, 0)

    cf = dataset_RGPS.arctic_plot_RGPS(
        deps_RGPS_plot[..., 0], "u", "_02_", ax=np.delete(axss, remove)[0]
    )

    for k, dataset in enumerate(datasets):
        dedt_plot = dataset.multi_load(
            datatype="u", time_end="2002-01-31-18-00", dt=dt
        )
        dedt_plot_ta = dataset._time_average(dedt_plot, dt)
        dataset.arctic_plot(
            dedt_plot_ta[..., 0],
            title=datasets_name[k + 1],
            ax=np.delete(axss, remove)[k + 1],
        )

    dataset29.multi_fig(fig, cf, save=1)

# ----------------------------------------------------------------------
# load everthing, compute du, mask du with RGPS80
# ----------------------------------------------------------------------
if deformation_plots:
    if not load:
        
        # mask80
        # mask80 = dataset_RGPS.mask80("RGPS_data", ti=-1, tf=88)
        mask80 = np.load("RGPS_mask/mask80JFM.npy")

        # RGPS data
        # deps, div, shear = dataset_RGPS.nc_load("RGPS_data/w0102n_3dys.nc", tf=29)
        # load derivatives and mask it
        dudx = dataset_RGPS.mask80_times_RGPS(
            np.load("RGPS_derivatives/DUDX.npy"), mask80
        )
        dudy = dataset_RGPS.mask80_times_RGPS(
            np.load("RGPS_derivatives/DUDY.npy"), mask80
        )
        dvdx = dataset_RGPS.mask80_times_RGPS(
            np.load("RGPS_derivatives/DVDX.npy"), mask80
        )
        dvdy = dataset_RGPS.mask80_times_RGPS(
            np.load("RGPS_derivatives/DVDY.npy"), mask80
        )

        # stack them
        du80_RGPS = np.stack((dudx, dudy, dvdx, dvdy), axis=-1)
        
        
        #dudx = np.load("../artificial_fields/DUDX.npy")
        #dudy = np.load("../artificial_fields/DUDY.npy")
        #dvdx = np.load("../artificial_fields/DVDX.npy")
        #dvdy = np.load("../artificial_fields/DVDY.npy")
        #print(np.shape(dudx), np.shape(dudy), np.shape(dvdx), np.shape(dvdy))
        #du80_RGPS = np.stack((dudx, dudy, dvdx, dvdy), axis=-1)

        # --------------------------------------------------------------
        # Scaling spatial
        # --------------------------------------------------------------
        (
            deps_RGPS_du,
            shear_RGPS_du,
            div_RGPS_du,
            scale_RGPS_du,
        ) = dataset_RGPS.spatial_mean_du(du80_RGPS, L_RGPS)

        # --------------------------------------------------------------
        # Scaling temporal
        # --------------------------------------------------------------
        (
            deps_RGPS_du_T,
            shear_RGPS_du_T,
            div_RGPS_du_T,
            scale_RGPS_du_T,
        ) = dataset_RGPS.temporal_mean_du(du80_RGPS, T10)

        # --------------------------------------------------------------
        # Means (only for spatial)
        # --------------------------------------------------------------
        (
            mean_deps_RGPS_du,
            mean_scale_RGPS_du,
            __,
        ) = dataset_RGPS.scale_plot_vect(
            deps_RGPS_du,
            scale_RGPS_du,
            L_RGPS,
            save=0,
            fig_name_supp="_dedt_02_RGPS_du",
        )
        # --------------------------------------------------------------
        # multifractality
        # --------------------------------------------------------------
        # spatial
        parametersRGPS, coeffRGPS = dataset_RGPS.multifractal_spatial(
            3,
            deps_RGPS_du,
            scale_RGPS_du,
            RGPS=1,
        )

        # temporal
        parametersRGPS_T, coeffRGPS_T = dataset_RGPS.multifractal_temporal(
            3, du80_RGPS
        )

        # --------------------------------------------------------------
        # initialize stack of deformations and means with RGPS
        # --------------------------------------------------------------
        du80_stack = [du80_RGPS]

        mean_deps_stack = np.empty(
            (mean_deps_RGPS_du.shape[0], len(datasets) + 1)
        )
        mean_scale_stack = np.empty(
            (mean_scale_RGPS_du.shape[0], len(datasets) + 1)
        )

        mean_deps_stack_T = np.empty(
            (deps_RGPS_du_T.shape[0], len(datasets) + 1)
        )
        mean_scale_stack_T = np.empty(
            (scale_RGPS_du_T.shape[0], len(datasets) + 1)
        )

        param_stack = np.empty((parametersRGPS.shape[0], len(datasets) + 1))
        coeff_stack = np.empty((coeffRGPS.shape[0], len(datasets) + 1))

        param_stack_T = np.empty((parametersRGPS_T.shape[0], len(datasets) + 1))
        coeff_stack_T = np.empty((coeffRGPS_T.shape[0], len(datasets) + 1))

        mean_deps_stack[:, 0] = mean_deps_RGPS_du
        mean_scale_stack[:, 0] = mean_scale_RGPS_du

        mean_deps_stack_T[:, 0] = deps_RGPS_du_T
        mean_scale_stack_T[:, 0] = scale_RGPS_du_T

        param_stack[:, 0] = parametersRGPS
        coeff_stack[:, 0] = coeffRGPS

        param_stack_T[:, 0] = parametersRGPS_T
        coeff_stack_T[:, 0] = coeffRGPS_T


        # --------------------------------------------------------------
        # loop on all sim datasets
        # --------------------------------------------------------------
        
        for j, dataset in enumerate(datasets):
            # calcul time averaged
            u_v = dataset.multi_load(dt, time_end, datatype=datatype)
            u_v = np.where(u_v == 0, np.nan, u_v)
            u_v_ta = dataset._time_average(u_v, dt)

            # calcul du
            du = dataset._derivative(u_v_ta[:, :, 0, :], u_v_ta[:, :, 1, :])

            # mask the data
            
            #dudx = np.load("../artificial_fields/DUDX.npy")
            #dudy = np.load("../artificial_fields/DUDY.npy")
            #dvdx = np.load("../artificial_fields/DVDX.npy")
            #dvdy = np.load("../artificial_fields/DVDY.npy")
            #du80 = np.stack((dudx, dudy, dvdx, dvdy), axis=-1)
            
            du80 = du
            du80[..., 0] = dataset.mask80_times(du[..., 0], mask80)[0]
            du80[..., 1] = dataset.mask80_times(du[..., 1], mask80)[0]
            du80[..., 2] = dataset.mask80_times(du[..., 2], mask80)[0]
            du80[..., 3] = dataset.mask80_times(du[..., 3], mask80)[0]
            

            # ----------------------------------------------------------
            #       Scaling
            # ----------------------------------------------------------

            # spatial
            deps, shear, div, scale = dataset.spatial_mean_du(du80, L10)

            # temporal
            deps_T, shear_T, div_T, scale_T = dataset.temporal_mean_du(
                du80, T10
            )

            # means (only spatial)
            mean_deps, mean_scale, __ = dataset.scale_plot_vect(
                deps, scale, L10, save=0, fig_name_supp="_dedt_02"
            )

            # ----------------------------------------------------------
            #       Multifractality
            # ----------------------------------------------------------

            # spatial
            parameters, coeff = dataset.multifractal_spatial(
                3,
                deps,
                scale,
            )

            # temporal
            parameters_T, coeff_T = dataset.multifractal_temporal(3, du80)

            # ----------------------------------------------------------
            # append to lists
            # ----------------------------------------------------------
            du80_stack.append(du80)

            mean_deps_stack[:, j + 1] = mean_deps
            mean_scale_stack[:, j + 1] = mean_scale

            mean_deps_stack_T[:, j + 1] = deps_T
            mean_scale_stack_T[:, j + 1] = scale_T

            param_stack[:, j + 1] = parameters
            coeff_stack[:, j + 1] = coeff

            param_stack_T[:, j + 1] = parameters_T
            coeff_stack_T[:, j + 1] = coeff_T
        
        '''
        # Check this
        np.savez(
            namefile,
            du80_stack,
            mean_deps_stack,
            mean_scale_stack,
            mean_deps_stack_T,
            mean_scale_stack_T,
            param_stack,
            coeff_stack,
            param_stack_T,
            coeff_stack_T,
        )
        '''

    elif load:
        datafile = np.load(namefile, allow_pickle=True)

        du80_stack = datafile["arr_0"]

        mean_deps_stack = datafile["arr_1"]
        mean_scale_stack = datafile["arr_2"]

        mean_deps_stack_T = datafile["arr_3"]
        mean_scale_stack_T = datafile["arr_4"]

        param_stack = datafile["arr_5"]
        coeff_stack = datafile["arr_6"]

        param_stack_T = datafile["arr_7"]
        coeff_stack_T = datafile["arr_8"]

    # ------------------------------------------------------------------
    # plot PDF and CDF
    # ------------------------------------------------------------------

    dataset29.pdf_du(
        du80_stack,
        save=1,
        fig_name_supp="_02",
        names_plot=datasets_name,
        colors_plot=datasets_color,
    )
    dataset29.cdf_du(
        du80_stack,
        save=1,
        fig_name_supp="_02",
        names_plot=datasets_name,
        colors_plot=datasets_color,
    )

    # ----------------------------------------------------------------------
    # multiplot scaling
    # ----------------------------------------------------------------------
    dataset29.multi_plot_spatial(
        mean_deps_stack,
        mean_scale_stack,
        fig_name_supp="_dedt_02",
        names_plot=datasets_name,
        colors_plot=datasets_color,
    )

    dataset29.multi_plot_temporal(
        mean_deps_stack_T,
        mean_scale_stack_T,
        fig_name_supp="_dedt_02",
        names_plot=datasets_name,
        colors_plot=datasets_color,
    )

    # ----------------------------------------------------------------------
    # multifractality plots
    # ----------------------------------------------------------------------
    dataset29.multifractal_plot(
        param_stack,
        coeff_stack,
        3,
        1,
        "_param_02",
        names_plot=datasets_name,
        colors_plot=datasets_color,
    )

    dataset29.multifractal_plot(
        param_stack_T,
        coeff_stack_T,
        3,
        1,
        "T_param_02",
        temp=1,
        names_plot=datasets_name,
        colors_plot=datasets_color,
    )
