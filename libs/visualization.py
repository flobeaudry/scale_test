# ----------------------------------------------------------------------
#   Data visualization module
# ----------------------------------------------------------------------
#   Its purpose is to plot the data it receives.
#
#   TODO:
#   DONE    define function that plots formated data
#   DONE    define function that formats data
#   DONE    add step variable for quiver
#   DONE    try to get better plots with fancy ellipsoid
#   -define function that makes a movie of the data
#   -better arrow representation for u data
#   DONE    pdf plot with fit on it
#   DONE    cdf plot
#   DONE    color scatter dependence on viscosity for scale plot
#   DONE    distribution subplot for scale plot
#
#   TODO (eventually):
#
# ----------------------------------------------------------------------


import string
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as colors
import libs.stats as sts
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.ticker
from libs.constants import *
from descartes import PolygonPatch
import alphashape
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Arctic(sts.Scale):
    """
    This class is a child class from sts.Scale class. Its main goal is to plot the data that has been extrated by the parent class, and by extension, plot any data similar to that. It plots the data on the Arctic circle above 65 degrees North with an Earth of radius 6370 km. The ploted domain is rotated by 32 degrees from Greenwich.

    Only works for 10-20-40-80 km ouputs given by McGill Sea Ice Model.

    arctic_plot: function that plots data over the arctic.
    scale_plot: function that plots the deformation for different scaling, and the linear regression model for the exponent H.
    """

    def __init__(
        self,
        directory: str = None,
        time: str = None,
        expno: str = None,
        datatype: str = None,
        tolerance: float = 0.1,
        resolution: int = None,
        nx: int = None,
        ny: int = None,
        fig_shape: str = "square",
        step: int = None,
        save: bool = 0,
        fig_type: str = "png",
        fig_name_supp: str = None,
        trans: bool = False,
    ):
        """
        Class attributes for Arctic.

        Args:
            directory (str, optional): directory from which to take data. Defaults to None.

            time (str, optional): starting time, format supported: yyyy-mm-dd-hh-mm. Defaults to None.

            expno (str, optional): experience number is of format nn. Defaults to None.
            datatype (str, optional): data types currently supported are: ice concentration (A), ice thickness (h), ice velocity vector (u), ice temp (Ti) (needs tweeks for pcolor), and ice deformation (dedt). Defaults to None.

            tolerance (float, optional): value at which dedt will be cut to get rid of high boundary values. Defaults to 0.1.

            resolution (int, optional): spatial resolution of the domain of interest.

            nx, ny (int, optional): number of cells in each direction.

            fig_shape (str, optional): round or square Arctic representation. Defaults to "square".

            step (int, optional): number of arrows to skip in quiver when plotting velocities. Defaults to None.

            save (bool, optional): save or not the figure. Defaults to 0.

            fig_type (str, optional): figure type. Defaults to "png".
        """

        self.fig_type = fig_type
        self.save = save
        self.step = step
        self.fig_shape = fig_shape
        self.fig_name_supp = fig_name_supp
        self.trans = trans
        super(Arctic, self).__init__(
            directory, time, expno, datatype, tolerance, resolution, nx, ny
        )

    def arctic_plot(
        self,
        formated_data: np.ndarray,
        title: string = None,
        ax=None,
        cmap=None,
        top=None,
        bot=None,
    ):
        """
        Plots passed data on the Arctic circle, and can save the image to images/ directory.

        Args:
            formated_data (np.ndarray): data to plot (ny, nx)
            title (string): title of subplot

        Raises:
            SystemExit: If invalid data format
            SystemExit: If invalid data type
        """
        # cartesian coordonates on the plane for the corners of the cells for pcolor
        x0 = np.arange(self.nx + 1) * self.resolution - 2500
        y0 = np.arange(self.ny + 1) * self.resolution - 2250

        lon, lat = self._coordinates(x0, y0)

        # for the quiver variables that are not in the corner of the cell grid like pcolormesh, but they are rather in the center of the grid so we have to interpolate the grid points
        if self.datatype == "u":
            x1 = (x0[1:] + x0[:-1]) / 2
            y1 = (y0[1:] + y0[:-1]) / 2

            lon1, lat1 = self._coordinates(x1, y1)

        axi = ax
        if axi is None:
            # figure initialization
            fig = plt.figure(dpi=300)
            if self.trans:
                fig.patch.set_facecolor("None")
            ax = plt.subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
            fig.subplots_adjust(
                bottom=0.05, top=0.95, left=0.04, right=0.95, wspace=0.02
            )

        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        if self.fig_shape == "round":
            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)

        # Limit the map to 65 degrees latitude and above.
        ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())

        # title
        ax.text(
            0.5,
            1.05,
            title,
            fontsize=10,
            transform=ax.transAxes,
            va="center",
            ha="center",
        )

        # all the plots
        # for concentration
        if self.datatype == "A":
            ax.add_feature(cfeature.OCEAN, color="black", zorder=0)
            cf = ax.pcolormesh(
                lon,
                lat,
                formated_data,
                norm=colors.Normalize(vmin=0.95, vmax=1),
                cmap=cmocean.cm.ice,
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

        # for thickness
        elif self.datatype == "h":
            cf = ax.pcolormesh(
                lon,
                lat,
                formated_data,
                cmap=cmap,
                norm=colors.Normalize(vmin=bot, vmax=top),
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

        # for temperature
        elif self.datatype == "Ti":
            cf = ax.pcolormesh(
                lon,
                lat,
                formated_data,
                cmap=cmocean.cm.thermal,
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

        # for deformation rates
        elif self.datatype in ["dedt", "shear"]:
            ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
            cf = ax.pcolormesh(
                lon,
                lat,
                formated_data,
                # np.where(self.load(datatype="A") > 0.15, formated_data, np.NaN),
                cmap=cmocean.cm.thermal,
                norm=colors.Normalize(vmin=0, vmax=0.1),
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

        elif self.datatype == "divergence":
            ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
            cf = ax.pcolormesh(
                lon,
                lat,
                formated_data,
                # np.where(self.load(datatype="A") > 0.15, formated_data, np.NaN),
                cmap=cmocean.cm.curl,
                norm=colors.Normalize(vmin=-0.3, vmax=0.3),
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

        # for viscosity
        elif self.datatype == "viscosity":
            print(formated_data.shape)
            cf = ax.pcolormesh(
                lon,
                lat,
                formated_data,
                # np.where(self.load(datatype="A") > 0.15, formated_data, np.NaN),
                cmap=cmocean.cm.amp,
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

        # for damage
        elif self.datatype == "dam":
            ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
            cf = ax.pcolormesh(
                lon,
                lat,
                formated_data,
                # np.where(self.load(datatype="A") > 0.15, formated_data, np.NaN),
                cmap=cmocean.cm.amp,
                norm=colors.Normalize(vmin=0.985, vmax=1),
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

        # for velocities
        elif self.datatype == "u":
            # verify arrow sparsing
            if self.step is None:
                self.step = 80 // self.resolution

            ax.quiver(
                lon1[0 :: self.step, 0 :: self.step],
                lat1[0 :: self.step, 0 :: self.step],
                formated_data[0 :: self.step, 0 :: self.step, 0],
                formated_data[0 :: self.step, 0 :: self.step, 1],
                color="black",
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
            # divide formated data above by this if you want arrows of same lenght.
            # / self._velolicty_vector_magnitude(
            #         formated_data[0 :: self.step, 0 :: self.step, 0],
            #         formated_data[0 :: self.step, 0 :: self.step, 1],
            #     )
            cf = ax.pcolormesh(
                lon,
                lat,
                self._velolicty_vector_magnitude(
                    formated_data[:, :, 0], formated_data[:, :, 1]
                ),
                cmap=cmocean.cm.speed,
                transform=ccrs.PlateCarree(),
                zorder=0,
            )

        else:
            raise SystemExit("Something is wrong with your data type...")

        # ax.gridlines(zorder=2)
        ax.add_feature(cfeature.LAND, zorder=3)
        ax.coastlines(resolution="50m", zorder=4)

        if axi is None:
            cbar = fig.colorbar(cf)
            cbar.ax.set_ylabel(self.name, rotation=-90, va="bottom")
            if self.save:
                fig.savefig(
                    "images/"
                    + self.datatype
                    + str(self.resolution)
                    + self.fig_name_supp
                    + "."
                    + self.fig_type,
                )
                fig.savefig(
                    "images/"
                    + self.datatype
                    + str(self.resolution)
                    + self.fig_name_supp
                    + ".pdf",
                )

        return cf

    def _encircle(
        self, x: np.ndarray, y: np.ndarray, ax: matplotlib.axes.SubplotBase
    ):
        """
        Function that computes the polygon around the relevant data in order to trace the taken values only (for plots only)

        Args:
            x (np.ndarray): x coordinates of the data 1D
            y (np.ndarray): y coordinates of the data 1D
            ax (matplotlib.axes.SubplotBase): the axis where to plot polygon
        """
        points = np.c_[x, y]
        # alpha = 0.95 * alphashape.optimizealpha(points)
        hull = alphashape.alphashape(points, 0.3)
        ax.add_patch(
            PolygonPatch(
                hull, fill=False, color="black", transform=ccrs.PlateCarree()
            )
        )

        return ax

    def arctic_plot_RGPS(
        self, data: np.ndarray, datatype: str, fig_name_supp: str = "_", ax=None
    ):
        """
        Function that plots data over the Arctic same as the other one but specifically for RGPS.

        Args:
            data (np.ndarray): data to plot in 2D
            datatype (str): type of the data
            fig_name_supp (str, optional): supplementary figure description in the name when saving the figure. Defaults to "_".

        Returns:
            cf
        """
        x0 = np.arange(data.shape[0] + 1) * RES_RGPS - 2300
        y0 = np.arange(data.shape[1] + 1) * RES_RGPS - 1000

        lon, lat = self._coordinates(x0, y0, RGPS=True)

        axi = ax
        if axi is None:
            # figure initialization
            fig = plt.figure(dpi=300)
            if self.trans:
                fig.patch.set_facecolor("None")
            ax = plt.subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
            fig.subplots_adjust(
                bottom=0.05, top=0.95, left=0.04, right=0.95, wspace=0.02
            )

        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        if self.fig_shape == "round":
            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)

        # Limit the map to 65 degrees latitude and above.
        ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())

        # all the plots
        # for deformation rates
        if datatype == "div":
            ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
            cf = ax.pcolormesh(
                lon,
                lat,
                data,
                # np.where(self.load(datatype="A") > 0.15, formated_data, np.NaN),
                cmap=cmocean.cm.curl,
                norm=colors.Normalize(vmin=-0.04, vmax=0.04),
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
            if axi is None:
                cbar = fig.colorbar(cf)
                cbar.ax.set_ylabel("[day$^{-1}$]", rotation=-90, va="bottom")

        elif datatype == "mask":
            ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
            cf = ax.pcolormesh(
                lon,
                lat,
                data * 100,
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
            x1 = np.arange(data.shape[0]) * RES_RGPS - 2300
            y1 = np.arange(data.shape[1]) * RES_RGPS - 1000

            lon1, lat1 = self._coordinates(x1, y1, RGPS=True)

            ax.contour(
                lon1,
                lat1,
                data * 100,
                levels=np.array([80]),
                transform=ccrs.PlateCarree(),
            )
            if axi is None:
                cbar = fig.colorbar(cf)
                cbar.ax.set_ylabel(
                    "Temporal presence [%]", rotation=-90, va="bottom"
                )

        else:
            ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
            cf = ax.pcolormesh(
                lon,
                lat,
                data,
                cmap=cmocean.cm.thermal,
                norm=colors.Normalize(vmin=0, vmax=0.1),
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
            if axi is None:
                cbar = fig.colorbar(cf)
                cbar.ax.set_ylabel(
                    "Ice deformation rate [day$^{-1}$]",
                    rotation=-90,
                    va="bottom",
                )

        # ax.gridlines(zorder=2)
        ax.add_feature(cfeature.LAND, zorder=3)
        ax.coastlines(resolution="50m", zorder=4)
        ax.text(
            0.5,
            1.05,
            "RGPS",
            fontsize=10,
            transform=ax.transAxes,
            va="center",
            ha="center",
        )

        if axi is None:
            if self.save:
                fig.savefig(
                    "images/"
                    + datatype
                    + str(self.resolution)
                    + fig_name_supp
                    + "RGPS"
                    + "."
                    + self.fig_type,
                )
                fig.savefig(
                    "images/"
                    + datatype
                    + str(self.resolution)
                    + fig_name_supp
                    + "RGPS"
                    + ".pdf",
                )

        return cf

    def multi_fig_precond(
        self, x: int, y: int, total: int, special: list = None
    ):
        """
        This function creates axis for multiplot of Arctic.

        Args:
            x (int): number of pictures on the x axis.

            y (int): number of pictures on the y axis.

            total (int): total number of plots to make in the grid.

            special (list, optional): list of indices in the flatten grid x by y to to not draw plots.

        Returns:
            fig, axss
        """
        # figure initialization
        fig, axss = plt.subplots(
            y,
            x,
            dpi=300,
            figsize=(2 * x + 1, 2 * y),
            subplot_kw={"projection": ccrs.NorthPolarStereo()},
        )
        fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.8196)
        axssf = axss.flatten()
        k = x * y - total
        if k > 0:
            if special is None:
                [ax.set_axis_off() for ax in axssf[-k:].ravel()]
                for i, ax in enumerate(axssf[:-k]):
                    ax.text(
                        -0.1,
                        1.1,
                        LETTER[i],
                        transform=ax.transAxes,
                        fontsize=10,
                        fontweight="normal",
                        va="top",
                        ha="left",
                    )
            else:
                [ax.set_axis_off() for ax in axssf[special].ravel()]
                for i, ax in enumerate(np.delete(axssf, special)):
                    ax.text(
                        -0.1,
                        1.1,
                        LETTER[i],
                        transform=ax.transAxes,
                        fontsize=10,
                        fontweight="normal",
                        va="top",
                        ha="left",
                    )
        elif k < 0:
            print("You need {} more axis in order to do this plot.".format(-k))
            exit()

        elif k == 0:
            for i, ax in enumerate(axssf):
                ax.text(
                    -0.1,
                    1.1,
                    LETTER[i],
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="normal",
                    va="top",
                    ha="left",
                )
        if self.trans:
            fig.patch.set_facecolor("None")

        return fig, axssf

    def multi_fig(
        self,
        fig,
        cf,
        save: bool = 0,
        label: str = "Ice deformation rate [day$^{-1}$]",
    ):
        """
        This function takes all arctic plots and put them in a nice array of dim (x,y).

        Args:
            fig: figure to save and plot
            ax: axis of data
            cf: plots of data

            save (bool): to save or not to save fig.
            label (str): name and units of the colorbar.
        """
        cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
        cbar = fig.colorbar(cf, cax=cbar_ax, extend="max")
        cbar.ax.set_ylabel(
            label,
            rotation=-90,
            va="bottom",
        )

        if save:
            fig.savefig(
                "images/M"
                + self.datatype
                + str(self.resolution)
                + self.fig_name_supp
                + "."
                + self.fig_type,
            )
            fig.savefig(
                "images/M"
                + self.datatype
                + str(self.resolution)
                + self.fig_name_supp
                + ".pdf",
            )

    def scale_plot_vect(
        self,
        deformation: np.ndarray,
        scaling: np.ndarray,
        scales: list,
        fig_name_supp: str,
        viscosity: np.ndarray = None,
        save: bool = True,
    ):
        """
        This function plots the spatial scale and computes the exponent of the scaling <dedt> ~ L^-H by doing a linear regression. It is the same as above but for the vectorized version of the code.

        Args:
            deformation (np.ndarray): array of the data inside each box. Shape (nL, ny, nx, nt). The first index is the studied scale.

            scaling (np.ndarray): array of the data for the scaling same shape as deformation.

            scales (list): list of all scales under study.

            viscosity (np.ndarray): if we want to plot colors for the viscosity norm. Give the data array.
        """

        fig = plt.figure(dpi=300, figsize=(6, 4))
        if self.trans:
            fig.patch.set_facecolor("None")

        # initialization of the list containing the means
        mean_def = np.zeros(len(scales))
        mean_scale = np.zeros(len(scales))
        # mean_def_cut = np.empty(len(scales))
        # mean_scale_cut = np.empty(len(scales))

        # definitions for the axes
        left, width = 0.14, 0.53
        bottom, height = 0.12, 0.8
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        ax = fig.add_axes(rect_scatter)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)

        # now determine nice limits by hand for the histogram
        ymax = []
        ymin = []
        all_def = []
        for k in range(len(scales)):
            ymax.append(np.nanmax(np.abs(deformation[k])))
            ymin.append(np.nanmin(np.abs(deformation[k])))
            if ymin[k] < 1e-10:
                ymin[k] = 1e-10
            all_def.append(deformation[k].flatten())
        all_def_array = np.concatenate(all_def)
        n = np.logspace(
            np.log10(np.min(np.asarray(ymin))),
            np.log10(np.max(np.asarray(ymax))),
            num=50,
        )
        ax_histy.hist(
            all_def_array,
            bins=n,
            orientation="horizontal",
            color="xkcd:dark blue grey",
        )

        # loop over the scales
        for k in range(len(scales)):
            # compute the means, ignoring NaNs
            indices1 = ~np.isnan(scaling[k])
            indices2 = ~np.isnan(deformation[k])
            indices = indices1 * indices2
            mean_def[k] = np.nanmean(deformation[k])
            mean_scale[k] = np.nanmean(scaling[k])

            # colormap
            base = cm.get_cmap("cmo.haline", 256)
            newcolors = base(np.linspace(0, 1, 256))
            bot = np.array([100 / 256, 20 / 256, 20 / 256, 1])
            newcolors[:51, :] = bot
            newcmp = ListedColormap(newcolors)
            # plot
            cf = ax.scatter(
                scaling[k][indices],
                deformation[k][indices],
                c=np.zeros_like(
                    deformation[k][indices]
                ),  # viscosity[k, indices],
                s=0.5,
                cmap=newcmp,
                norm=colors.Normalize(vmin=0, vmax=5 * ETA_MAX * E**2),
            )
            # same thing with only viscosities that are under visc_max (plastic def)
            # viscosity[k, viscosity[k] >= ETA_MAX * E ** 2] = np.NaN
            # indices = ~np.isnan(viscosity[k])
            # mean_def_cut[k] = np.average(deformation[k, indices, 0],)
            # mean_scale_cut[k] = np.average(deformation[k, indices, 1],)

        # add color bar
        cbar = fig.colorbar(cf, ax=ax_histy)
        cbar.set_label(
            "Bulk viscosity [N$\cdot$s$\cdot$m$^{-1}$]",
            rotation=-90,
            va="bottom",
        )
        # add red line for zeta max
        cax = cbar.ax
        cax.hlines(
            ETA_MAX * E**2,
            0,
            ETA_MAX * E**2 * 10,
            colors="r",
            linewidth=2,
        )
        # find the pre existing ticks
        ticks = [0, 0.5e13, 1e13, 1.5e13, 2e13]
        tick_labels = ["{:.1e}".format(tick) for tick in ticks]
        # set major ticks
        cax.yaxis.set_ticks(ticks)
        cax.yaxis.set_ticklabels(tick_labels)
        # set minor tick label for zeta max
        minortick = [4e12]
        minortick_label = ["$\zeta_{max}$"]
        cax.yaxis.set_ticks(minortick, minor=True)
        cax.yaxis.set_ticklabels(minortick_label, minor=True)
        cax.tick_params(
            axis="y",
            which="minor",
            labelsize=8,
            length=3.5,
            color="r",
            width=2,
        )
        # format the new ticks
        cax_format = matplotlib.ticker.ScalarFormatter()
        cax.yaxis.set_major_formatter(cax_format)
        cax.ticklabel_format(axis="y", style="sci")
        # linear regression over the means
        coefficients = np.polyfit(np.log(mean_scale), np.log(mean_def), 1)
        fit = np.poly1d(coefficients)
        t = np.linspace(mean_scale[0], mean_scale[-1], 10)
        # coefficients_cut = np.polyfit(
        #     np.log(mean_scale_cut), np.log(mean_def_cut), 1
        # )
        # fit_cut = np.poly1d(coefficients_cut)
        # t_cut = np.linspace(mean_scale_cut[0], mean_scale_cut[-1], 10)

        # correlation
        corr, _ = pearsonr(mean_scale, mean_def)
        # corr_cut, _ = pearsonr(mean_scale_cut, mean_def_cut)

        # plots for the means on all data
        ax.plot(
            mean_scale,
            mean_def,
            "^",
            color="xkcd:dark blue grey",
            label="H = {:.2f}, corr = {:.2f}".format(coefficients[0], corr),
            markersize=5,
        )
        ax.plot(t, np.exp(fit(np.log(t))), color="xkcd:dark blue grey")
        # plot means for plastic data
        # ax.plot(
        #     mean_scale_cut,
        #     mean_def_cut,
        #     "v",
        #     color="xkcd:golden rod",
        #     label="H = {:.2f}, corr = {:.2f}".format(
        #         coefficients_cut[0], corr_cut
        #     ),
        #     markersize=5,
        # )
        # ax.plot(t_cut, np.exp(fit_cut(np.log(t_cut))), color="xkcd:golden rod")
        ax.legend(loc=4, fontsize="x-small")
        # ticks style
        ax.grid(linestyle=":")
        ax.tick_params(
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
            labelleft=True,
        )
        ax_histy.grid(linestyle=":")
        ax_histy.tick_params(
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
            labelleft=False,
        )
        # axe labels
        ax.set_xlabel("Spatial scale [km]")
        ax.set_ylabel("Total deformation rate [day$^{-1}$]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin=8, xmax=7e2)
        # ax.set_title("H = {:.2f}, correlation = {:.2f}".format(coefficients[0], corr))
        if save:
            fig.savefig(
                "images/ss{}".format(self.resolution)
                + fig_name_supp
                + ".{}".format(self.fig_type)
            )
            fig.savefig(
                "images/ss{}".format(self.resolution) + fig_name_supp + ".pdf",
                format="pdf",
            )

        return mean_def, mean_scale, coefficients[0]

    def _multiplot_precond(self, type: bool):
        """
        Function that contains all the preconditions for the multiplot figure.

        Returns:
            fig: the figure
            ax: the axis in the figure
        """
        fig = plt.figure(dpi=300, figsize=(6, 4))
        if self.trans:
            fig.patch.set_facecolor("None")

        # definitions for the axes
        left, width = 0.12, 0.7
        bottom, height = 0.12, 0.8

        rect_scatter = [left, bottom, width, height]
        ax = fig.add_axes(rect_scatter)

        # ticks style
        ax.grid(linestyle=":")
        ax.tick_params(
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
        )
        ax.tick_params(
            which="minor",
            labelleft=False,
        )
        # axe labels
        if type == 0:  # space
            ax.set_xlabel("Spatial scale [km]")
            ax.set_ylabel(r"$\langle\dot\varepsilon_{tot}\rangle$ [day$^{-1}$]")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(ymin=6e-3, ymax=6e-2)
            ax.set_xlim(xmin=7, xmax=9e2)
            # ax.set_title("H = {:.2f}, correlation = {:.2f}".format(coefficients[0], corr))
        if type == 1:  # time
            ax.set_xlabel("Temporal scale [days]")
            ax.set_ylabel(r"$\langle\dot\varepsilon_{tot}\rangle$ [day$^{-1}$]")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(ymin=6e-3, ymax=6e-2)
            ax.set_xlim(xmin=2, xmax=4e1)
            # ax.set_title("H = {:.2f}, correlation = {:.2f}".format(coefficients[0], corr))

        return fig, ax

    def multi_plot_spatial(
        self,
        mean_def: np.ndarray,
        mean_scale: np.ndarray,
        fig_name_supp: str,
        names_plot: np.ndarray,
        colors_plot: np.ndarray,
        save: bool = True,
    ):
        """
        Function that computes and plots the multiplot figure for the deformation vs lenght scale.

        Args:
            mean_def (np.ndarray): simple array of the mean values of the wanted deformation along dim 0 and each experiment is on axis 1.
            mean_scale (np.ndarray): simple array of the mean values of the lenght scale along dim 0 and each experiment is on axis 1.
            fig_name_supp (str): name to put in figure name
            save (bool, optional): Should you want to save it. Defaults to True.
        """
        fig, ax = self._multiplot_precond(0)

        # loop over
        for k in range(mean_def.shape[1]):
            # linear regression over the means
            coefficients = np.polyfit(
                np.log(mean_scale[:, k]), np.log(mean_def[:, k]), 1
            )
            fit = np.poly1d(coefficients)
            t = np.linspace(mean_scale[0, k], mean_scale[-1, k], 10)

            # correlation
            corr, _ = pearsonr(mean_scale[:, k], mean_def[:, k])
            # corr_cut, _ = pearsonr(mean_scale_cut, mean_def_cut)

            # plots for the means on all data
            ax.plot(
                mean_scale[:, k],
                mean_def[:, k],
                ".",
                color=colors_plot[k],
                label="{:.2f}".format(np.abs(coefficients[0])),
                markersize=5,
            )
            ax.plot(t, np.exp(fit(np.log(t))), color=colors_plot[k])

        legend1 = ax.legend(
            loc=1,
            fontsize="x-small",
            frameon=False,
            labelcolor=colors_plot,
            handlelength=0,
            handletextpad=0,
            markerscale=0,
            title=r"$\beta$",
        )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labels=names_plot,
            fontsize="x-small",
            frameon=False,
            labelcolor=colors_plot,
            handlelength=0,
            handletextpad=0,
            markerscale=0,
            ncol=1,
        )
        plt.gca().add_artist(legend1)

        if save:
            fig.savefig(
                "images/ssm{}".format(self.resolution)
                + fig_name_supp
                + ".{}".format(self.fig_type),
            )
            fig.savefig(
                "images/ssm{}".format(self.resolution) + fig_name_supp + ".pdf",
            )

    def multi_plot_temporal(
        self,
        mean_def: np.ndarray,
        mean_scale: np.ndarray,
        fig_name_supp: str,
        names_plot: np.ndarray,
        colors_plot: np.ndarray,
        save: bool = True,
    ):
        """
        Function that computes and plots the multiplot figure for the deformation vs temporal scale.

        Args:
            mean_def (np.ndarray): simple array of the mean values of the wanted deformation along dim 0 and each experiment is on axis 1.
            mean_scale (np.ndarray): simple array of the values of the temporal scale.
            fig_name_supp (str): name to put in figure name
            save (bool, optional): Should you want to save it. Defaults to True.
        """
        fig, ax = self._multiplot_precond(1)

        # loop over
        for k in range(mean_def.shape[1]):
            # linear regression over the means
            coefficients = np.polyfit(
                np.log(mean_scale[:, k]), np.log(mean_def[:, k]), 1
            )
            fit = np.poly1d(coefficients)
            t = np.linspace(mean_scale[0, k], mean_scale[-1, k], 10)

            # correlation
            corr, _ = pearsonr(mean_scale[:, k], mean_def[:, k])
            # corr_cut, _ = pearsonr(mean_scale_cut, mean_def_cut)

            # plots for the means on all data
            ax.plot(
                mean_scale[:, k],
                mean_def[:, k],
                ".",
                color=colors_plot[k],
                label="{:.2f}".format(np.abs(coefficients[0])),
                markersize=5,
            )
            ax.plot(t, np.exp(fit(np.log(t))), color=colors_plot[k])

        legend1 = ax.legend(
            loc=1,
            fontsize="x-small",
            frameon=False,
            labelcolor=colors_plot,
            handlelength=0,
            handletextpad=0,
            markerscale=0,
            title=r"$\alpha$",
        )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labels=names_plot,
            fontsize="x-small",
            frameon=False,
            labelcolor=colors_plot,
            handlelength=0,
            handletextpad=0,
            markerscale=0,
            ncol=1,
        )
        plt.gca().add_artist(legend1)

        if save:
            fig.savefig(
                "images/ssmT{}".format(self.resolution)
                + fig_name_supp
                + ".{}".format(self.fig_type),
            )
            fig.savefig(
                "images/ssmT{}".format(self.resolution)
                + fig_name_supp
                + ".pdf",
            )

    def pdf_du(
        self,
        du_stack: list,
        save: bool,
        fig_name_supp: string,
        names_plot: np.ndarray,
        colors_plot: np.ndarray,
    ):
        """
        It simply computes everything for pdf plot. RGPS must be the FIRST in the stack!

        Args:
            du_stack (list): velocity derivatives of each model on last axis. we need list because RGPS is not same shape...
            save (bool): save or not the fig
            fig_name_supp (string): supplmentary info for fig name.

        Returns:
            [type]: figure of pdf
        """
        # init plot
        basefigsizey = 3
        basefigsizex = 8.2
        figsizex = basefigsizex
        figsizey = basefigsizey + 0.12 * (len(du_stack) - 1)
        fig = plt.figure(
            dpi=300,
            figsize=(figsizex, figsizey),
        )
        if self.trans:
            fig.patch.set_facecolor("None")

        # definitions for the axis
        extraspaceinit = 0.093 * figsizex / basefigsizex
        graphsizex = (0.267 - extraspaceinit / 3) * basefigsizex / figsizex
        separation = (1 - extraspaceinit - 3 * graphsizex) / 4

        bottom, height = (
            1 - 0.52 * basefigsizey / figsizey,
            0.45 * basefigsizey / figsizey,
        )
        bottomB, heightB = (
            0.17 * basefigsizey / figsizey,
            0.045 * len(du_stack) * basefigsizey / figsizey,
        )

        left_shear, width_shear = separation + extraspaceinit, graphsizex
        rect_scatter_shear = [
            left_shear,
            bottom,
            width_shear,
            height,
        ]

        left_ndiv, width_ndiv = (
            2 * separation + extraspaceinit + graphsizex,
            graphsizex,
        )
        rect_scatter_ndiv = [
            left_ndiv,
            bottom,
            width_ndiv,
            height,
        ]

        left_pdiv, width_pdiv = (
            3 * separation + extraspaceinit + 2 * graphsizex,
            graphsizex,
        )
        rect_scatter_pdiv = [
            left_pdiv,
            bottom,
            width_pdiv,
            height,
        ]

        left_shearB, width_shearB = separation + extraspaceinit, graphsizex
        rect_scatter_shearB = [
            left_shearB,
            bottomB,
            width_shearB,
            heightB,
        ]

        left_ndivB, width_ndivB = (
            2 * separation + extraspaceinit + graphsizex,
            graphsizex,
        )
        rect_scatter_ndivB = [
            left_ndivB,
            bottomB,
            width_ndivB,
            heightB,
        ]

        left_pdivB, width_pdivB = (
            3 * separation + extraspaceinit + 2 * graphsizex,
            graphsizex,
        )
        rect_scatter_pdivB = [
            left_pdivB,
            bottomB,
            width_pdivB,
            heightB,
        ]

        ax_shear = fig.add_axes(rect_scatter_shear)
        ax_ndiv = fig.add_axes(rect_scatter_ndiv)
        ax_pdiv = fig.add_axes(rect_scatter_pdiv)

        ax_shearB = fig.add_axes(rect_scatter_shearB)
        ax_ndivB = fig.add_axes(rect_scatter_ndivB)
        ax_pdivB = fig.add_axes(rect_scatter_pdivB)

        # ticks
        ax_shear.grid(
            axis="x", which="minor", linestyle=":", color="xkcd:light gray"
        )
        ax_shear.grid(
            axis="x", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_shear.grid(
            axis="y", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_shear.tick_params(
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
            labelleft=True,
        )

        ax_ndiv.grid(
            axis="x", which="minor", linestyle=":", color="xkcd:light gray"
        )
        ax_ndiv.grid(
            axis="x", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_ndiv.grid(
            axis="y", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_ndiv.tick_params(
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
            labelleft=True,
        )

        ax_pdiv.grid(
            axis="x", which="minor", linestyle=":", color="xkcd:light gray"
        )
        ax_pdiv.grid(
            axis="x", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_pdiv.grid(
            axis="y", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_pdiv.tick_params(
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
            labelleft=True,
        )

        ax_shearB.tick_params(
            which="both",
            direction="out",
            bottom=False,
            top=True,
            left=True,
            right=False,
            labelleft=True,
        )

        ax_ndivB.tick_params(
            which="both",
            direction="out",
            bottom=False,
            top=True,
            left=False,
            right=True,
            labelright=True,
        )

        ax_pdivB.tick_params(
            which="both",
            direction="out",
            bottom=False,
            top=True,
            left=True,
            right=False,
            labelleft=True,
        )

        model_diff_shear = []
        model_diff_ndiv = []
        model_diff_pdiv = []
        for k in range(len(du_stack)):
            shear = self._deformation(du_stack[k], 1)
            ndiv = np.where(
                self._deformation(du_stack[k], 2) < 0,
                -self._deformation(du_stack[k], 2),
                np.NaN,
            )
            pdiv = np.where(
                self._deformation(du_stack[k], 2) > 0,
                self._deformation(du_stack[k], 2),
                np.NaN,
            )

            shear_cut = shear[~np.isnan(shear)]
            ndiv_cut = ndiv[~np.isnan(ndiv)]
            pdiv_cut = pdiv[~np.isnan(pdiv)]

            # get correct data from box data
            n = np.logspace(np.log10(5e-3), 0, num=25)
            p_shear, x_shear = np.histogram(shear_cut, bins=n, density=1)
            p_ndiv, x_ndiv = np.histogram(ndiv_cut, bins=n, density=1)
            p_pdiv, x_pdiv = np.histogram(pdiv_cut, bins=n, density=1)
            p_shear = np.where(p_shear == 0.0, np.NaN, p_shear)
            p_ndiv = np.where(p_ndiv == 0.0, np.NaN, p_ndiv)
            p_pdiv = np.where(p_pdiv == 0.0, np.NaN, p_pdiv)

            # convert bin edges to centers
            x_shear_mid = (x_shear[:-1] + x_shear[1:]) / 2
            x_ndiv_mid = (x_ndiv[:-1] + x_ndiv[1:]) / 2
            x_pdiv_mid = (x_pdiv[:-1] + x_pdiv[1:]) / 2

            # save RGPS
            if k == 0:
                RGPS_p_shear = p_shear
                RGPS_p_ndiv = p_ndiv
                RGPS_p_pdiv = p_pdiv

            # not RGPS which is the first one.
            if k != 0:
                logdiff_shear = np.log10(p_shear) - np.log10(RGPS_p_shear)
                logdiff_ndiv = np.log10(p_ndiv) - np.log10(RGPS_p_ndiv)
                logdiff_pdiv = np.log10(p_pdiv) - np.log10(RGPS_p_pdiv)
                model_diff_shear.append(logdiff_shear)
                model_diff_ndiv.append(logdiff_ndiv)
                model_diff_pdiv.append(logdiff_pdiv)

            # variables for fit definitions
            indices_shear = []
            indices_ndiv = []
            indices_pdiv = []
            for i in range(p_shear.shape[0]):
                if np.isnan(p_shear[i]):
                    indices_shear.append(i)
            for i in range(p_ndiv.shape[0]):
                if np.isnan(p_ndiv[i]):
                    indices_ndiv.append(i)
            for i in range(p_pdiv.shape[0]):
                if np.isnan(p_pdiv[i]):
                    indices_pdiv.append(i)

            if (
                len(indices_shear) == 0
                and len(indices_ndiv) == 0
                and len(indices_pdiv) == 0
            ):
                t_shear = x_shear_mid
                t_ndiv = x_ndiv_mid
                t_pdiv = x_pdiv_mid
                pt_shear = p_shear
                pt_ndiv = p_ndiv
                pt_pdiv = p_pdiv
            elif (
                len(indices_shear) == 0
                and len(indices_ndiv) == 0
                and len(indices_pdiv) != 0
            ):
                t_shear = x_shear_mid
                t_ndiv = x_ndiv_mid
                t_pdiv = np.delete(x_pdiv_mid, np.asarray(indices_pdiv))
                pt_shear = p_shear
                pt_ndiv = p_ndiv
                pt_pdiv = np.delete(p_pdiv, np.asarray(indices_pdiv))
            elif (
                len(indices_shear) != 0
                and len(indices_ndiv) == 0
                and len(indices_pdiv) == 0
            ):
                t_shear = np.delete(x_shear_mid, np.asarray(indices_shear))
                t_ndiv = x_ndiv_mid
                t_pdiv = x_pdiv_mid
                pt_shear = np.delete(p_shear, np.asarray(indices_shear))
                pt_ndiv = p_ndiv
                pt_pdiv = p_pdiv
            elif (
                len(indices_shear) != 0
                and len(indices_ndiv) == 0
                and len(indices_pdiv) != 0
            ):
                t_shear = np.delete(x_shear_mid, np.asarray(indices_shear))
                t_ndiv = x_ndiv_mid
                t_pdiv = np.delete(x_pdiv_mid, np.asarray(indices_pdiv))
                pt_shear = np.delete(p_shear, np.asarray(indices_shear))
                pt_ndiv = p_ndiv
                pt_pdiv = np.delete(p_pdiv, np.asarray(indices_pdiv))
            elif (
                len(indices_shear) == 0
                and len(indices_ndiv) != 0
                and len(indices_pdiv) == 0
            ):
                t_shear = x_shear_mid
                t_ndiv = np.delete(x_ndiv_mid, np.asarray(indices_ndiv))
                t_pdiv = x_pdiv_mid
                pt_shear = p_shear
                pt_ndiv = np.delete(p_ndiv, np.asarray(indices_ndiv))
                pt_pdiv = p_pdiv
            elif (
                len(indices_shear) == 0
                and len(indices_ndiv) != 0
                and len(indices_pdiv) != 0
            ):
                t_shear = x_shear_mid
                t_ndiv = np.delete(x_ndiv_mid, np.asarray(indices_ndiv))
                t_pdiv = np.delete(x_pdiv_mid, np.asarray(indices_pdiv))
                pt_shear = p_shear
                pt_ndiv = np.delete(p_ndiv, np.asarray(indices_ndiv))
                pt_pdiv = np.delete(p_pdiv, np.asarray(indices_pdiv))
            elif (
                len(indices_shear) != 0
                and len(indices_ndiv) != 0
                and len(indices_pdiv) == 0
            ):
                t_shear = np.delete(x_shear_mid, np.asarray(indices_shear))
                t_ndiv = np.delete(x_ndiv_mid, np.asarray(indices_ndiv))
                t_pdiv = x_pdiv_mid
                pt_shear = np.delete(p_shear, np.asarray(indices_shear))
                pt_ndiv = np.delete(p_ndiv, np.asarray(indices_ndiv))
                pt_pdiv = p_pdiv
            elif (
                len(indices_shear) != 0
                and len(indices_ndiv) != 0
                and len(indices_pdiv) != 0
            ):
                t_shear = np.delete(x_shear_mid, np.asarray(indices_shear))
                t_ndiv = np.delete(x_ndiv_mid, np.asarray(indices_ndiv))
                t_pdiv = np.delete(x_pdiv_mid, np.asarray(indices_pdiv))
                pt_shear = np.delete(p_shear, np.asarray(indices_shear))
                pt_ndiv = np.delete(p_ndiv, np.asarray(indices_ndiv))
                pt_pdiv = np.delete(p_pdiv, np.asarray(indices_pdiv))

            # fit
            coeff_shear = (
                np.polynomial.Polynomial.fit(
                    np.log10(t_shear[-9:]), np.log10(pt_shear[-9:]), 1
                )
                .convert()
                .coef
            )
            coeff_ndiv = (
                np.polynomial.Polynomial.fit(
                    np.log10(t_ndiv[-9:]), np.log10(pt_ndiv[-9:]), 1
                )
                .convert()
                .coef
            )
            coeff_pdiv = (
                np.polynomial.Polynomial.fit(
                    np.log10(t_pdiv[-9:]), np.log10(pt_pdiv[-9:]), 1
                )
                .convert()
                .coef
            )
            best_fit_shear = np.polynomial.Polynomial(coeff_shear)
            best_fit_ndiv = np.polynomial.Polynomial(coeff_ndiv)
            best_fit_pdiv = np.polynomial.Polynomial(coeff_pdiv)

            # plots
            ax_shear.plot(
                x_shear_mid,
                p_shear,
                color=colors_plot[k],
                label="{:.1f}".format(-coeff_shear[-1]),
            )

            ax_ndiv.plot(
                x_ndiv_mid,
                p_ndiv,
                color=colors_plot[k],
                label="{:.1f}".format(-coeff_ndiv[-1]),
            )

            ax_pdiv.plot(
                x_pdiv_mid,
                p_pdiv,
                color=colors_plot[k],
                label="{:.1f}".format(-coeff_pdiv[-1]),
            )

        # model diff plots
        model_diff_shear = np.asarray(model_diff_shear)
        model_diff_ndiv = np.asarray(model_diff_ndiv)
        model_diff_pdiv = np.asarray(model_diff_pdiv)

        list_y = np.arange(0, model_diff_shear.shape[0] + 1)

        splot = ax_shearB.pcolormesh(
            n,
            list_y,
            model_diff_shear,
            cmap="coolwarm",  # "BrBG",
            norm=colors.Normalize(vmin=-1, vmax=1),
            linewidth=0.005,
        )
        ndplot = ax_ndivB.pcolormesh(
            n,
            list_y,
            model_diff_ndiv,
            cmap="coolwarm",  # "BrBG",
            norm=colors.Normalize(vmin=-1, vmax=1),
            linewidth=0.005,
        )
        pdplot = ax_pdivB.pcolormesh(
            n,
            list_y,
            model_diff_pdiv,
            cmap="coolwarm",  # "BrBG",
            norm=colors.Normalize(vmin=-1, vmax=1),
            linewidth=0.005,
        )

        # compute numbers for best fit
        num_shear = np.nanmean(np.abs(model_diff_shear), axis=1)
        num_ndiv = np.nanmean(np.abs(model_diff_ndiv), axis=1)
        num_pdiv = np.nanmean(np.abs(model_diff_pdiv), axis=1)

        # plot numbers for best fit
        for y in range(model_diff_shear.shape[0]):
            ax_shearB.text(
                2.9,
                y + 0.4,
                "{:.2f}".format(num_shear[y]),
                ha="right",
                va="center",
                fontsize=8,
            )
        for y in range(model_diff_ndiv.shape[0]):
            ax_ndivB.text(
                2.9,
                y + 0.4,
                "{:.2f}".format(num_ndiv[y]),
                ha="left",
                va="center",
                fontsize=8,
            )
        for y in range(model_diff_pdiv.shape[0]):
            ax_pdivB.text(
                2.9,
                y + 0.4,
                "{:.2f}".format(num_pdiv[y]),
                ha="right",
                va="center",
                fontsize=8,
            )

        # axis labels
        ax_shear.set_xlabel("Shear strain rate [day$^{-1}$]")
        ax_shear.set_ylabel("PDF")
        ax_shear.set_xscale("log")
        ax_shear.set_yscale("log")
        ax_shear.set_ylim(ymin=1e-5, ymax=1e3)
        ax_shear.set_xlim(xmin=5e-3, xmax=3)
        ax_shear.locator_params(axis="y", numticks=5)

        ax_ndiv.set_xlabel("Convergence [day$^{-1}$]")
        ax_ndiv.set_yscale("log")
        ax_ndiv.set_xscale("log")
        ax_ndiv.set_ylim(ymin=1e-5, ymax=1e3)
        ax_ndiv.set_xlim(xmin=5e-3, xmax=3)
        ax_ndiv.locator_params(axis="y", numticks=5)
        ax_ndiv.set_yticklabels([])
        ticks = [1e0, 1e-1, 1e-2]
        tick_labels = [r"$-10^{0}$", r"$-10^{-1}$", r"$-10^{-2}$"]
        ax_ndiv.xaxis.set_ticks(ticks)
        ax_ndiv.xaxis.set_ticklabels(tick_labels)
        ax_ndiv.invert_xaxis()

        ax_pdiv.set_xlabel("Divergence [day$^{-1}$]")
        ax_pdiv.set_xscale("log")
        ax_pdiv.set_yscale("log")
        ax_pdiv.set_ylim(ymin=1e-5, ymax=1e3)
        ax_pdiv.set_xlim(xmin=5e-3, xmax=3)
        ax_pdiv.locator_params(axis="y", numticks=5)
        ax_pdiv.set_yticklabels([])

        ax_shearB.set_xscale("log")
        ax_shearB.set_xlim(xmin=5e-3, xmax=3)
        ticksB = np.arange(0.5, len(du_stack) - 0.5, 1)
        tick_labelsB = names_plot[1 : len(du_stack)]
        ax_shearB.yaxis.set_ticks(ticksB)
        ax_shearB.yaxis.set_ticklabels(tick_labelsB, fontsize=8)
        for i in range(colors_plot[1 : len(du_stack)].shape[0]):
            ax_shearB.get_yticklabels()[i].set_color(colors_plot[i + 1])
        ax_shearB.set_xticklabels([])
        sdivider = make_axes_locatable(ax_shearB)
        scax = sdivider.append_axes("bottom", size=0.1, pad=0.1)
        scbar = fig.colorbar(splot, cax=scax, orientation="horizontal")
        scbar.ax.set_xlabel("Log difference to RGPS PDF")

        ax_ndivB.set_xscale("log")
        ax_ndivB.set_xlim(xmin=5e-3, xmax=3)
        ax_ndivB.set_yticklabels([])
        ax_ndivB.set_xticklabels([])
        ax_ndivB.yaxis.set_ticks(ticksB)
        ax_ndivB.invert_xaxis()
        nddivider = make_axes_locatable(ax_ndivB)
        ndcax = nddivider.append_axes("bottom", size=0.1, pad=0.1)
        ndcbar = fig.colorbar(ndplot, cax=ndcax, orientation="horizontal")
        ndcbar.ax.set_xlabel("Log difference to RGPS PDF")

        ax_pdivB.set_xscale("log")
        ax_pdivB.set_xlim(xmin=5e-3, xmax=3)
        ax_pdivB.set_yticklabels([])
        ax_pdivB.set_xticklabels([])
        ax_pdivB.yaxis.set_ticks(ticksB)
        pddivider = make_axes_locatable(ax_pdivB)
        pdcax = pddivider.append_axes("bottom", size=0.1, pad=0.1)
        pdcbar = fig.colorbar(pdplot, cax=pdcax, orientation="horizontal")
        pdcbar.ax.set_xlabel("Log difference to RGPS PDF")

        ax_shear.legend(
            loc=7,
            fontsize="xx-small",
            frameon=False,
            labelcolor=colors_plot,
            handlelength=0,
            handletextpad=0,
            labelspacing=0.35,
            title=r"$k_{\dot\varepsilon_{II}}$",
        )
        ax_ndiv.legend(
            loc=6,
            fontsize="xx-small",
            frameon=False,
            labelcolor=colors_plot,
            handlelength=0,
            handletextpad=0,
            labelspacing=0.35,
            title=r"$k_{\dot\varepsilon_{I<0}}$",
        )
        ax_pdiv.legend(
            loc=7,
            fontsize="xx-small",
            frameon=False,
            labelcolor=colors_plot,
            handlelength=0,
            handletextpad=0,
            labelspacing=0.35,
            title=r"$k_{\dot\varepsilon_{I>0}}$",
        )

        # axis name
        ax_shear.text(
            0,
            1.13,
            "(a)",
            transform=ax_shear.transAxes,
            fontsize=10,
            fontweight="normal",
            va="top",
            ha="left",
        )
        ax_pdiv.text(
            0,
            1.13,
            "(c)",
            transform=ax_pdiv.transAxes,
            fontsize=10,
            fontweight="normal",
            va="top",
            ha="left",
        )
        ax_ndiv.text(
            0,
            1.13,
            "(b)",
            transform=ax_ndiv.transAxes,
            fontsize=10,
            fontweight="normal",
            va="top",
            ha="left",
        )

        ax_shearB.text(
            0,
            1.175,
            "(d)",
            transform=ax_shearB.transAxes,
            fontsize=10,
            fontweight="normal",
            va="top",
            ha="left",
        )
        ax_pdivB.text(
            0,
            1.175,
            "(f)",
            transform=ax_pdivB.transAxes,
            fontsize=10,
            fontweight="normal",
            va="top",
            ha="left",
        )
        ax_ndivB.text(
            0,
            1.175,
            "(e)",
            transform=ax_ndivB.transAxes,
            fontsize=10,
            fontweight="normal",
            va="top",
            ha="left",
        )

        # save fig
        if save:
            fig.savefig(
                "images/pdfm{}".format(self.resolution)
                + fig_name_supp
                + ".{}".format(self.fig_type),
            )
            fig.savefig(
                "images/pdfm{}".format(self.resolution)
                + fig_name_supp
                + ".pdf",
                format="pdf",
            )

    def cdf_du(
        self,
        du_stack: list,
        save: bool,
        fig_name_supp: string,
        names_plot: np.ndarray,
        colors_plot: np.ndarray,
    ):
        """
        It simply computes everything for cdf plot.

        Args:
            du_stack (list): velocity derivatives of each model on last axis. we need list because RGPS is not same shape... RGPS NEEDS TO BE THE FIRST ITEM OF THIS LIST.
            save (bool): save or not the fig
            fig_name_supp (string): supplmentary info for fig name.

        Returns:
            [type]: figure of pdf
        """
        # init plot
        fig = plt.figure(dpi=300, figsize=(8, 2.2))
        if self.trans:
            fig.patch.set_facecolor("None")

        # definitions for the axis
        left_shear, width_shear = (1 - 3 * 0.28) / 4 + 0.033, 0.25
        bottom_shear, height_shear = 0.2, 0.68
        rect_scatter_shear = [
            left_shear,
            bottom_shear,
            width_shear,
            height_shear,
        ]

        left_ndiv, width_ndiv = (1 - 3 * 0.28) / 2 + 0.25 + 0.033, 0.25
        bottom_ndiv, height_ndiv = 0.2, 0.68
        rect_scatter_ndiv = [
            left_ndiv,
            bottom_ndiv,
            width_ndiv,
            height_ndiv,
        ]

        left_pdiv, width_pdiv = (
            3 * (1 - 3 * 0.28) / 4 + 2 * 0.25 + 0.033,
            0.25,
        )
        bottom_pdiv, height_pdiv = 0.2, 0.68
        rect_scatter_pdiv = [
            left_pdiv,
            bottom_pdiv,
            width_pdiv,
            height_pdiv,
        ]

        ax_shear = fig.add_axes(rect_scatter_shear)
        ax_ndiv = fig.add_axes(rect_scatter_ndiv)
        ax_pdiv = fig.add_axes(rect_scatter_pdiv)

        # ticks
        ax_shear.grid(
            axis="x", which="minor", linestyle=":", color="xkcd:light gray"
        )
        ax_shear.grid(
            axis="x", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_shear.grid(
            axis="y", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_shear.tick_params(
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
            labelleft=True,
        )
        ax_ndiv.grid(
            axis="x", which="minor", linestyle=":", color="xkcd:light gray"
        )
        ax_ndiv.grid(
            axis="x", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_ndiv.grid(
            axis="y", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_ndiv.tick_params(
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
            labelleft=True,
        )
        ax_pdiv.grid(
            axis="x", which="minor", linestyle=":", color="xkcd:light gray"
        )
        ax_pdiv.grid(
            axis="x", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_pdiv.grid(
            axis="y", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_pdiv.tick_params(
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
            labelleft=True,
        )

        shear_RGPS = self._deformation(du_stack[0], 1)
        ndiv_RGPS = np.where(
            self._deformation(du_stack[0], 2) > 0,
            np.NaN,
            -self._deformation(du_stack[0], 2),
        )
        pdiv_RGPS = np.where(
            self._deformation(du_stack[0], 2) < 0,
            np.NaN,
            self._deformation(du_stack[0], 2),
        )

        shear_RGPS_cut = shear_RGPS[~np.isnan(shear_RGPS)]
        ndiv_RGPS_cut = ndiv_RGPS[~np.isnan(ndiv_RGPS)]
        pdiv_RGPS_cut = pdiv_RGPS[~np.isnan(pdiv_RGPS)]

        # CDF, we have to cut under 0.005 for nice CDF
        shear_RGPS_cut1 = np.where(
            shear_RGPS_cut < 0.005, np.NaN, shear_RGPS_cut
        )
        ndiv_RGPS_cut1 = np.where(
            np.abs(ndiv_RGPS_cut) < 0.005, np.NaN, ndiv_RGPS_cut
        )
        pdiv_RGPS_cut1 = np.where(
            np.abs(pdiv_RGPS_cut) < 0.005, np.NaN, pdiv_RGPS_cut
        )
        shear_RGPS_cut2 = shear_RGPS_cut1[~np.isnan(shear_RGPS_cut1)]
        ndiv_RGPS_cut2 = ndiv_RGPS_cut1[~np.isnan(ndiv_RGPS_cut1)]
        pdiv_RGPS_cut2 = pdiv_RGPS_cut1[~np.isnan(pdiv_RGPS_cut1)]

        n = np.logspace(np.log10(5e-3), 0, num=1000)

        p_RGPS_shear, x_RGPS_shear = np.histogram(
            shear_RGPS_cut2, bins=n, density=1
        )
        p_RGPS_ndiv, x_RGPS_ndiv = np.histogram(
            ndiv_RGPS_cut2, bins=n, density=1
        )
        p_RGPS_pdiv, x_RGPS_pdiv = np.histogram(
            pdiv_RGPS_cut2, bins=n, density=1
        )

        dx_RGPS_shear = x_RGPS_shear[1:] - x_RGPS_shear[:-1]
        X1_RGPS_shear = (x_RGPS_shear[1:] + x_RGPS_shear[:-1]) / 2
        F1_RGPS_shear = np.cumsum(p_RGPS_shear * dx_RGPS_shear)

        dx_RGPS_ndiv = x_RGPS_ndiv[1:] - x_RGPS_ndiv[:-1]
        X1_RGPS_ndiv = (x_RGPS_ndiv[1:] + x_RGPS_ndiv[:-1]) / 2
        F1_RGPS_ndiv = np.cumsum(p_RGPS_ndiv * dx_RGPS_ndiv)

        dx_RGPS_pdiv = x_RGPS_pdiv[1:] - x_RGPS_pdiv[:-1]
        X1_RGPS_pdiv = (x_RGPS_pdiv[1:] + x_RGPS_pdiv[:-1]) / 2
        F1_RGPS_pdiv = np.cumsum(p_RGPS_pdiv * dx_RGPS_pdiv)

        for k in range(len(du_stack) - 1):
            shear = self._deformation(du_stack[k + 1], 1)
            ndiv = np.where(
                self._deformation(du_stack[k + 1], 2) > 0,
                np.NaN,
                -self._deformation(du_stack[k + 1], 2),
            )
            pdiv = np.where(
                self._deformation(du_stack[k + 1], 2) < 0,
                np.NaN,
                self._deformation(du_stack[k + 1], 2),
            )

            shear_cut = shear[~np.isnan(shear)]
            ndiv_cut = ndiv[~np.isnan(ndiv)]
            pdiv_cut = pdiv[~np.isnan(pdiv)]

            # CDF, we have to cut under 0.005 for nice CDF
            shear_cut1 = np.where(shear_cut < 0.005, np.NaN, shear_cut)
            ndiv_cut1 = np.where(np.abs(ndiv_cut) < 0.005, np.NaN, ndiv_cut)
            pdiv_cut1 = np.where(np.abs(pdiv_cut) < 0.005, np.NaN, pdiv_cut)
            shear_cut2 = shear_cut1[~np.isnan(shear_cut1)]
            ndiv_cut2 = ndiv_cut1[~np.isnan(ndiv_cut1)]
            pdiv_cut2 = pdiv_cut1[~np.isnan(pdiv_cut1)]

            p_shear, x_shear = np.histogram(shear_cut2, bins=n, density=1)
            p_ndiv, x_ndiv = np.histogram(ndiv_cut2, bins=n, density=1)
            p_pdiv, x_pdiv = np.histogram(pdiv_cut2, bins=n, density=1)

            dx_shear = x_shear[1:] - x_shear[:-1]
            X1_shear = (x_shear[1:] + x_shear[:-1]) / 2
            F1_shear = np.cumsum(p_shear * dx_shear)

            dx_ndiv = x_ndiv[1:] - x_ndiv[:-1]
            X1_ndiv = (x_ndiv[1:] + x_ndiv[:-1]) / 2
            F1_ndiv = np.cumsum(p_ndiv * dx_ndiv)

            dx_pdiv = x_pdiv[1:] - x_pdiv[:-1]
            X1_pdiv = (x_pdiv[1:] + x_pdiv[:-1]) / 2
            F1_pdiv = np.cumsum(p_pdiv * dx_pdiv)

            ks_distance_shear = np.amax(np.abs(F1_shear - F1_RGPS_shear))
            ks_distance_ndiv = np.amax(np.abs(F1_ndiv - F1_RGPS_ndiv))
            ks_distance_pdiv = np.amax(np.abs(F1_pdiv - F1_RGPS_pdiv))

            # plots
            ax_shear.plot(
                X1_shear,
                F1_shear,
                color=colors_plot[k + 1],
                label="{:.2f}".format(ks_distance_shear),
            )

            ax_ndiv.plot(
                X1_ndiv,
                F1_ndiv,
                color=colors_plot[k + 1],
                label="{:.2f}".format(ks_distance_ndiv),
            )

            ax_pdiv.plot(
                X1_pdiv,
                F1_pdiv,
                color=colors_plot[k + 1],
                label="{:.2f}".format(ks_distance_pdiv),
            )

        # RGPS plots
        ax_shear.plot(
            X1_RGPS_shear,
            F1_RGPS_shear,
            color=colors_plot[0],
        )
        ax_ndiv.plot(
            X1_RGPS_ndiv,
            F1_RGPS_ndiv,
            color=colors_plot[0],
        )

        ax_pdiv.plot(
            X1_RGPS_pdiv,
            F1_RGPS_pdiv,
            color=colors_plot[0],
        )

        # axis labels
        ax_shear.set_xlabel("Shear strain rate [day$^{-1}$]")
        ax_shear.set_ylabel("CDF")
        ax_shear.set_xscale("log")
        ax_shear.set_ylim(ymin=0, ymax=1.05)
        ax_shear.set_xlim(xmin=5e-3, xmax=1.5)

        ax_ndiv.set_xlabel("Convergence [day$^{-1}$]")
        ax_ndiv.set_xscale("log")
        ax_ndiv.set_ylim(ymin=0, ymax=1.05)
        ax_ndiv.set_xlim(xmin=5e-3, xmax=1.5)
        ax_ndiv.set_yticklabels([])
        ticks = [1e0, 1e-1, 1e-2]
        tick_labels = [r"$-10^{0}$", r"$-10^{-1}$", r"$-10^{-2}$"]
        ax_ndiv.xaxis.set_ticks(ticks)
        ax_ndiv.xaxis.set_ticklabels(tick_labels)
        ax_ndiv.invert_xaxis()

        ax_pdiv.set_xlabel("Divergence [day$^{-1}$]")
        ax_pdiv.set_xscale("log")
        ax_pdiv.set_ylim(ymin=0, ymax=1.05)
        ax_pdiv.set_xlim(xmin=5e-3, xmax=1.5)
        ax_pdiv.set_yticklabels([])

        # axis name
        ax_shear.text(
            0,
            1.13,
            "(a)",
            transform=ax_shear.transAxes,
            fontsize=10,
            fontweight="normal",
            va="top",
            ha="left",
        )
        ax_pdiv.text(
            0,
            1.13,
            "(c)",
            transform=ax_pdiv.transAxes,
            fontsize=10,
            fontweight="normal",
            va="top",
            ha="left",
        )
        ax_ndiv.text(
            0,
            1.13,
            "(b)",
            transform=ax_ndiv.transAxes,
            fontsize=10,
            fontweight="normal",
            va="top",
            ha="left",
        )

        legend1 = ax_pdiv.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labels=names_plot,
            fontsize="x-small",
            frameon=False,
            labelcolor=colors_plot,
            handlelength=0,
            handletextpad=0,
            markerscale=0,
            ncol=1,
        )

        ax_shear.legend(
            loc=4,
            fontsize="xx-small",
            frameon=False,
            labelcolor=colors_plot[1:],
            handlelength=0,
            handletextpad=0,
            title=r"$D_{\dot\varepsilon_{II}}$",
        )
        ax_ndiv.legend(
            loc=3,
            fontsize="xx-small",
            frameon=False,
            labelcolor=colors_plot[1:],
            handlelength=0,
            handletextpad=0,
            title=r"$D_{\dot\varepsilon_{I<0}}$",
        )
        ax_pdiv.legend(
            loc=4,
            fontsize="xx-small",
            frameon=False,
            labelcolor=colors_plot[1:],
            handlelength=0,
            handletextpad=0,
            title=r"$D_{\dot\varepsilon_{I>0}}$",
        )

        plt.gca().add_artist(legend1)

        # save fig
        if save:
            fig.savefig(
                "images/cdfm{}".format(self.resolution)
                + fig_name_supp
                + ".{}".format(self.fig_type),
            )
            fig.savefig(
                "images/cdfm{}".format(self.resolution)
                + fig_name_supp
                + ".pdf",
            )

    def multifractal_spatial(
        self,
        q: int,
        deps: np.ndarray,
        scale: np.ndarray,
        RGPS: bool = False,
    ) -> tuple:
        """
        Function that computes the multifractal fit with good parameters.

        Args:
            q (int): max moment to be computed
            deps (np.ndarray): deformations from spatial_mean_du
            scale (np.ndarray): scale to use from spatial_mean_du
            RGPS (bool, optional): RGPS pr not. Defaults to False.

        Returns:
            tuple: parameters, coefficients
        """

        from scipy.optimize import differential_evolution

        q_array = np.arange(0.1, q + 0.1, 0.1)
        coeff_list = []

        if RGPS == True:
            for n in q_array:
                mean_depsq, mean_scale, coeff = self.scale_plot_vect(
                    deps**n,
                    scale,
                    L_RGPS,
                    save=0,
                    fig_name_supp="_dedtQ_02_RGPS_du",
                )
                coeff_list.append(coeff)
        else:
            for n in q_array:
                mean_depsq, mean_scale, coeff = self.scale_plot_vect(
                    deps**n,
                    scale,
                    L10,
                    save=0,
                    fig_name_supp="_dedtQ_97",
                )
                coeff_list.append(coeff)

        coeff = np.abs(np.asarray(coeff_list))

        def sum_leastsqr(paramtuple):
            beta = self.structure_function(q_array, *paramtuple)

            return np.sum((coeff - beta) ** 2)

        results = differential_evolution(
            sum_leastsqr, bounds=[[0, 2], [0, 2], [0, 1]]
        )

        return results.x, coeff

    def multifractal_temporal(self, q: int, du: np.ndarray) -> tuple:
        """
        Function that computes the multifractal fit with good parameters.

        Args:
            q (int): _description_
            du (np.ndarray): _description_

        Returns:
            tuple: _description_
        """

        from scipy.optimize import differential_evolution

        q_array = np.arange(0.1, q + 0.1, 0.1)
        coeff_list = []

        for n in q_array:
            deps, __, __, __ = self.temporal_mean_du(du, T10, q=n)
            coeff = self.temporal_scaling_slope(deps, T10)
            coeff_list.append(coeff)
            print("Done with moment {:.1f}.".format(n))

        coeff = np.abs(np.asarray(coeff_list))

        def sum_leastsqr(paramtuple):
            alpha = self.structure_function(q_array, *paramtuple)

            return np.sum((coeff - alpha) ** 2)

        results = differential_evolution(
            sum_leastsqr, bounds=[[0, 2], [0, 2], [0, 1]]
        )

        return results.x, coeff

    def multifractal_plot(
        self,
        param: np.ndarray,
        coeff: np.ndarray,
        q: int,
        save: bool,
        fig_name_supp: string,
        names_plot: np.ndarray,
        colors_plot: np.ndarray,
        temp: bool = 0,
    ):
        """
        Fonction that plots the multifractal fit.

        Args:
            param (np.ndarray): parameters of the fit
            coeff (np.ndarray): coefficients of the data
            q (int): max moment
            save (bool): save or not fig
            fig_name_supp (string): fig supplemental info
            temp (bool, optional): temporal or spatial. Defaults to 0.
        """

        fig = plt.figure(dpi=300, figsize=(6, 4))
        if self.trans:
            fig.patch.set_facecolor("None")

        # definitions for the axes
        left, width = 0.14, 0.7
        bottom, height = 0.14, 0.75

        rect_scatter = [left, bottom, width, height]
        ax = fig.add_axes(rect_scatter)

        # ticks style
        ax.grid(linestyle=":")
        ax.tick_params(
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
        )
        ax.tick_params(
            which="minor",
            labelleft=False,
        )
        # axe labels
        ax.set_xlabel(r"Moment $q$")
        if temp == 1:
            ax.set_ylabel(r"$\alpha(q)$")
        elif temp == 0:
            ax.set_ylabel(r"$\beta(q)$")
        ax.set_ylim(ymin=-0.1, ymax=2)
        ax.set_xlim(xmin=0, xmax=3.5)

        # find the pre existing ticks
        yticks = [0, 0.5, 1, 1.5, 2]
        ytick_labels = ["{:.1f}".format(ytick) for ytick in yticks]
        # set major ticks
        ax.yaxis.set_ticks(yticks)
        ax.yaxis.set_ticklabels(ytick_labels)

        # find the pre existing ticks
        xticks = [1, 2, 3]
        xtick_labels = ["{:d}".format(xtick) for xtick in xticks]
        # set major ticks
        ax.xaxis.set_ticks(xticks)
        ax.xaxis.set_ticklabels(xtick_labels)

        q_array1 = np.arange(0.1, q + 0.6, 0.1)
        q_array2 = np.arange(0.1, q + 0.1, 0.1)
        # loop over
        for k in range(param.shape[1]):
            # plots for the means on all data
            ax.plot(
                q_array1,
                self.structure_function(
                    q_array1, param[0, k], param[1, k], param[2, k]
                ),
                ":",
                color=colors_plot[k],
                label="({:.2f}, {:.2f}, {:.2f})".format(
                    param[0, k], param[1, k], param[2, k]
                ),
            )
            ax.plot(
                q_array2, coeff[:, k], ".", color=colors_plot[k], markersize=5
            )

        legend1 = ax.legend(
            loc=2,
            fontsize="x-small",
            frameon=False,
            labelcolor=colors_plot,
            handlelength=0,
            handletextpad=0,
            markerscale=0,
            title=r"$(\nu,C_1,H)$",
        )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labels=names_plot,
            fontsize="x-small",
            frameon=False,
            labelcolor=colors_plot,
            handlelength=0,
            handletextpad=0,
            markerscale=0,
            ncol=1,
        )
        plt.gca().add_artist(legend1)

        if save:
            fig.savefig(
                "images/multifractal{}".format(self.resolution)
                + fig_name_supp
                + ".{}".format(self.fig_type),
            )
            fig.savefig(
                "images/multifractal{}".format(self.resolution)
                + fig_name_supp
                + ".pdf",
            )
