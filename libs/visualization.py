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


from cProfile import label
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
from scipy.spatial import ConvexHull
from descartes import PolygonPatch
import alphashape


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
        super(Arctic, self).__init__(
            directory, time, expno, datatype, tolerance, resolution, nx, ny
        )

    def arctic_plot(self, formated_data: np.ndarray):
        """
        Plots passed data on the Arctic circle, and can save the image to images/ directory.

        Args:
            formated_data (np.ndarray): data to plot (ny, nx)

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

        # figure initialization
        fig = plt.figure(dpi=300)
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
        ax.set_extent([-180, 180, 65, 90], ccrs.PlateCarree())

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
            cbar = fig.colorbar(cf)
            cbar.ax.set_ylabel(self.name, rotation=-90, va="bottom")

        # for thickness
        elif self.datatype == "h":
            cf = ax.pcolormesh(
                lon,
                lat,
                formated_data,
                cmap=cmocean.cm.dense,
                norm=colors.Normalize(vmin=0, vmax=5),
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
            cbar = fig.colorbar(cf)
            cbar.ax.set_ylabel(self.name, rotation=-90, va="bottom")

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
            cbar = fig.colorbar(cf)
            cbar.ax.set_ylabel(self.name, rotation=-90, va="bottom")

        # for deformation rates
        elif self.datatype in ["dedt", "shear"]:
            ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
            cf = ax.pcolormesh(
                lon,
                lat,
                formated_data,
                # np.where(self.load(datatype="A") > 0.15, formated_data, np.NaN),
                cmap=cmocean.cm.amp,
                norm=colors.Normalize(vmin=0, vmax=0.1),
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
            cbar = fig.colorbar(cf)
            cbar.ax.set_ylabel(self.name, rotation=-90, va="bottom")

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
            cbar = fig.colorbar(cf)
            cbar.ax.set_ylabel(self.name, rotation=-90, va="bottom")

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
            cbar = fig.colorbar(cf)
            cbar.ax.set_ylabel(self.name, rotation=-90, va="bottom")

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
            cbar = fig.colorbar(cf)
            cbar.ax.set_ylabel(self.name, rotation=-90, va="bottom")

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
            cbar = fig.colorbar(cf)
            cbar.ax.set_ylabel(self.name, rotation=-90, va="bottom")

        else:
            raise SystemExit("Something is wrong with your data type...")

        ax.gridlines(zorder=2)
        ax.add_feature(cfeature.LAND, zorder=3)
        ax.coastlines(resolution="50m", zorder=4)

        if self.save:
            fig.savefig(
                "images/"
                + self.datatype
                + str(self.resolution)
                + self.fig_name_supp
                + "."
                + self.fig_type,
                transparent=0,
            )

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
        self,
        data: np.ndarray,
        datatype: str,
        fig_name_supp: str = "_",
        mask: bool = 0,
    ):
        """
        Function that plots data over the Arctic same as the other one but specifically for RGPS.

        Args:
            data (np.ndarray): data to plot in 2D
            datatype (str): type of the data
            fig_name_supp (str, optional): supplementary figure description in the name when saving the figure. Defaults to "_".
            mask (bool, optional): whether to plot the mask polygon or not on top of the data. Defaults to 0.
        """
        x0 = np.arange(data.shape[0] + 1) * RES_RGPS - 2300
        y0 = np.arange(data.shape[1] + 1) * RES_RGPS - 1000

        lon, lat = self._coordinates(x0, y0, RGPS=True)

        # figure initialization
        fig = plt.figure(dpi=300)
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
            cbar = fig.colorbar(cf)
            cbar.ax.set_ylabel("[day$^{-1}$]", rotation=-90, va="bottom")

        elif datatype == "mask":
            ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
            cf = ax.pcolormesh(
                lon,
                lat,
                data,
                # np.where(self.load(datatype="A") > 0.15, formated_data, np.NaN),
                cmap=cmocean.cm.oxy,
                norm=colors.Normalize(vmin=0, vmax=1),
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
        else:
            ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
            cf = ax.pcolormesh(
                lon,
                lat,
                data,
                # np.where(self.load(datatype="A") > 0.15, formated_data, np.NaN),
                cmap=cmocean.cm.amp,
                norm=colors.Normalize(vmin=0, vmax=0.1),
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
            cbar = fig.colorbar(cf)
            cbar.ax.set_ylabel("[day$^{-1}$]", rotation=-90, va="bottom")

        if mask:
            x1 = np.arange(data.shape[0]) * RES_RGPS - 2300
            y1 = np.arange(data.shape[0]) * RES_RGPS - 1000
            lon1, lat1 = self._coordinates(x1, y1, RGPS=True)
            indices = np.where(data == 1)
            self._encircle(lon1[indices], lat1[indices], ax=ax)

        ax.gridlines(zorder=2)
        ax.add_feature(cfeature.LAND, zorder=3)
        ax.coastlines(resolution="50m", zorder=4)

        if self.save:
            fig.savefig(
                "images/"
                + datatype
                + str(self.resolution)
                + fig_name_supp
                + "RGPS"
                + "."
                + self.fig_type,
                transparent=0,
            )

    def scale_plot(
        self,
        deformation: np.ndarray,
        scales: list,
        viscosity: np.ndarray = None,
    ):
        """
        This function plots the spatial scale and computes the exponent of the scaling <dedt> ~ L^-H by doing a linear regression.

        Args:
            deformation (np.ndarray): array of the data inside each box. Shape (nL, nBox + (nx*ny-nBox)*NaN, 2). The first index is the studied scaling, the second is the number of box in which is has been computed, the third is deformation rate in 0 and effective scale in 1.

            scales (list): list of all scales under study.

            viscosity (np.ndarray): if we want to plot colors for the viscosity norm. Give the data array.
        """

        fig = plt.figure(dpi=300, figsize=(6, 4))

        # initialization of the list containing the means
        mean_def = np.empty(len(scales))
        mean_scale = np.empty(len(scales))
        mean_def_cut = np.empty(len(scales))
        mean_scale_cut = np.empty(len(scales))

        # definitions for the axes
        left, width = 0.14, 0.53
        bottom, height = 0.12, 0.8
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        ax = fig.add_axes(rect_scatter)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)

        # now determine nice limits by hand for the histogram
        ymax = np.nanmax(np.abs(deformation[..., 0]))
        ymin = np.nanmin(np.abs(deformation[..., 0]))
        n = np.logspace(np.log10(ymin), np.log10(ymax), num=50)
        ax_histy.hist(
            deformation[..., 0].flatten(),
            bins=n,
            orientation="horizontal",
            color="xkcd:dark blue grey",
        )

        # loop over the scales
        for k in range(len(scales)):
            # compute the means, ignoring NaNs
            indices = ~np.isnan(viscosity[k])
            mean_def[k] = np.average(deformation[k, indices, 0],)
            mean_scale[k] = np.average(deformation[k, indices, 1],)
            # colormap
            base = cm.get_cmap("cmo.haline", 256)
            newcolors = base(np.linspace(0, 1, 256))
            bot = np.array([100 / 256, 20 / 256, 20 / 256, 1])
            newcolors[:51, :] = bot
            newcmp = ListedColormap(newcolors)
            # plot
            cf = ax.scatter(
                deformation[k, indices, 1],
                deformation[k, indices, 0],
                c=viscosity[k, indices],
                s=0.5,
                cmap=newcmp,
                norm=colors.Normalize(vmin=0, vmax=5 * ETA_MAX * E ** 2),
            )
            # same thing with only viscosities that are under visc_max (plastic def)
            viscosity[k, viscosity[k] >= ETA_MAX * E ** 2] = np.NaN
            indices = ~np.isnan(viscosity[k])
            mean_def_cut[k] = np.average(deformation[k, indices, 0],)
            mean_scale_cut[k] = np.average(deformation[k, indices, 1],)

        # add color bar
        cbar = fig.colorbar(cf)
        cbar.set_label(
            "Bulk viscosity [N$\cdot$s$\cdot$m$^{-1}$]",
            rotation=-90,
            va="bottom",
        )
        # add red line for zeta max
        cax = cbar.ax
        cax.hlines(
            ETA_MAX * E ** 2,
            0,
            ETA_MAX * E ** 2 * 10,
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
        coefficients_cut = np.polyfit(
            np.log(mean_scale_cut), np.log(mean_def_cut), 1
        )
        fit_cut = np.poly1d(coefficients_cut)
        t_cut = np.linspace(mean_scale_cut[0], mean_scale_cut[-1], 10)

        # correlation
        corr, _ = pearsonr(mean_scale, mean_def)
        corr_cut, _ = pearsonr(mean_scale_cut, mean_def_cut)

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
        ax.plot(
            mean_scale_cut,
            mean_def_cut,
            "v",
            color="xkcd:golden rod",
            label="H = {:.2f}, corr = {:.2f}".format(
                coefficients_cut[0], corr_cut
            ),
            markersize=5,
        )
        ax.plot(t_cut, np.exp(fit_cut(np.log(t_cut))), color="xkcd:golden rod")
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
        # ax.set_title("H = {:.2f}, correlation = {:.2f}".format(coefficients[0], corr))
        if self.save:
            fig.savefig(
                "images/spatial_scale{}".format(self.resolution)
                + self.fig_name_supp
                + ".{}".format(self.fig_type)
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
                norm=colors.Normalize(vmin=0, vmax=5 * ETA_MAX * E ** 2),
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
            ETA_MAX * E ** 2,
            0,
            ETA_MAX * E ** 2 * 10,
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

        return mean_def, mean_scale, coefficients[0]

    def _multiplot_precond(self, type: bool):
        """
        Function that contains all the preconditions for the multiplot figure.

        Returns:
            fig: the figure
            ax: the axis in the figure
        """
        fig = plt.figure(dpi=300, figsize=(6, 4))

        # definitions for the axes
        left, width = 0.14, 0.73
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
            which="minor", labelleft=False,
        )
        # axe labels
        if type == 0:  # space
            ax.set_xlabel("Spatial scale [km]")
            ax.set_ylabel(
                r"$\langle\dot\varepsilon_{tot}\rangle$ [day$^{-1}$]"
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(ymin=4e-3, ymax=2e-1)
            ax.set_xlim(xmin=6, xmax=8e2)
            # ax.set_title("H = {:.2f}, correlation = {:.2f}".format(coefficients[0], corr))
        if type == 1:  # time
            ax.set_xlabel("Temporal scale [km]")
            ax.set_ylabel(
                r"$\langle\dot\varepsilon_{tot}\rangle$ [day$^{-1}$]"
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(ymin=4e-3, ymax=2e-1)
            ax.set_xlim(xmin=2, xmax=4e1)
            # ax.set_title("H = {:.2f}, correlation = {:.2f}".format(coefficients[0], corr))

        return fig, ax

    def multi_plot_spatial(
        self,
        mean_def: np.ndarray,
        mean_scale: np.ndarray,
        fig_name_supp: str,
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
        colors = np.array(
            [
                "xkcd:dark blue grey",
                "xkcd:tomato",
                # "xkcd:blush",
                "xkcd:gross green",
            ]
        )
        shape = np.array(["^", "v"])
        dam = np.array(
            [
                "No damage: ",
                "Damage: ",
                # "Advection + healing: ",
                "RGPS: ",
            ]
        )
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
                "^",
                color=colors[k],
                label=dam[k]
                + r"$\beta$ = {:.2f}, corr = {:.2f}".format(
                    np.abs(coefficients[0]), corr
                ),
                markersize=5,
            )
            ax.plot(t, np.exp(fit(np.log(t))), color=colors[k])

        ax.legend(loc=1, fontsize="x-small")
        if save:
            fig.savefig(
                "images/ssm{}".format(self.resolution)
                + fig_name_supp
                + ".{}".format(self.fig_type),
                transparent=0,
            )

    def multi_plot_temporal(
        self,
        mean_def: np.ndarray,
        mean_scale: np.ndarray,
        fig_name_supp: str,
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
        colors = np.array(
            [
                "xkcd:dark blue grey",
                "xkcd:tomato",
                # "xkcd:blush",
                "xkcd:gross green",
            ]
        )
        shape = np.array(["^", "v"])
        dam = np.array(
            [
                "No damage: ",
                "Damage: ",
                # "Advection + healing: ",
                "RGPS: ",
            ]
        )
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
                "^",
                color=colors[k],
                label=dam[k]
                + r"$\alpha$ = {:.2f}, corr = {:.2f}".format(
                    np.abs(coefficients[0]), corr
                ),
                markersize=5,
            )
            ax.plot(t, np.exp(fit(np.log(t))), color=colors[k])

        ax.legend(loc=1, fontsize="x-small")
        if save:
            fig.savefig(
                "images/ssmT{}".format(self.resolution)
                + fig_name_supp
                + ".{}".format(self.fig_type),
                transparent=0,
            )

    def pdf_plot(self, data: np.ndarray):
        """
        Function that computes the PDF plot with the MLE fit.

        Args:
            data (np.ndarray): Data from box data.
        """
        data = self._clean(data)

        # get correct data from box data
        n = np.logspace(np.log10(5e-3), 0, num=50)
        p, x = np.histogram(data, bins=n, density=1)

        # convert bin edges to centers
        x = (x[:-1] + x[1:]) / 2

        # compute best estimator
        dedt_min, ks_dist, best_fit, min_index = self.ks_distance_minimizer(
            x, p
        )
        t = np.linspace(dedt_min, x[-1], 10)
        alpha, sigma = self.mle_exponent(x[min_index:], dedt_min)

        # plots
        fig = plt.figure(dpi=300, figsize=(6, 4))
        ax = plt.subplot()
        ax.plot(x, p, "o", color="black", markersize=3)
        ax.plot(t, np.exp(best_fit(np.log(t))), "--", color="red")
        # ticks
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
        # axe labels
        ax.set_xlabel("Total deformation rate [day$^{-1}$]")
        ax.set_ylabel("PDF")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            r"$\hat\alpha$ = {:.1f}$\pm${:.1f}, KS-distance = {:.2f}".format(
                alpha, sigma, ks_dist
            )
        )
        if self.save:
            fig.savefig(
                "images/pdf{}".format(self.resolution)
                + self.fig_name_supp
                + ".{}".format(self.fig_type),
                transparent=0,
            )

    def pdf_du(
        self, du_stack: list, save: bool, fig_name_supp: string,
    ):
        """
        It simply computes everything for pdf plot.

        Args:
            du_stack (list): velocity derivatives of each model on last axis. we need list because RGPS is not same shape...
            save (bool): save or not the fig
            fig_name_supp (string): supplmentary info for fig name.

        Returns:
            [type]: figure of pdf
        """
        # init plot
        fig = plt.figure(dpi=300, figsize=(8, 4))

        # definitions for the axis
        left_shear, width_shear = (1 - 2 * 0.4) / 3 + 0.02, 0.4
        bottom_shear, height_shear = 0.12, 0.8
        rect_scatter_shear = [
            left_shear,
            bottom_shear,
            width_shear,
            height_shear,
        ]

        left_div, width_div = 2 * (1 - 2 * 0.4) / 3 + 0.42, 0.4
        bottom_div, height_div = 0.12, 0.8
        rect_scatter_div = [
            left_div,
            bottom_div,
            width_div,
            height_div,
        ]

        ax_shear = fig.add_axes(rect_scatter_shear)
        ax_div = fig.add_axes(rect_scatter_div)

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
        ax_div.grid(
            axis="x", which="minor", linestyle=":", color="xkcd:light gray"
        )
        ax_div.grid(
            axis="x", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_div.grid(
            axis="y", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_div.tick_params(
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
            labelleft=True,
        )

        colors = np.array(
            [
                "xkcd:dark blue grey",
                "xkcd:tomato",
                # "xkcd:blush",
                "xkcd:gross green",
            ]
        )
        dam = np.array(
            [
                "No damage: ",
                "Damage: ",
                # "Advection + healing: ",
                "RGPS: ",
            ]
        )

        for k in range(len(du_stack)):
            shear = self._deformation(du_stack[k], 1)
            div = np.abs(self._deformation(du_stack[k], 2))

            shear_cut = shear[~np.isnan(shear)]
            div_cut = div[~np.isnan(div)]

            # get correct data from box data
            n = np.logspace(np.log10(5e-3), 0, num=20)
            p_shear, x_shear = np.histogram(shear_cut, bins=n, density=1)
            p_div, x_div = np.histogram(div_cut, bins=n, density=1)
            p_shear = np.where(p_shear == 0.0, np.NaN, p_shear)
            p_div = np.where(p_div == 0.0, np.NaN, p_div)

            # convert bin edges to centers
            x_shear_mid = (x_shear[:-1] + x_shear[1:]) / 2
            x_div_mid = (x_div[:-1] + x_div[1:]) / 2

            # variables for fit definitions
            indices_shear = []
            indices_div = []
            for i in range(p_shear.shape[0]):
                if np.isnan(p_shear[i]):
                    indices_shear.append(i)
            for i in range(p_div.shape[0]):
                if np.isnan(p_div[i]):
                    indices_div.append(i)

            if len(indices_shear) == 0 and len(indices_div) == 0:
                t_shear = x_shear_mid
                t_div = x_div_mid
                pt_shear = p_shear
                pt_div = p_div
            elif len(indices_shear) == 0 and len(indices_div) != 0:
                t_shear = x_shear_mid
                t_div = np.delete(x_div_mid, np.asarray(indices_div))
                pt_shear = p_shear
                pt_div = np.delete(p_div, np.asarray(indices_div))
            elif len(indices_shear) != 0 and len(indices_div) == 0:
                t_shear = np.delete(x_shear_mid, np.asarray(indices_shear))
                t_div = x_div_mid
                pt_shear = np.delete(p_shear, np.asarray(indices_shear))
                pt_div = p_div
            elif len(indices_shear) != 0 and len(indices_div) != 0:
                t_shear = np.delete(x_shear_mid, np.asarray(indices_shear))
                t_div = np.delete(x_div_mid, np.asarray(indices_div))
                pt_shear = np.delete(p_shear, np.asarray(indices_shear))
                pt_div = np.delete(p_div, np.asarray(indices_div))

            # fit
            coeff_shear = (
                np.polynomial.Polynomial.fit(
                    np.log10(t_shear[-9:]), np.log10(pt_shear[-9:]), 1
                )
                .convert()
                .coef
            )
            coeff_div = (
                np.polynomial.Polynomial.fit(
                    np.log10(t_div[-9:]), np.log10(pt_div[-9:]), 1
                )
                .convert()
                .coef
            )
            best_fit_shear = np.polynomial.Polynomial(coeff_shear)
            best_fit_div = np.polynomial.Polynomial(coeff_div)

            # plots
            ax_shear.plot(
                x_shear_mid,
                p_shear,
                color=colors[k],
                label=dam[k] + "({:.1f})".format(-coeff_shear[-1]),
            )
            # ax_shear.plot(
            #     t_shear[-9:],
            #     10 ** (best_fit_shear(np.log10(t_shear[-9:]))),
            #     "-.",
            #     color="red",
            #     lw=0.7,
            # )
            ax_div.plot(
                x_div_mid,
                p_div,
                color=colors[k],
                label=dam[k] + "({:.1f})".format(-coeff_div[-1]),
            )
            # ax_div.plot(
            #     t_div[-9:],
            #     10 ** (best_fit_div(np.log10(t_div[-9:]))),
            #     "-.",
            #     color="red",
            #     lw=0.7,
            # )

        # axis labels
        ax_shear.set_xlabel("Shear rate [day$^{-1}$]")
        ax_shear.set_ylabel("PDF")
        ax_shear.set_xscale("log")
        ax_shear.set_yscale("log")
        ax_shear.set_ylim(ymin=1e-4, ymax=1e3)
        ax_shear.set_xlim(xmin=5e-3, xmax=1.5)
        ax_shear.locator_params(axis="y", numticks=5)

        ax_div.set_xlabel("Absolute divergence rate [day$^{-1}$]")
        ax_div.set_xscale("log")
        ax_div.set_yscale("log")
        ax_div.set_ylim(ymin=1e-4, ymax=1e3)
        ax_div.set_xlim(xmin=5e-3, xmax=1.5)
        ax_div.locator_params(axis="y", numticks=5)

        ax_shear.legend(loc=1, fontsize="x-small")
        ax_div.legend(loc=1, fontsize="x-small")
        # save fig
        if save:
            fig.savefig(
                "images/pdfm{}".format(self.resolution)
                + fig_name_supp
                + ".{}".format(self.fig_type),
                transparent=0,
            )

    def cdf_du(
        self, du_stack: list, save: bool, fig_name_supp: string,
    ):
        """
        It simply computes everything for pdf plot.

        Args:
            du_stack (list): velocity derivatives of each model on last axis. we need list because RGPS is not same shape... RGPS NEEDS TO BE THE LAST ITEM OF THIS LIST.
            save (bool): save or not the fig
            fig_name_supp (string): supplmentary info for fig name.

        Returns:
            [type]: figure of pdf
        """
        # init plot
        fig = plt.figure(dpi=300, figsize=(8, 4))

        # definitions for the axis
        left_shear, width_shear = (1 - 2 * 0.4) / 3 + 0.02, 0.4
        bottom_shear, height_shear = 0.12, 0.8
        rect_scatter_shear = [
            left_shear,
            bottom_shear,
            width_shear,
            height_shear,
        ]

        left_div, width_div = 2 * (1 - 2 * 0.4) / 3 + 0.42, 0.4
        bottom_div, height_div = 0.12, 0.8
        rect_scatter_div = [
            left_div,
            bottom_div,
            width_div,
            height_div,
        ]

        ax_shear = fig.add_axes(rect_scatter_shear)
        ax_div = fig.add_axes(rect_scatter_div)

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
        ax_div.grid(
            axis="x", which="minor", linestyle=":", color="xkcd:light gray"
        )
        ax_div.grid(
            axis="x", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_div.grid(
            axis="y", which="major", linestyle="-", color="xkcd:light gray"
        )
        ax_div.tick_params(
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
            labelleft=True,
        )

        colors = np.array(
            [
                "xkcd:dark blue grey",
                "xkcd:tomato",
                # "xkcd:blush",
                "xkcd:gross green",
            ]
        )
        dam = np.array(
            [
                "No damage: ",
                "Damage: ",
                # "Advection + healing: ",
                "RGPS: ",
            ]
        )

        shear_RGPS = self._deformation(du_stack[-1], 1)
        div_RGPS = np.abs(self._deformation(du_stack[-1], 2))

        shear_RGPS_cut = shear_RGPS[~np.isnan(shear_RGPS)]
        div_RGPS_cut = div_RGPS[~np.isnan(div_RGPS)]

        # CDF, we have to cut under 0.005 for nice CDF
        shear_RGPS_cut1 = np.where(
            shear_RGPS_cut < 0.005, np.NaN, shear_RGPS_cut
        )
        div_RGPS_cut1 = np.where(div_RGPS_cut < 0.005, np.NaN, div_RGPS_cut)
        shear_RGPS_cut2 = shear_RGPS_cut1[~np.isnan(shear_RGPS_cut1)]
        div_RGPS_cut2 = div_RGPS_cut1[~np.isnan(div_RGPS_cut1)]

        n = np.logspace(np.log10(5e-3), 0, num=1000)

        p_RGPS_shear, x_RGPS_shear = np.histogram(
            shear_RGPS_cut2, bins=n, density=1
        )
        p_RGPS_div, x_RGPS_div = np.histogram(div_RGPS_cut2, bins=n, density=1)

        dx_RGPS_shear = x_RGPS_shear[1:] - x_RGPS_shear[:-1]
        X1_RGPS_shear = (x_RGPS_shear[1:] + x_RGPS_shear[:-1]) / 2
        F1_RGPS_shear = np.cumsum(p_RGPS_shear * dx_RGPS_shear)

        dx_RGPS_div = x_RGPS_div[1:] - x_RGPS_div[:-1]
        X1_RGPS_div = (x_RGPS_div[1:] + x_RGPS_div[:-1]) / 2
        F1_RGPS_div = np.cumsum(p_RGPS_div * dx_RGPS_div)

        # N_RGPS_shear = shear_RGPS_cut2.flatten().shape[0]
        # X_RGPS_shear = np.sort(shear_RGPS_cut2.flatten())
        # F_RGPS_shear = np.array(range(N_RGPS_shear)) / float(N_RGPS_shear - 1)

        # N_RGPS_div = div_RGPS_cut2.flatten().shape[0]
        # X_RGPS_div = np.sort(div_RGPS_cut2.flatten())
        # F_RGPS_div = np.array(range(N_RGPS_div)) / float(N_RGPS_div - 1)

        for k in range(len(du_stack) - 1):
            shear = self._deformation(du_stack[k], 1)
            div = np.abs(self._deformation(du_stack[k], 2))

            shear_cut = shear[~np.isnan(shear)]
            div_cut = div[~np.isnan(div)]

            # CDF, we have to cut under 0.005 for nice CDF
            shear_cut1 = np.where(shear_cut < 0.005, np.NaN, shear_cut)
            div_cut1 = np.where(div_cut < 0.005, np.NaN, div_cut)
            shear_cut2 = shear_cut1[~np.isnan(shear_cut1)]
            div_cut2 = div_cut1[~np.isnan(div_cut1)]

            # N_shear = shear_cut2.flatten().shape[0]
            # X_shear = np.sort(shear_cut2.flatten())
            # F_shear = np.array(range(N_shear)) / float(N_shear - 1)

            # N_div = div_cut2.flatten().shape[0]
            # X_div = np.sort(div_cut2.flatten())
            # F_div = np.array(range(N_div)) / float(N_div - 1)

            p_shear, x_shear = np.histogram(shear_cut2, bins=n, density=1)
            p_div, x_div = np.histogram(div_cut2, bins=n, density=1)

            dx_shear = x_shear[1:] - x_shear[:-1]
            X1_shear = (x_shear[1:] + x_shear[:-1]) / 2
            F1_shear = np.cumsum(p_shear * dx_shear)

            dx_div = x_div[1:] - x_div[:-1]
            X1_div = (x_div[1:] + x_div[:-1]) / 2
            F1_div = np.cumsum(p_div * dx_div)

            ks_distance_shear = np.amax(np.abs(F1_shear - F1_RGPS_shear))
            ks_distance_div = np.amax(np.abs(F1_div - F1_RGPS_div))

            # plots
            ax_shear.plot(
                X1_shear,
                F1_shear,
                color=colors[k],
                label=dam[k] + "({:.2f})".format(ks_distance_shear),
            )

            ax_div.plot(
                X1_div,
                F1_div,
                color=colors[k],
                label=dam[k] + "({:.2f})".format(ks_distance_div),
            )

        # RGPS plots
        ax_shear.plot(
            X1_RGPS_shear, F1_RGPS_shear, color=colors[-1], label=dam[-1],
        )
        ax_div.plot(
            X1_RGPS_div, F1_RGPS_div, color=colors[-1], label=dam[-1],
        )

        # axis labels
        ax_shear.set_xlabel("Shear rate [day$^{-1}$]")
        ax_shear.set_ylabel("CDF")
        ax_shear.set_xscale("log")
        ax_shear.set_ylim(ymin=0, ymax=1.05)
        ax_shear.set_xlim(xmin=5e-3, xmax=1.5)

        ax_div.set_xlabel("Absolute divergence rate [day$^{-1}$]")
        ax_div.set_xscale("log")
        ax_div.set_ylim(ymin=0, ymax=1.05)
        ax_div.set_xlim(xmin=5e-3, xmax=1.5)

        ax_shear.legend(loc=4, fontsize="x-small")
        ax_div.legend(loc=4, fontsize="x-small")
        # save fig
        if save:
            fig.savefig(
                "images/cdfm{}".format(self.resolution)
                + fig_name_supp
                + ".{}".format(self.fig_type),
                transparent=0,
            )

    def pdf_plot_vect(
        self, shear: np.ndarray, ndiv: np.ndarray, pdiv: np.ndarray
    ):
        """
        Function that computes the PDF plot with the MLE fit. Same as above but for vectorized output.

        Args:
            deformation (np.ndarray): Data from box data.
            scales (list): scales of interests
        """
        # init plot
        fig = plt.figure(dpi=300, figsize=(12, 4))
        (
            x_shear,
            p_shear,
            ks_dist_shear,
            best_fit_shear,
            min_index_shear,
            coefficient_shear,
            dedt_min_shear,
            alpha_shear,
            sigma_shear,
        ) = self._pdf_interior(shear, 1)

        (
            x_ndiv,
            p_ndiv,
            ks_dist_ndiv,
            best_fit_ndiv,
            min_index_ndiv,
            coefficient_ndiv,
            dedt_min_ndiv,
            alpha_ndiv,
            sigma_ndiv,
        ) = self._pdf_interior(ndiv, 3)

        (
            x_pdiv,
            p_pdiv,
            ks_dist_pdiv,
            best_fit_pdiv,
            min_index_pdiv,
            coefficient_pdiv,
            dedt_min_pdiv,
            alpha_pdiv,
            sigma_pdiv,
        ) = self._pdf_interior(pdiv, 3)

        # fit definitions
        t_shear = np.logspace(np.log10(dedt_min_shear), np.log10(2), 10)
        t_ndiv = np.logspace(np.log10(dedt_min_ndiv), np.log10(0.3), 10)
        t_pdiv = np.logspace(np.log10(dedt_min_pdiv), np.log10(0.3), 10)

        # definitions for the axis
        left_shear, width_shear = (1 - 3 * 0.267) / 4, 0.267
        bottom_shear, height_shear = 0.12, 0.8
        rect_scatter_shear = [
            left_shear,
            bottom_shear,
            width_shear,
            height_shear,
        ]

        left_ndiv, width_ndiv = (1 - 3 * 0.267) / 2 + 0.267, 0.267
        bottom_ndiv, height_ndiv = 0.12, 0.8
        rect_scatter_ndiv = [
            left_ndiv,
            bottom_ndiv,
            width_ndiv,
            height_ndiv,
        ]

        left_pdiv, width_pdiv = 3 * (1 - 3 * 0.267) / 4 + 2 * 0.267, 0.267
        bottom_pdiv, height_pdiv = 0.12, 0.8
        rect_scatter_pdiv = [
            left_pdiv,
            bottom_pdiv,
            width_pdiv,
            height_pdiv,
        ]

        ax_shear = fig.add_axes(rect_scatter_shear)
        ax_ndiv = fig.add_axes(rect_scatter_ndiv)
        ax_pdiv = fig.add_axes(rect_scatter_pdiv)

        # plots
        ax_shear.plot(x_shear, p_shear, ".", color="black", markersize=2)
        ax_shear.plot(x_shear, p_shear, color="black", lw=0.7)
        ax_shear.plot(
            t_shear,
            10 ** (best_fit_shear(np.log10(t_shear))),
            "-.",
            color="red",
            lw=0.7,
        )
        ax_ndiv.plot(x_ndiv, p_ndiv, ".", color="black", markersize=2)
        ax_ndiv.plot(x_ndiv, p_ndiv, color="black", lw=0.7)
        ax_ndiv.plot(
            t_ndiv,
            10 ** (best_fit_ndiv(np.log10(t_ndiv))),
            "-.",
            color="red",
            lw=0.7,
        )
        ax_pdiv.plot(x_pdiv, p_pdiv, ".", color="black", markersize=2)
        ax_pdiv.plot(x_pdiv, p_pdiv, color="black", lw=0.7)
        ax_pdiv.plot(
            t_pdiv,
            10 ** (best_fit_pdiv(np.log10(t_pdiv))),
            "-.",
            color="red",
            lw=0.7,
        )
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
        # axis labels
        ax_shear.set_xlabel("Shear rate [day$^{-1}$]")
        ax_shear.set_ylabel("PDF")
        ax_shear.set_xscale("log")
        ax_shear.set_yscale("log")
        ax_shear.set_ylim(ymin=1e-6, ymax=1e3)
        ax_shear.set_xlim(xmin=5e-3, xmax=10)
        ax_shear.locator_params(axis="y", numticks=5)
        ax_shear.set_title(
            r"$\hat\alpha$ = {:.2f}$\pm${:.1f}, KS-distance = {:.2f}".format(
                alpha_shear, sigma_shear, ks_dist_shear
            )
        )
        ax_ndiv.set_xlabel("Negative divergence rate [day$^{-1}$]")
        ax_ndiv.set_yscale("log")
        ax_ndiv.set_xscale("log")
        ax_ndiv.set_ylim(ymin=1e-6, ymax=1e3)
        ax_ndiv.set_xlim(xmin=5e-3, xmax=1e0)
        ax_ndiv.invert_xaxis()
        ax_ndiv.locator_params(axis="y", numticks=5)
        ax_ndiv.set_title(
            r"$\hat\alpha$ = {:.2f}$\pm${:.1f}, KS-distance = {:.2f}".format(
                alpha_ndiv, sigma_ndiv, ks_dist_ndiv
            )
        )
        ax_pdiv.set_xlabel("Positive divergence rate [day$^{-1}$]")
        ax_pdiv.set_xscale("log")
        ax_pdiv.set_yscale("log")
        ax_pdiv.set_ylim(ymin=1e-6, ymax=1e3)
        ax_pdiv.set_xlim(xmin=5e-3, xmax=1)
        ax_pdiv.locator_params(axis="y", numticks=5)
        ax_pdiv.set_title(
            r"$\hat\alpha$ = {:.2f}$\pm${:.1f}, KS-distance = {:.2f}".format(
                alpha_pdiv, sigma_pdiv, ks_dist_pdiv
            )
        )
        # save fig
        if self.save:
            fig.savefig(
                "images/pdf{}".format(self.resolution)
                + self.fig_name_supp
                + ".{}".format(self.fig_type),
                transparent=0,
            )

    def cdf_plot(self, data: np.ndarray):
        """
        Function that computes the CDF plot of data.

        Args:
            data (np.ndarray): Data from box data.
        """

        data = self._clean(data)

        # get correct data from box data
        n = np.logspace(np.log10(5e-3), 0, num=50)
        p, x = np.histogram(data, bins=n, density=1)

        # convert bin edges to centers
        x = (x[:-1] + x[1:]) / 2
        dedt_min, ks_dist, best_fit, min_index = self.ks_distance_minimizer(
            x, p
        )
        fit = np.exp(best_fit(np.log(data)))
        fit = self._clean(fit, 0)

        # compute CDF
        cdf_data, norm_data = self.cumul_dens_func(data)
        cdf_fit, norm_fit = self.cumul_dens_func(fit)

        # plots
        fig = plt.figure(dpi=300, figsize=(6, 4))
        ax = plt.subplot()
        ax.plot(cdf_data, norm_data, color="black")
        ax.plot(cdf_fit, norm_fit, color="red")
        # ticks
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
        # axe labels
        ax.set_xlabel("Total deformation rate [day$^{-1}$]")
        ax.set_ylabel("CDF")
        ax.set_xscale("log")
        if self.save:
            fig.savefig(
                "images/cdf{}.{}".format(self.resolution, self.fig_type),
                transparent=0,
            )

    def multifractal_spatial(
        self, q: int, deps: np.ndarray, scale: np.ndarray, RGPS: bool = False,
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
                    deps ** n,
                    scale,
                    L_RGPS,
                    save=0,
                    fig_name_supp="_dedtQ_02_RGPS_du",
                )
                coeff_list.append(coeff)
        else:
            for n in q_array:
                mean_depsq, mean_scale, coeff = self.scale_plot_vect(
                    deps ** n, scale, L10, save=0, fig_name_supp="_dedtQ_97",
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

        fig = plt.figure(dpi=300, figsize=(4, 4))

        # definitions for the axes
        left, width = 0.14, 0.73
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
            which="minor", labelleft=False,
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

        colors = np.array(
            [
                "xkcd:dark blue grey",
                "xkcd:tomato",
                # "xkcd:blush",
                "xkcd:gross green",
            ]
        )
        dam = np.array(
            [
                "No damage: ",
                "Damage: ",
                # "Advection + healing: ",
                "RGPS: ",
            ]
        )

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
                color=colors[k],
                label=dam[k]
                + "({:.2f}, {:.2f}, {:.2f})".format(
                    param[0, k], param[1, k], param[2, k]
                ),
            )
            ax.plot(q_array2, coeff[:, k], ".", color=colors[k], markersize=5)

        ax.legend(loc=1, fontsize="x-small")
        if save:
            fig.savefig(
                "images/multifractal{}".format(self.resolution)
                + fig_name_supp
                + ".{}".format(self.fig_type),
                transparent=0,
            )
