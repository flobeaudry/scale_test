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
                norm=colors.Normalize(vmin=0, vmax=10),
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
        elif self.datatype in ["dedt", "shear", "divergence"]:
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
                + self.fig_type
            )

    def _encircle(self, x, y, ax, **kw):
        p = np.c_[x, y]
        hull = ConvexHull(p)
        for simplex in hull.simplices:
            ax.plot(
                p[simplex, 0],
                p[simplex, 1],
                "k-",
                transform=ccrs.PlateCarree(),
            )

        poly = plt.Polygon(
            p[hull.vertices, :], transform=ccrs.PlateCarree(), **kw
        )
        ax.add_patch(poly)

    def arctic_plot_RGPS(
        self, data: np.ndarray, datatype: str, fig_name_supp: str = "_",
    ):
        x0 = np.arange(data.shape[0] + 1) * 12.5 - 2300
        y0 = np.arange(data.shape[0] + 1) * 12.5 - 1000

        x1 = np.arange(data.shape[0]) * 12.5 - 2300
        y1 = np.arange(data.shape[0]) * 12.5 - 1000

        x = np.broadcast_to(x1, (len(y1), len(x1)))
        y = np.broadcast_to(y1, (len(x1), len(y1))).T

        lon, lat = self._coordinates(x0, y0, RGPS=True)
        lon1, lat1 = self._coordinates(x1, y1, RGPS=True)

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
        # for deformation rates
        cf = ax.pcolormesh(
            lon,
            lat,
            data,
            # np.where(self.load(datatype="A") > 0.15, formated_data, np.NaN),
            cmap=cmocean.cm.amp,
            norm=colors.Normalize(vmin=0, vmax=1),
            transform=ccrs.PlateCarree(),
            zorder=1,
        )

        indices = np.where(data == 1)
        self._encircle(lon1[indices], lat1[indices], ax=ax, ec="k", fc="none")

        cbar = fig.colorbar(cf)
        # cbar.ax.set_ylabel("[day$^{-1}$]", rotation=-90, va="bottom")

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
                + self.fig_type
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
        mean_def = np.empty(len(scales))
        mean_scale = np.empty(len(scales))
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
            mean_def[k] = np.average(deformation[k][indices])
            mean_scale[k] = np.average(scaling[k][indices])

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

        return mean_def, mean_scale

    def _multiplot_precond(self):
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
        ax.set_xlabel("Spatial scale [km]")
        ax.set_ylabel(r"$\langle\dot\varepsilon_{tot}\rangle$ [day$^{-1}$]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(ymin=1e-2, ymax=1e-1)
        ax.set_xlim(xmin=6, xmax=1e3)
        # ax.set_title("H = {:.2f}, correlation = {:.2f}".format(coefficients[0], corr))

        return fig, ax

    def multi_plot(
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
        fig, ax = self._multiplot_precond()
        colors = np.array(["xkcd:dark blue grey", "xkcd:tomato", "xkcd:blush"])
        shape = np.array(["^", "v"])
        dam = np.array(["No damage: ", "Damage: ", "RGPS: "])
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
                + "H = {:.2f}, corr = {:.2f}".format(coefficients[0], corr),
                markersize=5,
            )
            ax.plot(t, np.exp(fit(np.log(t))), color=colors[k])

        ax.legend(loc=1, fontsize="x-small")
        if save:
            fig.savefig(
                "images/ssm{}".format(self.resolution)
                + fig_name_supp
                + ".{}".format(self.fig_type)
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
                + ".{}".format(self.fig_type)
            )

    def pdf_plot_vect(self, deformation: np.ndarray, scales):
        """
        Function that computes the PDF plot with the MLE fit. Same as above but for vectorized output.

        Args:
            deformation (np.ndarray): Data from box data.
            scales (list): scales of interests
        """
        data = self._clean_vect(deformation, scales)

        # get correct data from box data
        n = np.logspace(np.log10(5e-3), 0, num=50)
        p, x = np.histogram(data, bins=n, density=1)
        p_size, x_size = np.histogram(data, bins=n)

        # convert bin edges to centers
        x = (x[:-1] + x[1:]) / 2

        # compute best estimator
        (
            dedt_min,
            ks_dist,
            best_fit,
            min_index,
            coefficient,
        ) = self.ks_distance_minimizer(x, p)
        print(coefficient)
        t = np.linspace(dedt_min, x[-1], 10)
        alpha, sigma = self.mle_exponent(
            data[-np.sum(p_size[min_index:]) :], dedt_min
        )

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
                + ".{}".format(self.fig_type)
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
                "images/cdf{}.{}".format(self.resolution, self.fig_type)
            )

