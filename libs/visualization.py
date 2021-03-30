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
#   -pdf plot with fit on it
#   -cdf plot
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

        lon, lat = self._coordinates(x0, y0)[0], self._coordinates(x0, y0)[1]

        # for the quiver variables that are not in the corner of the cell grid like pcolormesh, but they are rather in the center of the grid so we have to interpolate the grid points
        if self.datatype == "u":
            x1 = (x0[1:] + x0[:-1]) / 2
            y1 = (y0[1:] + y0[:-1]) / 2

            lon1, lat1 = (
                self._coordinates(x1, y1)[0],
                self._coordinates(x1, y1)[1],
            )

        # figure initialization
        fig = plt.figure(dpi=300)
        ax = plt.subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
        fig.subplots_adjust(bottom=0.05, top=0.95, left=0.04, right=0.95, wspace=0.02)

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
            cf = ax.pcolormesh(
                lon,
                lat,
                formated_data,
                cmap=cmocean.cm.amp,
                norm=colors.Normalize(vmin=0, vmax=0.1),
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
                formated_data[0 :: self.step, 0 :: self.step, 0]
                / self._velolicty_vector_magnitude(
                    formated_data[0 :: self.step, 0 :: self.step, 0],
                    formated_data[0 :: self.step, 0 :: self.step, 1],
                ),
                formated_data[0 :: self.step, 0 :: self.step, 1]
                / self._velolicty_vector_magnitude(
                    formated_data[0 :: self.step, 0 :: self.step, 0],
                    formated_data[0 :: self.step, 0 :: self.step, 1],
                ),
                color="black",
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
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
            raise SystemExit("\nSomething is wrong with your data type...")

        ax.gridlines(zorder=2)
        ax.add_feature(cfeature.LAND, zorder=3)
        ax.coastlines(resolution="50m", zorder=4)

        if self.save:
            fig.savefig("images/" + self.datatype + "." + self.fig_type)

    def scale_plot(self, deformation: np.ndarray, scales: list):
        """
        This function plots the spatial scale and computes the exponent of the scaling <dedt> ~ L^-H by doing a linear regression.

        Args:
            formated_data (np.ndarray): array of the data inside eache box. Shape (nL, nBox + (nx*ny-nBox)*NaN, 2). The first index is the studied scaling, the second is the number of box in which is has been computed, the third is deformation rate in 0 and effective scale in 1.

            scales (list): list of all scales under study.
        """
        fig = plt.figure(dpi=300, figsize=(6, 4))
        ax = plt.subplot()
        # initialization of the list containing the means
        mean_def = np.empty(len(scales))
        mean_scale = np.empty(len(scales))
        # loop over the scales
        for k in range(len(scales)):
            # compute the means, ignoring NaNs
            mean_def[k] = np.nanmean(deformation[k, :, 0])
            mean_scale[k] = np.nanmean(deformation[k, :, 1])
            ax.plot(
                deformation[k, :, 1],
                deformation[k, :, 0],
                ".",
                color="black",
                markersize=3,
            )
        # linear regression over the means
        coefficients = np.polyfit(np.log(mean_scale), np.log(mean_def), 1)
        polynomial = np.poly1d(coefficients)
        t = np.linspace(mean_scale[0], mean_scale[-1], 10)

        # correlation
        corr, _ = pearsonr(mean_scale, mean_def)

        # plots
        ax.plot(mean_scale, mean_def, "v", color="darkgray")
        ax.plot(t, np.exp(polynomial(np.log(t))), color="darkgray")
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
        ax.set_xlabel("Spatial scale [km]")
        ax.set_ylabel("Total deformation rate [day$^{-1}$]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("H = {:.2f}, correlation = {:.2f}".format(coefficients[0], corr))
        if self.save:
            fig.savefig(
                "images/spatial_scale{}.{}".format(self.resolution, self.fig_type)
            )

    def pdf_plot(self, data: np.ndarray):

        # get correct data from box data
        cdf_data, cdf_norm = self.cumul_dens_func(data)
        n = np.logspace(np.log10(5e-3), 0, num=50)
        p, x = np.histogram(cdf_data, bins=n, density=1)

        # convert bin edges to centers
        x = (x[:-1] + x[1:]) / 2

        # coefficients = np.polyfit(np.log10(x), np.log10(y), 1)
        # polynomial = np.poly1d(coefficients)
        # log10_y_fit = polynomial(np.log10(x))  # <-- Changed

        # plt.plot(x, y, "o-")
        # plt.plot(x, 10 ** log10_y_fit, "*-")  # <-- Changed
        # plt.yscale("log")
        # plt.xscale("log")

        dedt_min = 0.04
        alpha = -2.5
        fit = self._power_law(x, alpha)

        # plots
        fig = plt.figure(dpi=300, figsize=(6, 4))
        ax = plt.subplot()
        ax.loglog(x, p, color="black")
        ax.loglog(x[25:], fit[25:], "--", color="red")
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
        if self.save:
            fig.savefig("images/pdf{}.{}".format(self.resolution, self.fig_type))