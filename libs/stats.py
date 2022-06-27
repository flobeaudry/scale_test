# ----------------------------------------------------------------------
#   Data statistics module
# ----------------------------------------------------------------------
#   This module's purpose is to compute all relevant statistics and
#   other data manipulation in order to extract information.
#   It does not plot anything. To do this, go to visualization module.
#
#   TODO:
#   DONE    compute KS distance for dedt_min
#   -compute time scaling
#   DONE    compute space scaling
#   DONE    compute mask for all boxes
#   DONE    compute time average for 3 days
#   DONE    compute power law from data
#   DONE    compute CDF with histogram
#   DONE    compute alpha (exponent)
#   DONE    verify that every things works.
#   -compute multifractal parameters
#
#   CANCELED  compute lagrangian trajectories from eulerian ones
# ----------------------------------------------------------------------

from pickletools import optimize
from matplotlib import pyplot as plt
import numpy as np
import libs.selection as sel
from libs.constants import *


class Scale(sel.Data):
    """
    This class is a child class to sel.Data, and a parent class to vis.Arctic. Its main goal is to compute the wanted statistics on the scale.
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
    ):
        """
        Class attributes for Scale.

        Args:
            directory (str, optional): directory from which to take data. Defaults to None.

            time (str, optional): starting time, format supported: yyyy-mm-dd-hh-mm. Defaults to None.

            expno (str, optional): experience number is of format nn. Defaults to None.
            datatype (str, optional): data types currently supported are: ice concentration (A), ice thickness (h), ice velocity vector (u), ice temp (Ti) (needs tweeks for pcolor), and ice deformation (dedt). Defaults to None.

            tolerance (float, optional): value at which dedt will be cut to get rid of high boundary values. Defaults to 0.1.

            resolution (int, optional): spatial resolution of the domain of interest.

            nx, ny (int, optional): number of cells in each direction.
        """
        super(Scale, self).__init__(
            directory=directory,
            time=time,
            expno=expno,
            datatype=datatype,
            tolerance=tolerance,
            resolution=resolution,
            nx=nx,
            ny=ny,
        )

    def _signal_to_noise(self):
        pass

    def _time_average(self, formated_data: np.ndarray, dt: str) -> np.ndarray:
        """
        Function that computes the time average over 3 days depdending on the time dicretization of the data.

        Args:
            formated_data (np.ndarray): array of size (ny, nx, nt) where each nt is a snapshot at a given time = time_ini + nt * dt
            dt (str): time difference between two points of data.

        Raises:
            SystemExit: if data is not 1 day, 1 hour, or a multiple or 1 hour in minutes

        Returns:
            np.ndarray: all the means of size (ny , nx, 86(87)) (because 86(87) periods between 02/01 and 31/03 1997(2008))
        """
        # create list
        dtlist = [int(n) for n in dt.split("-") if n.isdigit()]

        # if dt is days
        if dtlist[0] == 1:
            data_time_mean = [
                (formated_data[..., 3 * n : 3 * n + 3].mean(axis=-1))
                for n in range(formated_data.shape[-1] // 3)
            ]

        # if dt is hours
        elif dtlist[1] != 0:
            period_per_day = 24 // dtlist[1]
            data_time_mean = [
                formated_data[
                    ..., period_per_day * 3 * n : period_per_day * (3 * n + 3)
                ].mean(axis=-1)
                for n in range(
                    (formated_data.shape[-1] // period_per_day) // 3
                )
            ]

        # if dt is minutes (unlikely)
        elif dtlist[2] != 0:
            period_per_day = 24 * 60 // dtlist[2]
            data_time_mean = [
                formated_data[
                    ..., period_per_day * 3 * n : period_per_day * (3 * n + 3)
                ].mean(axis=-1)
                for n in range(
                    (formated_data.shape[-1] - 3 * period_per_day)
                    // period_per_day
                    + 1
                )
            ]

        else:
            raise SystemExit(
                "Unsupported time delta. Supported are 1 day, a multiple of 24 hours, or any multiple or 60 minutes."
            )

        return np.stack(data_time_mean, axis=-1)

    def _pdf_interior(self, data: np.ndarray, choice: int):
        """
        It simply computes everything for pdf plot.

        Args:
            data (np.ndarray): data to process (flat array with no nans)
            choice (int): what type of data (shear, ndiv, pdiv)

        Returns:
            [type]: multiple things
        """

        # get correct data from box data
        if choice == 1:
            n = np.logspace(np.log10(5e-3), 1, num=66)
            p, x = np.histogram(data, bins=n, density=1)
            end = -24
        elif choice == 2:
            n = np.logspace(np.log10(5e-3), 0, num=46)
            p, x = np.histogram(-data, bins=n, density=1)
            end = -14
        elif choice == 3:
            n = np.logspace(np.log10(5e-3), 0, num=46)
            p, x = np.histogram(data, bins=n, density=1)
            end = -14

        cdf = self.cumul_dens_func(np.diff(x), p, method=1)
        p = np.where(p == 0.0, np.NaN, p)

        # convert bin edges to centers
        x_mid = (x[:-1] + x[1:]) / 2

        # compute best estimator
        (
            dedt_min,
            ks_dist,
            best_fit,
            min_index,
            coefficient,
        ) = self.ks_distance_minimizer(x_mid, p, cdf, end)
        alpha, sigma = self.mle_exponent(
            data[~np.isnan(np.where(data >= dedt_min, data, np.NaN))], dedt_min
        )
        return (
            x_mid,
            p,
            ks_dist,
            best_fit,
            min_index,
            coefficient,
            dedt_min,
            alpha,
            sigma,
        )

    def _definitions_RGPS(
        self, data: np.ndarray, scale_grid_unit: int
    ) -> np.ndarray:

        shape = list(data.shape)
        shape[0] = shape[0] // scale_grid_unit * 2
        shape[1] = shape[1] // scale_grid_unit * 2
        shape = tuple(shape)

        bool_sum = np.zeros(shape)
        data_sum = np.zeros(shape)

        return bool_sum, data_sum

    def _loop_interior_RGPS(
        self,
        i: int,
        j: int,
        data_bool: np.ndarray,
        data: np.ndarray,
        scale_grid_unit: int,
    ) -> np.ndarray:
        # numbers of dimensions that needs to be skipped
        dim = 2 - len(data.shape)
        # algo
        bool_sum = np.sum(
            data_bool[
                scale_grid_unit // 2 * i : scale_grid_unit // 2 * i
                + scale_grid_unit,
                scale_grid_unit // 2 * j : scale_grid_unit // 2 * j
                + scale_grid_unit,
            ].reshape(
                -1,
                *data_bool[
                    scale_grid_unit // 2 * i : scale_grid_unit // 2 * i
                    + scale_grid_unit,
                    scale_grid_unit // 2 * j : scale_grid_unit // 2 * j
                    + scale_grid_unit,
                ].shape[dim:]
            ),
            axis=0,
        )
        data_sum = np.nansum(
            data[
                scale_grid_unit // 2 * i : scale_grid_unit // 2 * i
                + scale_grid_unit,
                scale_grid_unit // 2 * j : scale_grid_unit // 2 * j
                + scale_grid_unit,
            ].reshape(
                -1,
                *data[
                    scale_grid_unit // 2 * i : scale_grid_unit // 2 * i
                    + scale_grid_unit,
                    scale_grid_unit // 2 * j : scale_grid_unit // 2 * j
                    + scale_grid_unit,
                ].shape[dim:]
            ),
            axis=0,
        )

        return bool_sum, data_sum

    def spatial_mean_du(self, du: np.ndarray, scales: list) -> np.ndarray:
        """
            Same function as spatial_mean_box above, but this is the vectorized form of it. It is WAY faster. USE FOR ALL.

            Args:
                du (np.ndarray): derivatives in x and y, shape is (ny, nx, nt, 4), already time averaged
                scales (list): list of scales to compute.

            Returns:
                np.ndarray: returns an array of size (len(scales),) where each element of the array are of different sizes (because it depends on the sizes of the boxes, therefore, for each scale the size of the data changes).
            """

        du_bool = ~np.isnan(du)
        du_bool = du_bool.astype(int)

        # initialize output
        deps_list = []
        shear_list = []
        div_list = []
        scaling_list = []

        # loop over all scales
        for scale_km_unit in scales:
            # verify validity of scale
            if scale_km_unit <= self.resolution:
                shear = self._deformation(du, 1)
                div = self._deformation(du, 2)
                deps = self._deformation(du, 0)
                shear_list.append(shear)
                div_list.append(div)
                deps_list.append(deps)
                scaling_list.append(self.resolution * np.ones_like(deps))
                print("Done with spatial scale {}.".format(scale_km_unit))
                continue

            # convert km into grid cell units
            scale_grid_unit = int(scale_km_unit // self.resolution)

            # implementation of the algorithm
            # definitions
            du_bool_sum, du_sum = self._definitions_RGPS(du, scale_grid_unit)

            # big loop over all the indices
            for i in range(du.shape[0] // scale_grid_unit * 2):
                for j in range(du.shape[1] // scale_grid_unit * 2):
                    # algo for all derivatives
                    (
                        du_bool_sum[i, j],
                        du_sum[i, j],
                    ) = self._loop_interior_RGPS(
                        i, j, du_bool, du, scale_grid_unit
                    )

            # take only boxes that are at least 50% filled
            du_bool_sum = np.where(
                du_bool_sum < scale_grid_unit ** 2 // 2, np.NaN, du_bool_sum,
            )

            # compute the means
            du_mean = du_sum / du_bool_sum

            # delete boxes with mean smaller than 5e-3 and compute the deformation
            # shear_mean = np.where(shear_mean < 5e-3, np.NaN, shear_mean)
            # div_mean = np.where(np.abs(div_mean) < 5e-3, np.NaN, div_mean)
            shear_mean = self._deformation(du_mean, 1)
            div_mean = self._deformation(du_mean, 2)
            deps_mean = self._deformation(du_mean, 0)

            # compute the scaling associated with each box
            scaling_array = (
                np.sqrt(
                    du_bool_sum[..., 0]
                    * du_bool_sum[..., 2]
                    / du_bool_sum[..., 2]
                )
                * self.resolution
            )

            deps_list.append(deps_mean)
            shear_list.append(shear_mean)
            div_list.append(div_mean)
            scaling_list.append(scaling_array)

            print("Done with spatial scale {}.".format(scale_km_unit))

        # creates an array of arrays but of different shapes
        deps = np.asarray(deps_list, dtype=object)
        shear = np.asarray(shear_list, dtype=object)
        div = np.asarray(div_list, dtype=object)
        scaling = np.asarray(scaling_list, dtype=object)

        return (deps, shear, div, scaling)

    def structure_function(
        self, q: np.ndarray, v: int, C1: int, H: int
    ) -> np.ndarray:
        """
        This function will compute the multifractality of the data

        Args:
            q (np.ndarray): moment q (x axis)
            v (int): multifractality (0 < v < 2)
            C1 (int): heterogeneity (0 < C1 < 2)
            H (int): Hurst exponent (0 < H < 1) beta(1) = 1 - H -> H is determined by slope of q = 1.

        Returns:
            np.ndarray: value of the slope beta(q).
        """

        beta = C1 / (v - 1) * q ** v + (1 - H - C1 / (v - 1)) * q

        return beta

    def temporal_mean_du(
        self,
        du: np.ndarray,
        temp_scales: list,
        q: float = 1.0,
        RGPS: bool = False,
    ) -> tuple:
        """
        Function that computes the mean deformations for each temporal scale. How it works: start by averaging du on proper time scale; compute spatial mean and stock value; do that for all slice of time; then compute mean of all slice of time; stock value and start angain with next time scale.

        Args:
            du (np.ndarray): Velocity derivatives. (ny,nx,nt,4)
            temp_scales (list): Temporal scales under inspection
            q (float): moment for the multifractality.

        Returns:
            tuple: This is a tuple of size 4 with deps, shear, div and temp scale as 1D array of size shape(temp_scales).
        """
        # initialize output
        mean_deps = []
        mean_div = []
        mean_shear = []

        temp_scales = np.asarray(temp_scales)

        if RGPS == True:
            # this works, but it doesn't because of the NaNs and this prevents the interpolation from working (mask). In the meantime keep it at 12.5 km for RGPS and don't use this until I find a solution.
            du_interp = np.zeros((310, 310, du.shape[2], du.shape[3]))
            for i in range(du.shape[2]):
                for j in range(du.shape[3]):
                    du_interp[:, :, i, j] = self.interp_RGPS(du[:, :, i, j])
            du = du_interp

        for i in temp_scales // 3:
            # initialize output
            deps_list = []
            shear_list = []
            div_list = []

            for n in range(du.shape[2] - i + 1):
                du_mean = np.nanmean(du[:, :, n : n + i, :], axis=2)
                shear = self._deformation(du_mean, 1)
                div = self._deformation(du_mean, 2)
                deps = self._deformation(du_mean, 0)
                shear_mean = np.nanmean(shear ** q)
                div_mean = np.nanmean(div ** q)
                deps_mean = np.nanmean(deps ** q)

                shear_list.append(shear_mean)
                div_list.append(div_mean)
                deps_list.append(deps_mean)

            depsA = np.mean(np.asarray(deps_list))
            shearA = np.mean(np.asarray(shear_list))
            divA = np.mean(np.asarray(div_list))
            mean_deps.append(depsA)
            mean_div.append(divA)
            mean_shear.append(shearA)
            print("Done with temporal scale {}.".format(int(i * 3)))

        deps = np.asarray(mean_deps)
        shear = np.asarray(mean_shear)
        div = np.asarray(mean_div)

        return (deps, shear, div, temp_scales)

    def temporal_scaling_slope(
        self, mean_def: np.ndarray, mean_scale: list,
    ) -> float:
        """
        Simple function that computes the slope of the temporal scaling.

        Args:
            mean_def (np.ndarray): deformations
            mean_scale (list): temporal scale
        Returns:
            float: coefficient of the linear fit (the slope)
        """

        # linear regression over the means
        coefficients = np.polyfit(np.log(mean_scale), np.log(mean_def), 1)

        return coefficients[0]
