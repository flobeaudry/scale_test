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
#   -verify that every things works.
#
#   CANCELED  compute lagrangian trajectories from eulerian ones
# ----------------------------------------------------------------------

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

    def _box(self, scale: int, i: int, j: int) -> np.ndarray:
        """
        Computes the mask for the box for a given scale, with a corner point (i ,j).

        Args:
            scale (int): scale in grid cell number (e.g. 2, 4, 8, etc).
            i (int): position in ny of the corner of the box.
            j (int): position in nx of the corner of the box.

        Returns:
            np.ndarray: mask of the box in a (ny, nx) grid.
        """
        # boxes definition
        if scale + i <= self.ny and scale + j <= self.nx:
            indices = np.ix_(np.arange(scale) + i, np.arange(scale) + j)

        elif scale + i > self.ny and scale + j > self.nx:
            extra_i = scale + i - self.ny
            extra_j = scale + j - self.nx
            indices = np.ix_(
                np.arange(scale - extra_i) + i, np.arange(scale - extra_j) + j
            )

        elif scale + i > self.ny and scale + j <= self.nx:
            extra_i = scale + i - self.ny
            indices = np.ix_(
                np.arange(scale - extra_i) + i, np.arange(scale) + j
            )

        elif scale + i <= self.ny and scale + j > self.nx:
            extra_j = scale + j - self.nx
            indices = np.ix_(
                np.arange(scale) + i, np.arange(scale - extra_j) + j
            )

        # create box by creating a mask of ones on the grid
        box = np.full((self.ny, self.nx), 1, dtype=int)
        box[indices] = 0

        return box

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
                (formated_data[..., n : n + 3].mean(axis=-1))
                for n in range(formated_data.shape[-1] - 2)
            ]

        # if dt is hours
        elif dtlist[1] != 0:
            period_per_day = 24 // dtlist[1]
            data_time_mean = [
                formated_data[
                    ..., period_per_day * n : period_per_day * (n + 3)
                ].mean(axis=-1)
                for n in range(
                    (formated_data.shape[-1] - 3 * period_per_day)
                    // period_per_day
                    + 1
                )
            ]

        # if dt is minutes (unlikely)
        elif dtlist[2] != 0:
            period_per_day = 24 * 60 // dtlist[2]
            data_time_mean = [
                formated_data[
                    ..., period_per_day * n : period_per_day * (n + 3)
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

    def spatial_mean_box(
        self,
        formated_data: np.ndarray,
        scales: list,
        dt: str = None,
        time_end: str = None,
        from_velocity: bool = 0,
        choice: int = 0,
    ) -> np.ndarray:
        """
        Function that computes the lenght and deformation rate means over all boxes and all scales for all period of 3 days.

        Args:
            formated_data (np.ndarray): array of size (ny, nx, nt) or (ny, nx, 2, nt) where each nt is a snapshot at a given time = time_ini + nt * dt

            scales (list): all scales under study in km.
            dt (str, optional): time difference between two points of data. Defaults to None for one snapshot.

            time_end (str, optional): time of the last point in the data. Defaults to None for one snapshot.

            from_velocity (bool, optional): wether to compute deformations because using velocities. Defaults to 0. Only matters when using time average.

            choice (int, optional): when computing vel_to_def, choice for which deformation to compute.

        Raises:
            SystemExit: when input scale is smaller than or equal to resolution of the data.

        Returns:
            data (np.ndarray): array of size (n_scales, nx * ny * 86(87), 2). First is number of scales, second is max number of boxes (if smaller, replace the rest by NaNs). Third are: 0: deformation mean, 1: lenght mean.
            visc (np.ndarray): array of size (n_scales, nx * ny * 86(87)). Same thing but for the viscosity instead of deformation + lenght.
        """

        # check if time average preprocessing is necessary
        if len(formated_data.shape) >= 3:
            # load viscosities
            visc_raw = self.multi_load(dt, time_end, datatype="viscosity")

            # time average the data
            formated_data = self._time_average(formated_data, dt)
            formated_visc = self._time_average(visc_raw, dt)

            if from_velocity:
                # compute the derivatives and the deformations
                du, dv = self._derivative(
                    formated_data[:, :, 0, :], formated_data[:, :, 1, :]
                )
                formated_data = self._deformation(du, dv, choice)
                formated_visc = formated_visc[1:-1, 1:-1, :]

            # computes all the areas
            areas = np.ones_like(formated_data[..., 0])

            # initialize data array where we will put our means
            data = np.empty(
                (len(scales), self.ny * self.nx * formated_data.shape[-1], 2)
            )
            visc = np.empty(
                (len(scales), self.ny * self.nx * formated_visc.shape[-1])
            )

            # loop over all scales
            scale_iter = 0
            for scale_km_unit in scales:
                # verify validity of scale
                if scale_km_unit <= self.resolution:
                    scale_iter += 1
                    raise SystemExit(
                        "Scale is smaller than or equal to resolution. It's not implemented yet."
                    )

                # convert km into grid cell units
                scale_grid_unit = scale_km_unit // self.resolution

                # total number of boxes
                box_iter = 0
                # loop over all periods of 3 days
                for period_iter in range(formated_data.shape[-1]):
                    # loops over all possible boxes that are in the domain
                    for i in range(0, self.ny, scale_grid_unit // 2):
                        for j in range(0, self.nx, scale_grid_unit // 2):

                            # verify that box is big enough (for boundaries).
                            mask = self._box(scale_grid_unit, i, j)
                            counts = np.unique(mask, return_counts=True)[1][0]
                            if counts >= scale_grid_unit ** 2 / 2:
                                # define arrays for mask
                                masked_data = np.ma.asarray(
                                    formated_data[..., period_iter]
                                )
                                masked_areas = np.ma.asarray(areas)
                                masked_visc = np.ma.asarray(
                                    formated_visc[..., period_iter]
                                )
                                # mask data with box + invalid
                                masked_data.mask = mask
                                masked_data = np.ma.masked_invalid(masked_data)
                                # obtain new mask for both conditions
                                mask = np.ma.getmask(masked_data)
                                # mask the other arrays
                                masked_areas.mask = mask
                                masked_visc.mask = mask

                                # verify that there is enough data in the box
                                if (
                                    masked_data.count()
                                    >= scale_grid_unit ** 2 / 2
                                ):
                                    data_mean = np.ma.average(
                                        masked_data, weights=masked_areas
                                    )
                                    visc_mean = np.ma.average(
                                        masked_visc, weights=masked_areas
                                    )
                                    spatial_scale = (
                                        np.sqrt(masked_data.count())
                                        * self.resolution
                                    )
                                    data[scale_iter, box_iter, 0] = data_mean
                                    data[
                                        scale_iter, box_iter, 1
                                    ] = spatial_scale
                                    visc[scale_iter, box_iter] = visc_mean
                                    box_iter += 1
                    print(
                        "Done with period {} at box {}/{}.".format(
                            period_iter + 1,
                            box_iter,
                            int(
                                self.ny
                                * self.nx
                                / (scale_grid_unit ** 2)
                                * formated_data.shape[-1]
                            ),
                        )
                    )
                data[scale_iter, box_iter:, :] = np.NaN
                visc[scale_iter, box_iter:] = np.NaN
                print("\nDone with {} km scale.\n".format(scales[scale_iter]))
                scale_iter += 1

        # when we do this procedure on snapshots instead of time averages
        else:
            # load viscosities
            visc_raw = self._load_datatype("viscosity")
            visc = np.empty((len(scales), self.ny * self.nx))

            # computes all the areas
            areas = np.ones_like(formated_data)

            # initialize data array where we will put our means
            data = np.empty((len(scales), self.ny * self.nx, 2))

            # loop over all scales
            scale_iter = 0
            for scale_km_unit in scales:
                if scale_km_unit <= self.resolution:
                    scale_iter += 1
                    raise SystemExit(
                        "Scale is smaller than or equal to resolution. It's not implemented yet."
                    )
                # convert km into grid cell units
                scale_grid_unit = scale_km_unit // self.resolution

                # loops over all possible boxes that are in the domain
                box_iter = 0
                for i in range(0, self.ny, scale_grid_unit // 2):
                    for j in range(0, self.nx, scale_grid_unit // 2):
                        # verify that box it big enough.
                        mask = self._box(scale_grid_unit, i, j)
                        counts = np.unique(mask, return_counts=True)[1][1]
                        if counts >= scale_grid_unit ** 2 / 2:
                            # define arrays for mask
                            masked_data = np.ma.asarray(formated_data)
                            masked_areas = np.ma.asarray(areas)
                            masked_visc = np.ma.asarray(visc_raw)
                            # mask data with box + invalid
                            masked_data.mask = mask
                            masked_data = np.ma.masked_invalid(masked_data)
                            # obtain new mask for both conditions
                            mask = np.ma.getmask(masked_data)
                            # mask the other arrays
                            masked_areas.mask = mask
                            masked_visc.mask = mask

                            # verify that there is enough data in the box
                            if masked_data.count() >= scale_grid_unit ** 2 / 2:
                                data_mean = np.ma.average(
                                    masked_data, weights=masked_areas
                                )
                                visc_mean = np.ma.average(
                                    masked_visc, weights=masked_areas
                                )
                                spatial_scale = (
                                    np.sqrt(masked_data.count())
                                    * self.resolution
                                )
                                data[scale_iter, box_iter, 0] = data_mean
                                data[scale_iter, box_iter, 1] = spatial_scale
                                visc[scale_iter, box_iter] = visc_mean
                                box_iter += 1
                                print(
                                    "Done with box {}/{}.".format(
                                        box_iter,
                                        int(
                                            self.ny
                                            * self.nx
                                            / (scale_grid_unit ** 2)
                                        ),
                                    )
                                )
                data[scale_iter, box_iter:, :] = np.NaN
                visc[scale_iter, box_iter:] = np.NaN
                print("Done with {} km scale.".format(scales[scale_iter]))
                scale_iter += 1

        return data, visc

    def _clean(self, data: np.ndarray, box: bool = 1) -> np.ndarray:
        """
        Function that cleans the data for statistics.

        Args:
            data (np.ndarray): unclean data
            box (bool): if data comes from box or not

        Returns:
            np.ndarray: clean data
        """
        # clean data of non contributing NaNs
        if box:
            deformation = data[..., 0].flatten()

        else:
            deformation = data

        deformation = np.where(deformation >= 0.005, deformation, np.NaN)
        nas = np.isnan(deformation)
        deformation = deformation[~nas]

        return deformation

    def _clean_vect(self, deformation: np.ndarray, scales: list) -> np.ndarray:
        """
        Same as function above but for vectorized output.

        Args:
            deformation (np.ndarray): unclean data

        Returns:
            np.ndarray: clean data
        """
        all_def = []
        for k in range(len(scales)):
            all_def.append(deformation[k].flatten())
        all_def_array = np.concatenate(all_def)

        all_def_array = np.where(all_def_array >= 0.005, all_def_array, np.NaN)
        nas = np.isnan(all_def_array)
        all_def_array = all_def_array[~nas]

        return all_def_array

    def mle_exponent(
        self, data: np.ndarray, minimum_deformation: float
    ) -> float:
        """
        Computes the exponent alpha of the power law.

        Args:
            data (np.ndarray): array of all the deformations
            minimum_deformation (float): deformation at which the PDF is exhibiting a power law behaviour. It is to only get the tail of the PDF.

        Returns:
            alpha (float): exponent (slope in log-log) of the power law distribution.
            sigma (float): error
        """

        # count the number of deformations (equivalent to the sum of the number of boxes in each scale)
        n = np.count_nonzero(data)
        # print(np.where(data > minimum_deformation, 0))

        # compute alpha
        alpha = 1 + n * (np.sum(np.log(data / minimum_deformation))) ** (-1)
        sigma = (alpha - 1) / np.sqrt(n)

        return alpha, sigma

    def cumul_dens_func(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the CDF of the data simply by sorting the array from the smallest value to the biggest.

        Args:
            data (np.ndarray): data from which to compute the CDF

        Returns:
            np.ndarray: sorted data set
            np.ndarray: proportional values of sample (y axis)
        """

        # y axis for the data
        cdf_norm = np.linspace(0, 1, len(data))

        return np.sort(data), cdf_norm

    def ks_distance_minimizer(
        self, pdf_data: np.ndarray, pdf_norm: np.ndarray, end: int = -5
    ) -> float:
        """
        Function that minimizes the kolmogorov-smirnov distance. This is a measure of the distance betweeen the CDF of the data and the CDF of the fit/observations.

        Args:
            pdf_data (np.ndarray): proprocessed data on which to perform the algo.
            pdf_norm (np.ndarray): y axis corresponding to the data.
            end (int, optional): final data point on which to perform the algo. Just to get rid of cases where we try ti fit less than ten points in the PDF.

        Returns:
            dedt_min (float): deformation for which the ks distance in minimal.
            min_ks (float): minimum ks distance.
            best_fit (np.poly1D): polynomial that fits the pdf the best.
            min_index (float): index of the minimum.
        """

        # initialize Kolmogorov-Smirnov array
        ks_dist = np.empty_like(pdf_data[:end])

        # loop over all possible dedt_min
        for i in range(len(ks_dist)):
            # fit for values over dedt_min
            coefficients = np.polyfit(
                np.log(pdf_data[i:]), np.log(pdf_norm[i:]), 1
            )
            fit = np.poly1d(coefficients)

            # compute CDF
            cdf_fit, _ = self.cumul_dens_func(
                np.exp(fit(np.log(pdf_data[i:])))
            )
            cdf_data, _ = self.cumul_dens_func(pdf_data[i:])

            # compute kolmogorov-smirnov distance
            ks_dist[i] = np.max(np.abs(cdf_data - cdf_fit))

        # extract index of min value
        min_index = np.where(ks_dist == np.min(ks_dist))[0][0]
        dedt_min = pdf_data[min_index]
        min_ks = ks_dist[min_index]
        coefficients = np.polyfit(
            np.log(pdf_data[min_index:]), np.log(pdf_norm[min_index:]), 1
        )
        best_fit = np.poly1d(coefficients)

        return dedt_min, min_ks, best_fit, min_index, coefficients

    def spatial_mean_vect(
        self, u_v: np.ndarray, scales: list, dt: str = None,
    ) -> np.ndarray:
        """
        Same function as spatial_mean_box above, but this is the vectorized form of it. It is WAY faster.

        Args:
            u_v (np.ndarray): Velocities in x and y, shape is (ny, nx, 2, nt)
            scales (list): list of scales to compute.
            dt (str, optional): Time incrementation for the time averaging function. Defaults to None.

        Raises:
            SystemExit: If the input scale is equal to or smaller than the resolution of the data.

        Returns:
            np.ndarray: returns an array of size (len(scales),) where each element of the array are of different sizes (because it depends on the sizes of the boxes, therefore, for each scale the size of the data changes).
        """

        # time average the data
        u_v_ta = self._time_average(u_v, dt)

        u_v_ta_bool = u_v_ta != 0
        u_v_ta_bool = u_v_ta_bool.astype(int)

        # initialize output
        deps = []
        shear = []
        div = []
        scaling = []

        # loop over all scales
        for scale_km_unit in scales:
            # verify validity of scale
            if scale_km_unit <= self.resolution:
                raise SystemExit(
                    "Scale is smaller than or equal to resolution. It's not implemented yet."
                )

            # convert km into grid cell units
            scale_grid_unit = scale_km_unit // self.resolution

            # implementation of the algorithm
            u_v_ta_bool_sum = np.empty(
                (
                    u_v_ta.shape[0] // scale_grid_unit * 2,
                    u_v_ta.shape[1] // scale_grid_unit * 2,
                    u_v_ta.shape[2],
                    u_v_ta.shape[3],
                )
            )
            u_v_ta_sum = np.empty(
                (
                    u_v_ta.shape[0] // scale_grid_unit * 2,
                    u_v_ta.shape[1] // scale_grid_unit * 2,
                    u_v_ta.shape[2],
                    u_v_ta.shape[3],
                )
            )
            for i in range(u_v_ta.shape[0] // scale_grid_unit * 2):
                for j in range(u_v_ta.shape[1] // scale_grid_unit * 2):
                    u_v_ta_bool_sum[i, j, :, :] = np.sum(
                        u_v_ta_bool[
                            scale_grid_unit // 2 * i : scale_grid_unit // 2 * i
                            + scale_grid_unit,
                            scale_grid_unit // 2 * j : scale_grid_unit // 2 * j
                            + scale_grid_unit,
                            :,
                            :,
                        ].reshape(
                            -1,
                            *u_v_ta_bool[
                                scale_grid_unit
                                // 2
                                * i : scale_grid_unit
                                // 2
                                * i
                                + scale_grid_unit,
                                scale_grid_unit
                                // 2
                                * j : scale_grid_unit
                                // 2
                                * j
                                + scale_grid_unit,
                                :,
                                :,
                            ].shape[-2:]
                        ),
                        axis=0,
                    )
                    u_v_ta_sum[i, j, :, :] = np.sum(
                        u_v_ta[
                            scale_grid_unit // 2 * i : scale_grid_unit // 2 * i
                            + scale_grid_unit,
                            scale_grid_unit // 2 * j : scale_grid_unit // 2 * j
                            + scale_grid_unit,
                            :,
                            :,
                        ].reshape(
                            -1,
                            *u_v_ta[
                                scale_grid_unit
                                // 2
                                * i : scale_grid_unit
                                // 2
                                * i
                                + scale_grid_unit,
                                scale_grid_unit
                                // 2
                                * j : scale_grid_unit
                                // 2
                                * j
                                + scale_grid_unit,
                                :,
                                :,
                            ].shape[-2:]
                        ),
                        axis=0,
                    )

            u_v_ta_bool_sum = np.where(
                u_v_ta_bool_sum < scale_grid_unit ** 2 // 2,
                np.NaN,
                u_v_ta_bool_sum,
            )

            u_v_ta_mean = u_v_ta_sum / u_v_ta_bool_sum

            # compute the derivatives and the deformations
            du_mean, dv_mean = self._derivative(
                u_v_ta_mean[:, :, 0, :],
                u_v_ta_mean[:, :, 1, :],
                scale_grid_unit // 2,
            )
            # viscosity = viscosity[1:-1, 1:-1, :]

            # compute the deformation
            deps_array = self._deformation(du_mean, dv_mean, 0)
            shear_array = self._deformation(du_mean, dv_mean, 1)
            div_array = self._deformation(du_mean, dv_mean, 2)

            # compute the scaling associated with each box note here that I multiply by v then divide by v so that I get both the NaNs in u and v (to match the NaNs in deps).
            scale_array = (
                np.sqrt(
                    u_v_ta_bool_sum[:, :, 0, :]
                    * u_v_ta_bool_sum[:, :, 1, :]
                    / u_v_ta_bool_sum[:, :, 1, :]
                )
                * self.resolution
            )

            deps.append(deps_array)
            shear.append(shear_array)
            div.append(div_array)
            scaling.append(scale_array)

            print("Done with scale {}.".format(scale_km_unit))

        # creates an array of arrays but of different shapes
        deps = np.asarray(deps, dtype=object)
        shear = np.asarray(shear, dtype=object)
        div = np.asarray(div, dtype=object)
        scaling = np.asarray(scaling, dtype=object)

        return deps, shear, div, scaling

    def spatial_mean_RGPS(
        self, shear: np.ndarray, div: np.ndarray, scales: list,
    ) -> np.ndarray:
        """
        Same function as spatial_mean_box above, but this is the vectorized form of it. It is WAY faster.

        Args:
            shear/div (np.ndarray): deformations in x and y, shape is (ny, nx, nt), already time averaged
            scales (list): list of scales to compute.
        Raises:
            SystemExit: If the input scale is equal to or smaller than the resolution of the data.

        Returns:
            np.ndarray: returns an array of size (len(scales),) where each element of the array are of different sizes (because it depends on the sizes of the boxes, therefore, for each scale the size of the data changes).
        """

        shear_bool = ~np.isnan(shear)
        shear_bool = shear_bool.astype(int)
        div_bool = ~np.isnan(div)
        div_bool = div_bool.astype(int)

        # initialize output
        deps_list = []
        shear_list = []
        div_list = []
        deps_scaling_list = []
        shear_scaling_list = []
        div_scaling_list = []

        # loop over all scales
        for scale_km_unit in scales:
            # verify validity of scale
            if scale_km_unit <= RES_RGPS:
                raise SystemExit(
                    "Scale is smaller than or equal to resolution. It's not implemented yet."
                )

            # convert km into grid cell units
            scale_grid_unit = int(scale_km_unit // RES_RGPS)

            # implementation of the algorithm for shear
            shear_bool_sum = np.empty(
                (
                    shear.shape[0] // scale_grid_unit * 2,
                    shear.shape[1] // scale_grid_unit * 2,
                    shear.shape[2],
                )
            )
            shear_sum = np.empty(
                (
                    shear.shape[0] // scale_grid_unit * 2,
                    shear.shape[1] // scale_grid_unit * 2,
                    shear.shape[2],
                )
            )
            for i in range(shear.shape[0] // scale_grid_unit * 2):
                for j in range(shear.shape[1] // scale_grid_unit * 2):
                    shear_bool_sum[i, j, :] = np.sum(
                        shear_bool[
                            scale_grid_unit // 2 * i : scale_grid_unit // 2 * i
                            + scale_grid_unit,
                            scale_grid_unit // 2 * j : scale_grid_unit // 2 * j
                            + scale_grid_unit,
                            :,
                        ].reshape(
                            -1,
                            *shear_bool[
                                scale_grid_unit
                                // 2
                                * i : scale_grid_unit
                                // 2
                                * i
                                + scale_grid_unit,
                                scale_grid_unit
                                // 2
                                * j : scale_grid_unit
                                // 2
                                * j
                                + scale_grid_unit,
                                :,
                            ].shape[-1:]
                        ),
                        axis=0,
                    )
                    shear_sum[i, j, :] = np.sum(
                        shear[
                            scale_grid_unit // 2 * i : scale_grid_unit // 2 * i
                            + scale_grid_unit,
                            scale_grid_unit // 2 * j : scale_grid_unit // 2 * j
                            + scale_grid_unit,
                            :,
                        ].reshape(
                            -1,
                            *shear[
                                scale_grid_unit
                                // 2
                                * i : scale_grid_unit
                                // 2
                                * i
                                + scale_grid_unit,
                                scale_grid_unit
                                // 2
                                * j : scale_grid_unit
                                // 2
                                * j
                                + scale_grid_unit,
                                :,
                            ].shape[-1:]
                        ),
                        axis=0,
                    )

            shear_bool_sum = np.where(
                shear_bool_sum < scale_grid_unit ** 2 // 2,
                np.NaN,
                shear_bool_sum,
            )

            shear_mean = shear_sum / shear_bool_sum

            # implementation of the algorithm for div
            div_bool_sum = np.empty(
                (
                    div.shape[0] // scale_grid_unit * 2,
                    div.shape[1] // scale_grid_unit * 2,
                    div.shape[2],
                )
            )
            div_sum = np.empty(
                (
                    div.shape[0] // scale_grid_unit * 2,
                    div.shape[1] // scale_grid_unit * 2,
                    div.shape[2],
                )
            )
            for i in range(div.shape[0] // scale_grid_unit * 2):
                for j in range(div.shape[1] // scale_grid_unit * 2):
                    div_bool_sum[i, j, :] = np.sum(
                        div_bool[
                            scale_grid_unit // 2 * i : scale_grid_unit // 2 * i
                            + scale_grid_unit,
                            scale_grid_unit // 2 * j : scale_grid_unit // 2 * j
                            + scale_grid_unit,
                            :,
                        ].reshape(
                            -1,
                            *div_bool[
                                scale_grid_unit
                                // 2
                                * i : scale_grid_unit
                                // 2
                                * i
                                + scale_grid_unit,
                                scale_grid_unit
                                // 2
                                * j : scale_grid_unit
                                // 2
                                * j
                                + scale_grid_unit,
                                :,
                            ].shape[-1:]
                        ),
                        axis=0,
                    )
                    div_sum[i, j, :] = np.sum(
                        div[
                            scale_grid_unit // 2 * i : scale_grid_unit // 2 * i
                            + scale_grid_unit,
                            scale_grid_unit // 2 * j : scale_grid_unit // 2 * j
                            + scale_grid_unit,
                            :,
                        ].reshape(
                            -1,
                            *div[
                                scale_grid_unit
                                // 2
                                * i : scale_grid_unit
                                // 2
                                * i
                                + scale_grid_unit,
                                scale_grid_unit
                                // 2
                                * j : scale_grid_unit
                                // 2
                                * j
                                + scale_grid_unit,
                                :,
                            ].shape[-1:]
                        ),
                        axis=0,
                    )

            div_bool_sum = np.where(
                div_bool_sum < scale_grid_unit ** 2 // 2, np.NaN, div_bool_sum,
            )

            div_mean = div_sum / div_bool_sum

            # compute the deformation
            deps_mean = np.sqrt(shear_mean ** 2 + div_mean ** 2)

            # compute the scaling associated with each box note here that I multiply by v then divide by v so that I get both the NaNs in u and v (to match the NaNs in deps).
            deps_scale_array = (
                np.sqrt(shear_bool_sum * div_bool_sum / div_bool_sum)
                * RES_RGPS
            )
            shear_scale_array = np.sqrt(shear_bool_sum) * RES_RGPS
            div_scale_array = np.sqrt(div_bool_sum) * RES_RGPS

            deps_list.append(deps_mean)
            shear_list.append(shear_mean)
            div_list.append(div_mean)
            deps_scaling_list.append(deps_scale_array)
            shear_scaling_list.append(shear_scale_array)
            div_scaling_list.append(div_scale_array)

            print("Done with scale {}.".format(scale_km_unit))

        # creates an array of arrays but of different shapes
        deps = np.asarray(deps_list, dtype=object)
        shear = np.asarray(shear_list, dtype=object)
        div = np.asarray(div_list, dtype=object)
        deps_scaling = np.asarray(deps_scaling_list, dtype=object)
        shear_scaling = np.asarray(shear_scaling_list, dtype=object)
        div_scaling = np.asarray(div_scaling_list, dtype=object)

        return deps, shear, div, deps_scaling, shear_scaling, div_scaling

