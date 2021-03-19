# ----------------------------------------------------------------------
#   Data statistics module
# ----------------------------------------------------------------------
#   This module's purpose is to compute all relevant statistics and
#   other data manipulation in order to extract information.
#   It does not plot anything. To do this, go to visualization module.
#
#   TODO:
#   -compute KS distance
#   -compute time scaling
#   -compute space scaling
#
#   CANCELED  compute lagrangian trajectories from eulerian ones
# ----------------------------------------------------------------------

import numpy as np
import libs.selection as sel


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
            scale (int): scale in grid cell number (e.g. 2 , 4, 8, etc).
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
            indices = np.ix_(np.arange(scale - extra_i) + i, np.arange(scale) + j)

        elif scale + i <= self.ny and scale + j > self.nx:
            extra_j = scale + j - self.nx
            indices = np.ix_(np.arange(scale) + i, np.arange(scale - extra_j) + j)

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
            formated_data (np.ndarray): array of size (ny, nx, 2, nt) where each nt is a snapshot at a given time = time_ini + nt * dt
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
            data_time_mean = []
            for n in range(formated_data.shape[-1] - 3):
                data_time_mean.append(formated_data[..., n : n + 3].mean(axis=-1))

        # if dt is hours
        elif dtlist[1] == 1:
            data_time_mean = []
            for n in range(formated_data.shape[-1] - 3 * 24):
                data_time_mean.append(
                    formated_data[..., 24 * n : 24 * (n + 3)].mean(axis=-1)
                )

        # if dt is minutes (unlikely)
        elif dtlist[2] != 0:
            data_time_mean = []
            for n in range(formated_data.shape[-1] - 3 * 24 * 60 / dtlist[2]):
                data_time_mean.append(
                    formated_data[
                        ..., 60 / dtlist[2] * 24 * n : 60 / dtlist[2] * 24 * (n + 3)
                    ].mean(axis=-1)
                )

        else:
            raise SystemExit(
                "Unsupported time delta. Supported are 1 day, 1 hour, or any multiple or 1 jour in minutes."
            )

        return np.stack(data_time_mean, axis=-1)

    def _grid_size(self):
        # cartesian coordonates on the plane for the corners of the cells
        x0 = np.arange(self.nx + 1) * self.resolution - 2500
        y0 = np.arange(self.ny + 1) * self.resolution - 2250

        lon, lat = np.radians(self._coordinates(x0, y0)[0]), np.radians(
            self._coordinates(x0, y0)[1]
        )

        lon = np.abs(lon[1:, :] - lon[:-1, :])
        lat = np.abs(lat[:, 1:] - lat[:, :-1])

        areas = (
            2
            * self.R_EARTH ** 2
            * np.arcsin(
                np.tan(lon / (2 * self.R_EARTH)) * np.tan(lat / (2 * self.R_EARTH))
            )
        )

        return areas * self.R_EARTH ** 2

    def spatial_mean_box(
        self,
        formated_data: np.ndarray,
        scales: list,
        dt: str = None,
        deformation: bool = 0,
    ) -> np.ndarray:
        """
        Function that computes the lenght and deformation rate means over all boxes and all scales for all period of 3 days.

        Args:
            formated_data (np.ndarray): array of size (ny, nx, 2, nt) where each nt is a snapshot at a given time = time_ini + nt * dt
            scales (list): all scales under study in km.
            dt (str, optional): time difference between two points of data. Defaults to None for one snapshot.
            deformation (bool, optional): wether to compute deformations because using velocities. Defaults to 0.

        Raises:
            SystemExit: when input scale is smaller than or equal to resolution of the data.

        Returns:
            np.ndarray: array of size (n_scales, nx * ny * 86(87), 2). First is number of scales, second is max number of boxes (if smaller, replace the rest by NaNs). Third are: 0: deformation mean, 1: lenght mean.
        """

        # check if time average preprocessing is necessary
        if len(formated_data.shape) >= 3:
            formated_data = self._time_average(formated_data, dt)

            # computes the deformation rates
            if deformation:
                formated_data = self._deformation(formated_data)

            # computes all the areas
            areas = np.ones_like(formated_data[..., 0])

            # initialize data array where we will put our means
            data = np.empty(
                (len(scales), self.ny * self.nx * formated_data.shape[-1], 2)
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
                            counts = np.unique(mask, return_counts=True)[1][1]
                            if counts >= scale_grid_unit ** 2 / 2:
                                masked_data = np.ma.asarray(
                                    formated_data[:, :, period_iter]
                                )
                                masked_areas = np.ma.asarray(areas)
                                masked_data.mask = mask
                                masked_areas.mask = mask
                                masked_data = np.ma.masked_invalid(masked_data)

                                # verify that there is enough data in the box
                                if masked_data.count() >= scale_grid_unit ** 2 / 2:
                                    data_mean = np.ma.average(
                                        masked_data, weights=masked_areas
                                    )
                                    spatial_scale = (
                                        np.sqrt(masked_data.count()) * self.resolution
                                    )
                                    data[scale_iter, box_iter, 0] = data_mean
                                    data[scale_iter, box_iter, 1] = spatial_scale
                                    box_iter += 1
                    print("Done with period {}.".format(period_iter + 1))
                data[scale_iter, box_iter:, :] = np.NaN
                print("\nDone with {} km scale.\n".format(scales[scale_iter]))
                scale_iter += 1

        # when we do this procedure on snapshots instead of time averages
        else:
            # computes the deformation rates
            if deformation:
                formated_data = self._deformation(formated_data)

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
                            masked_data = np.ma.asarray(formated_data)
                            masked_areas = np.ma.asarray(areas)
                            masked_data.mask = mask
                            masked_areas.mask = mask
                            masked_data = np.ma.masked_invalid(masked_data)

                            # verify that there is enough data in the box
                            if masked_data.count() >= scale_grid_unit ** 2 / 2:
                                data_mean = np.ma.average(
                                    masked_data, weights=masked_areas
                                )
                                spatial_scale = (
                                    np.sqrt(masked_data.count()) * self.resolution
                                )
                                data[scale_iter, box_iter, 0] = data_mean
                                data[scale_iter, box_iter, 1] = spatial_scale
                                box_iter += 1
                data[scale_iter, box_iter:, :] = np.NaN
                print("Done with {} km scale.".format(scales[scale_iter]))
                scale_iter += 1

        return data