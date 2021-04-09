# ----------------------------------------------------------------------
#   Data selection module
# ----------------------------------------------------------------------
#   Its purpose is to fetch, load, and format the data from the
#   output files that I previously downloaded. There are many
#   options in this module.
#
#   TODO:
#   DONE    define function that loads various output infos
#   DONE    define function that loads values into numpy array
#   DONE    objectify the code
#   DONE    add option to write inputs in function arguments
#   DONE    add deformation option
#   DONE    add _deformation_format function to get rid of -999
#           and -888 values
#   DONE    look if we can do anything with class methods? probably yes for
#           loading multiple snapshots because classmethod can be used to
#           change parameters in the class (not done this way but still)
#   -add spline cubic interpolation for _velocity_format function
#   -automatization of the process by fetching on panda by itself
#
#
#   TODO (eventually):
#   -do a film of many snapshots (daily). To do this, load all
#   files into a numpy array (formated) 24 fps = 15 sec/year
#   Simply have to do the same code as here but put it in a
#   loop over the time variable in order to read and load
#   all files of a given experience. Could be done in the
#   same module but in a new function to not overwrite
#   these ones.
#   -combine load and multi_load so that you can call only module to
#   load anytype of data.
# ----------------------------------------------------------------------

import numpy as np
from os import path, listdir
from datetime import datetime, timedelta


class Data:
    """
    Data aquisition class, regroup everything linked to this subject.

    _list_files: restricted function that lists all files in a given directory with given extension
    _load_something: restricted functions that load specific variables
    _datatype_format: restricted functions that format specific datatype
    _get_times: restricted function that gets a list of all wanted snapshots
    _coordinates: function that computes the proper lon, lat coordinates.
    _deformation: function that commptes deformation with averaged velocities.
    _velolicty_vector_magnitude: restricted function that computes magnitude of the velocity of the ice.

    load: function that takes user input in order to load the wanted data set into a numpy array
    multi_load : function that loads multiple snapshots

    """

    R_EARTH = 6370  # Earth's radius (is smaller for better looking plots)
    BETA = 32  # Angle between domain and Greenwich

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
        Class attributes for Data.

        Args:
            directory (str, optional): directory from which to take data. Defaults to None.

            time (str, optional): starting time, format supported: yyyy-mm-dd-hh-mm. Defaults to None.

            expno (str, optional): experience number is of format nn. Defaults to None.
            datatype (str, optional): data types currently supported are: ice concentration (A), ice thickness (h), ice velocity vector (u), ice temp (Ti) (needs tweeks for pcolor), and ice deformation (dedt). Defaults to None.

            tolerance (float, optional): value at which dedt will be cut to get rid of high boundary values. Defaults to 0.1.

            resolution (int, optional): spatial resolution of the domain of interest.

            nx, ny (int, optional): number of cells in each direction.
        """

        print(
            "\nWelcome in the data loader routine! To cancel selection, enter exit or quit."
        )

        self.directory = directory
        self.time = time
        self.expno = expno
        self.datatype = datatype
        self.tolerance = tolerance
        self.nx = nx
        self.ny = ny
        self.resolution = resolution
        self.data = None
        self.name = None

    def _list_files(self, directory: str, extension: str) -> list:
        """
        Gives all files with given extension inside given directory.

        Args:
            directory (str): directory name
            extension (str): file extension to list

        Returns:
            (list): all files with given extension inside given directory
        """
        files = (f for f in listdir(directory) if f.endswith("." + extension))

        return [f for f in files]

    def _load_directory(self, directory: str):
        """
        Loads output directory.

        Args:
            directory (str): directory to load in class attribute

        Raises:
            SystemExit: manual exit
            SystemExit: directory does not exist
        """

        if directory is not None:
            self.directory = directory

        # loop until input is valid
        while True:
            # counter to verify if it is input or argument (0: arg, 1:input)
            n = 0

            # enter directory
            if self.directory is None:
                self.directory = input(
                    "\nEnter the directory name of your output files: "
                )
                n += 1

            # verify its existence
            if path.exists(self.directory):
                break

            # to exit manually
            elif self.directory.startswith(("exit", "quit")):
                raise SystemExit("\nYou exited properly.")

            # error if directory does not exist
            else:
                print("Directory exists: " + str(path.exists(self.directory)))

                if n:
                    print("Please enter an existing directory name.\n")
                    self.directory = None

                else:
                    raise SystemExit("\nError in directory name, it does not exist.")

    def _load_time(self, time: str):
        """
        Loads proper time and date.

        Args:
            time (str): time to load of format YYYY-MM-DD-HH-MM

        Raises:
            SystemExit: manual exit
            SystemExit: format is not good
        """

        if time is not None:
            self.time = time

        # loop until input is valid
        while 1:
            # counter to verify if it is input or argument (0: arg, 1:input)
            n = 0

            # enter date and time and split it
            if self.time is None:
                self.time = input("At what time? (format is yyyy-mm-dd-hh-mm) ").split(
                    "-"
                )
                n += 1

            # if arguments, split it
            else:
                self.time = self.time.split("-")

            # verify input validity
            if len(self.time) == 5:
                break

            # to exit manually
            elif self.time.startswith(("exit", "quit")):
                raise SystemExit("\nYou exited properly.")

            # error if date is not in correct format
            else:
                print(
                    "Date is invalid, looking for yyyy-mm-dd-hh-mm (lenght is 5).\nReceived input of lenght "
                    + str(len(self.time))
                    + "\n"
                )

                if n:
                    self.time = None

                else:
                    raise SystemExit()

    def _load_expno(self, expno: str):
        """
        Loads proper experience number.
        Needs directory to have been load previously to work.

        Args:
            expno (str): experience number format nn

        Raises:
            SystemExit: manual exit
            SystemExit: experience number is invalid
        """

        if expno is not None:
            self.expno = expno

        # loop until input is valid
        while 1:
            # counter to verify if it is input or argument (0: arg, 1:input)
            n = 0

            # enter experience number
            if self.expno is None:
                self.expno = input("Experience number? (format is ##) ")
                n += 1

            # verify input validity
            if self._list_files(self.directory, self.expno):
                break

            # to exit manually
            elif self.expno.startswith(("exit", "quit")):
                raise SystemExit("\nYou exited properly.")

            # error if exp number is invalid
            else:
                print(
                    "Exprience number is invalid, please provide a number from 00 to 99 that is in your output folder.\n"
                )

                if n:
                    self.expno = None

                else:
                    raise SystemExit()

    def _load_datatype(self, datatype: str) -> np.ndarray:
        """
        Loads proper data type.
        Needs directory, time and expno to have been previously load to work.

        Args:
            datatype (str): data type to load

        Raises:
            SystemExit: manual exit
            SystemExit: invalid data type

        Returns:
            np.ndarray: array of size (ny, nx) that contains values of data type at the grid center (Arakawa C-grid), velocity is of size (ny, nx, 2) because 2 components
        """

        if datatype is not None:
            self.datatype = datatype

        # loop until input is valid
        while 1:
            # counter to verify if it is input or argument (0: arg, 1:input)
            n = 0

            # enter data type
            if self.datatype is None:
                self.datatype = input("What data do you want to analyse? ")
                n += 1

            # ice concentration
            if self.datatype.startswith("A"):
                print(
                    "\nLoading ice concentration in file "
                    + self.directory
                    + "/A{}_{}_{}_{}_{}.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    )
                )

                fic = open(
                    self.directory
                    + "/A{}_{}_{}_{}_{}.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    ),
                    "r",
                )
                self.data = np.loadtxt(fic)
                fic.close()
                self.name = "Ice concentration"
                break

            # ice thickness
            elif self.datatype.startswith("h"):
                print(
                    "\nLoading ice thickness in file "
                    + self.directory
                    + "/h{}_{}_{}_{}_{}.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    )
                )

                fic = open(
                    self.directory
                    + "/h{}_{}_{}_{}_{}.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    ),
                    "r",
                )
                self.data = np.loadtxt(fic)
                fic.close()
                self.name = "Ice thickness [m]"
                break

            # for viscosity
            elif self.datatype.startswith("viscosity"):
                print(
                    "\nLoading ice viscosity in file "
                    + self.directory
                    + "/zetaC{}_{}_{}_{}_{}_k0000.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    )
                )

                fic = open(
                    self.directory
                    + "/zetaC{}_{}_{}_{}_{}_k0000.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    ),
                    "r",
                )
                self.data = np.loadtxt(fic)
                fic.close()
                self.name = "Viscosity [N day m$^-2$]"
                break

            # for shear
            elif self.datatype.startswith("shear"):
                print(
                    "\nLoading ice shear in file "
                    + self.directory
                    + "/shear{}_{}_{}_{}_{}_k0000.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    )
                )

                fic = open(
                    self.directory
                    + "/shear{}_{}_{}_{}_{}_k0000.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    ),
                    "r",
                )
                shear_raw = np.loadtxt(fic)
                self.data = self._deformation_format(shear_raw)
                fic.close()
                self.name = "Shear rate [day$^{-1}$]"
                break

            # for divergence
            elif self.datatype.startswith("divergence"):
                print(
                    "\nLoading ice divergence in file "
                    + self.directory
                    + "/div{}_{}_{}_{}_{}_k0000.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    )
                )

                fic = open(
                    self.directory
                    + "/div{}_{}_{}_{}_{}_k0000.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    ),
                    "r",
                )
                div_raw = np.loadtxt(fic)
                self.data = self._deformation_format(div_raw)
                fic.close()
                self.name = "Divergence rate [day$^{-1}$]"
                break

            # total ice deformation rate
            elif self.datatype.startswith("dedt"):
                print(
                    "\nLoading ice shear in file "
                    + self.directory
                    + "/shear{}_{}_{}_{}_{}_k0000.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    )
                )
                print(
                    "Loading ice divergence in file "
                    + self.directory
                    + "/div{}_{}_{}_{}_{}_k0000.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    )
                )

                fic = open(
                    self.directory
                    + "/shear{}_{}_{}_{}_{}_k0000.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    ),
                    "r",
                )
                shear_raw = np.loadtxt(fic)
                shear = self._deformation_format(shear_raw)
                fic.close()

                fic = open(
                    self.directory
                    + "/div{}_{}_{}_{}_{}_k0000.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    ),
                    "r",
                )
                div_raw = np.loadtxt(fic)
                div = self._deformation_format(div_raw)
                fic.close()

                self.data = np.sqrt(shear ** 2 + div ** 2)
                self.name = "Ice deformation rate [day$^{-1}$]"
                break

            # ice temperature
            elif self.datatype.startswith("Ti"):
                print(
                    "\nLoading ice temperature in file "
                    + self.directory
                    + "/Ti{}_{}_{}_{}_{}.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    )
                )

                fic = open(
                    self.directory
                    + "/Ti{}_{}_{}_{}_{}.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    ),
                    "r",
                )
                self.data = np.loadtxt(fic)
                fic.close()
                self.name = "Ice temperature [K]"
                break

            # ice velocity vector
            elif self.datatype.startswith("u"):
                print(
                    "\nLoading ice velocity u in file "
                    + self.directory
                    + "/u{}_{}_{}_{}_{}.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    )
                )
                print(
                    "Loading ice velocity v in file "
                    + self.directory
                    + "/v{}_{}_{}_{}_{}.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    )
                )

                fic = open(
                    self.directory
                    + "/u{}_{}_{}_{}_{}.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    ),
                    "r",
                )
                data_u = np.loadtxt(fic)
                fic.close()

                fic = open(
                    self.directory
                    + "/v{}_{}_{}_{}_{}.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    ),
                    "r",
                )
                data_v = np.loadtxt(fic)
                fic.close()

                formated_data_u = self._velocity_format(data_u, "u")
                formated_data_v = self._velocity_format(data_v, "v")

                self.data = np.stack((formated_data_u, formated_data_v), axis=2)
                self.name = "Ice velocity [m/s]"
                break

            # to exit manually
            elif self.datatype.startswith(("exit", "quit")):
                raise SystemExit("\nYou exited properly.")

            # error if not entered correctly
            else:
                print(
                    "You did not entered a valid option, please try again!\nValid inputs are: A, h, u, ti, dedt, exit."
                )

                if n:
                    self.datatype = None

                else:
                    raise SystemExit()

        return self.data

    def _velocity_format(self, raw_vel: np.ndarray, vel_type: str) -> np.ndarray:
        """
        Function that formats the velocity data. Note that only the velocities needs
        formating, since we are on an Arakawa C-grid.
        Simple linear regression between the two sides to interpolate in the center.
                ____o____
                |q_{i,j}|
        u_{i,j} o   o   o       where q is A, h, T, p
                |       |
                ----o----
                 v_{i,j}

        Args:
            raw_vel (np.ndarray): data velocity
            vel_type (str): component examined (u or v)

        Returns:
            (ndarray): formated velocities
        """

        # if data is the x component
        if vel_type.startswith("u"):
            formated_vel = (raw_vel[:, :-1] + raw_vel[:, 1:]) / 2

        # if data is the y component
        if vel_type.startswith("v"):
            formated_vel = (raw_vel[:-1, :] + raw_vel[1:, :]) / 2

        return formated_vel

    def _deformation_format(self, raw_def: np.ndarray) -> np.ndarray:
        """
        Function that formats the deformation 2D array. It takes all the -999 and -888 values and replaces them with NaN values.

        Args:
            raw_def (np.ndarray): raw deformation 2D array

        Returns:
            (ndarray): formated 2D array of same shape
        """

        ix = np.isin(raw_def, [-999, -888])
        vc = np.vectorize(lambda x: np.NaN if x == -999 else np.NaN)
        formated_def = np.where(ix, vc(raw_def), raw_def)

        return formated_def

    def load(
        self,
        directory: str = None,
        time: str = None,
        expno: str = None,
        datatype: str = None,
    ) -> np.ndarray:
        """
        Takes directory, experience number, data type, and snapshot date to load the proper data file, but as input from the user, not as arguments. Works for only one sheet of data. Also defines resolution.

        Args:
            directory (str, optional): directory where the ouputs are. Defaults to None.

            time (str, optional): time you are interested in (this have to be one of the snapshot you took before). Defaults to None.

            expno (str, optional): experience number. Defaults to None.

            datatype (str, optional): data type you want to load. Defaults to None.

        Returns:
            (ndarray): numpy array of formated data
        """

        # load all necessary parts, order is important.
        self._load_directory(directory)
        self._load_time(time)
        self._load_expno(expno)
        self._load_datatype(datatype)

        print("Done\n")

        # data size
        self.nx = self.data.shape[1]
        self.ny = self.data.shape[0]

        if self.resolution is None:
            # resolution computation
            if self.nx * self.ny == 520 * 440 or self.nx * self.ny == 518 * 438:
                self.resolution = 10

            elif self.nx * self.ny == 260 * 220 or self.nx * self.ny == 258 * 218:
                self.resolution = 20

            elif self.nx * self.ny == 130 * 110 or self.nx * self.ny == 128 * 108:
                self.resolution = 40

            elif self.nx * self.ny == 65 * 55 or self.nx * self.ny == 63 * 53:
                self.resolution = 80

            else:
                raise SystemExit(
                    "Resolution is invalid, your data is not properly formated.\n"
                )

        return self.data

    def multi_load(self, dt: str, time_end: str) -> np.ndarray:
        """
        Function that loads multiple data sheets from time given in class definition and store them in along a new axis where each sheet corresponds to a new time. For example, ice concentration is on a 2D grid, and the third axis would be for each timeframe. Therefore it has shape (ny, nx, nt).

        Args:
            dt (str): incrementation, dd-hh-mm
            time_end (str): last element of the list, yyyy-mm-dd-hh-mm

        Returns:
            (ndarray): size is (ny, nx, nt)
        """

        # Compute time array
        time_stamps = self._get_times(dt, time_end)

        # load and append all data sets into one list
        datalist = [self.load(time=t) for t in time_stamps]

        # create and return a stack along new axis
        return np.stack(datalist, axis=-1)

    def _get_times(self, dt: str, time_end: str) -> list:
        """
        Function that creates the values of the time array when loading multiple frames of data. It takes the initial time, the final time and the end time, and computes all times in between. It computes from time given in class definition.

        Args:
            dt (str): incrementation, dd-hh-mm
            time_end (str): last element of the list, yyyy-mm-dd-hh-mm

        Returns:
            (list): all time stamps needed for loading multiple sheets of data
        """

        # Convert input strings into datetime
        time_ini = datetime.strptime(self.time, "%Y-%m-%d-%H-%M")
        time_end = datetime.strptime(time_end, "%Y-%m-%d-%H-%M")

        dtlist = [int(n) for n in dt.split("-") if n.isdigit()]
        dt = timedelta(days=dtlist[0], hours=dtlist[1], minutes=dtlist[2])

        if dtlist[0]:
            time_stamps = [
                datetime.strftime(time_ini + timedelta(days=x), "%Y-%m-%d-%H-%M")
                for x in range(0, int(abs(time_end - time_ini).days), dtlist[0])
            ]

        elif dtlist[1]:
            time_stamps = [
                datetime.strftime(time_ini + timedelta(hours=x), "%Y-%m-%d-%H-%M")
                for x in range(
                    0,
                    int(abs(time_end - time_ini).total_seconds() / 3600),
                    dtlist[1],
                )
            ]

        elif dtlist[2]:
            time_stamps = [
                datetime.strftime(time_ini + timedelta(minutes=x), "%Y-%m-%d-%H-%M")
                for x in range(
                    0,
                    int(abs(time_end - time_ini).total_seconds() / 60),
                    dtlist[2],
                )
            ]

        time_stamps.append(datetime.strftime(time_end, "%Y-%m-%d-%H-%M"))

        return time_stamps

    def _coordinates(self, x0: np.ndarray, y0: np.ndarray) -> np.ndarray:
        """
        Function that computes the latitude and longitude of the grid data using a moving cone inside (see book 1, page 147-148).

        Args:
            x0 (np.ndarray): x distance in the tangent plane from the north pole
            y0 (np.ndarray): y distance in the tangent plane from the north pole

        Returns:
            np.ndarray: returns lon, lat in degrees
        """
        # convert to matrix
        x = np.broadcast_to(x0, (len(y0), len(x0)))
        y = np.broadcast_to(y0, (len(x0), len(y0))).T

        # polar coordinates on the plane
        r = np.sqrt((x) ** 2 + (y) ** 2)
        lon = np.degrees(np.arctan2(y, x)) + self.BETA

        # angle of the cone
        tan_theta = r / (2 * self.R_EARTH)
        # short radius on sphere
        rs = 2 * self.R_EARTH * tan_theta / (1 + tan_theta ** 2)

        lat = np.degrees(np.arccos(rs / self.R_EARTH))

        return lon, lat

    def _velolicty_vector_magnitude(
        self, formated_vel_u: np.ndarray, formated_vel_v: np.ndarray
    ) -> np.ndarray:
        """
        Computes the velocity vector magnitude. Replaces 0 by NaNs to be sure to not divide by 0.

        Args:
            formated_vel_u (np.ndarray): formated velocity x component
            formated_vel_v (np.ndarray): formated velocity y component

        Returns:
            (ndarray): sqrt(u**2 + v**2)
        """

        return np.where(
            np.sqrt(formated_vel_u ** 2 + formated_vel_v ** 2) == 0,
            np.NaN,
            np.sqrt(formated_vel_u ** 2 + formated_vel_v ** 2),
        )

    def _deformation(self, formated_data: np.ndarray) -> np.ndarray:
        """
        Function that computes deformation rates from velocities.

        Args:
            formated_data (np.ndarray): array of size (ny , nx, 2, nt) of all the data in u and v for many dates or time averages.

        Returns:
            np.ndarray: array of size (ny, nx, nt) of all the deformation rates.
        """
        # assign velocities
        u = formated_data[:, :, 0, ...]
        v = formated_data[:, :, 1, ...]

        # computes mean gradients
        du = u[1:, :] - u[:-1, :]
        dv = v[:, 1:] - v[:, :-1]
        dx, dy = 10e3, 10e3

        # put back values in center of grid cells
        du = self._velocity_format(du, "u")
        dv = self._velocity_format(dv, "v")

        # computes strain invariants
        divergence = du / dx + dv / dy
        shear = np.sqrt((du / dx - dv / dy) ** 2 + (du / dy - dv / dx) ** 2)
        deformations = np.sqrt(shear ** 2 + divergence ** 2)

        return deformations * 86400