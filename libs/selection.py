# ----------------------------------------------------------------------
#   Data selection module
# ----------------------------------------------------------------------
#   Its purpose is to fetch, load, and format the data from the
#   output files that I previously downloaded. There are many
#   options in this module. The long side is x, for 10 km shape is (438,#   518)
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
#   -combine load and multi_load so that you can call only 1 module to
#   load anytype of data.
# ----------------------------------------------------------------------

from cmath import sin
import numpy as np
import netCDF4 as nc
from os import path, listdir
from datetime import datetime, timedelta
from libs.constants import *


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
            "Welcome in the data loader routine! To cancel selection, enter exit or quit."
        )

        self.directory = directory
        self.time = time
        self.time_init = time
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
                    raise SystemExit(
                        "\nError in directory name, it does not exist."
                    )

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
                self.time = input(
                    "At what time? (format is yyyy-mm-dd-hh-mm) "
                ).split("-")
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

    def _load_datatype(
        self, datatype: str, vel_to_def: bool = 0, choice: int = 0
    ) -> np.ndarray:
        """
        Loads proper data type.
        Needs directory, time and expno to have been previously load to work.

        Args:
            datatype (str): data type to load

            vel_to_def (bool, optional): is we compute velocity to deformation or just velocity

            choice (int, optional): when computing vel_to_def, choice for which deformation to compute.

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

            # ice damage
            if self.datatype.startswith("dam"):
                print(
                    "\nLoading ice damage in file "
                    + self.directory
                    + "/dam{}_{}_{}_{}_{}.{}".format(
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
                    + "/dam{}_{}_{}_{}_{}.{}".format(
                        self.time[0],
                        self.time[1],
                        self.time[2],
                        self.time[3],
                        self.time[4],
                        self.expno,
                    ),
                    "r",
                )
                dam_raw = np.loadtxt(fic)
                self.data = self._deformation_format(dam_raw)
                fic.close()
                self.name = "Ice damage"
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
                viscosity_raw = np.loadtxt(fic)
                self.data = self._deformation_format(viscosity_raw)
                fic.close()
                self.name = "Viscosity [N$\cdot$day$\cdot$m$^{-1}$]"
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

                if vel_to_def:
                    formated_data_u = self._velocity_format(data_u, "u")
                    formated_data_v = self._velocity_format(data_v, "v")
                    du, dv = self._derivative(formated_data_u, formated_data_v)
                    self.data = self._deformation(du, dv, choice)

                    if choice == 0:
                        self.name = "Ice deformation rate [day$^{-1}$]"
                        self.datatype = "dedt"

                    elif choice == 1:
                        self.name = "Shear rate [day$^{-1}$]"
                        self.datatype = "shear"

                    elif choice == 2:
                        self.name = "Divergence rate [day$^{-1}$]"
                        self.datatype = "divergence"

                else:
                    formated_data_u = self._velocity_format(data_u, "u")
                    formated_data_v = self._velocity_format(data_v, "v")

                    self.data = np.stack(
                        (formated_data_u, formated_data_v), axis=2
                    )
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

        print("Done")
        return self.data

    def _velocity_format(
        self, raw_vel: np.ndarray, vel_type: str
    ) -> np.ndarray:
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
        vel_to_def: bool = 0,
        choice: int = 0,
    ) -> np.ndarray:
        """
        Takes directory, experience number, data type, and snapshot date to load the proper data file, but as input from the user, not as arguments. Works for only one sheet of data. Also defines resolution.

        Args:
            directory (str, optional): directory where the ouputs are. Defaults to None.

            time (str, optional): time you are interested in (this have to be one of the snapshot you took before). Defaults to None.

            expno (str, optional): experience number. Defaults to None.

            datatype (str, optional): data type you want to load. Defaults to None.

            vel_to_def (bool, optional): is we compute velocity to deformation or just velocity

            choice (int, optional): when computing vel_to_def, choice for which deformation to compute.

        Returns:
            (ndarray): numpy array of formated data
        """
        if time is None:
            time = self.time_init

        if vel_to_def:
            self.datatype = "u"

        # load all necessary parts, order is important.
        self._load_directory(directory)
        self._load_time(time)
        self._load_expno(expno)
        self._load_datatype(datatype, vel_to_def=vel_to_def, choice=choice)

        # data size
        self.nx = self.data.shape[1]
        self.ny = self.data.shape[0]

        if self.resolution is None:
            # resolution computation
            if (
                self.nx * self.ny == 520 * 440
                or self.nx * self.ny == 518 * 438
            ):
                self.resolution = 10

            elif (
                self.nx * self.ny == 260 * 220
                or self.nx * self.ny == 258 * 218
            ):
                self.resolution = 20

            elif (
                self.nx * self.ny == 130 * 110
                or self.nx * self.ny == 128 * 108
            ):
                self.resolution = 40

            elif self.nx * self.ny == 65 * 55 or self.nx * self.ny == 63 * 53:
                self.resolution = 80

            else:
                raise SystemExit(
                    "Resolution is invalid, your data is not properly formated.\n"
                )

        return self.data

    def multi_load(
        self,
        dt: str,
        time_end: str,
        directory: str = None,
        expno: str = None,
        datatype: str = None,
        vel_to_def: bool = 0,
        choice: int = 0,
    ) -> np.ndarray:
        """
        Function that loads multiple data sheets from time given in class definition and store them in along a new axis where each sheet corresponds to a new time. For example, ice concentration is on a 2D grid, and the third axis would be for each timeframe. Therefore it has shape (ny, nx, nt).

        Args:
            dt (str): incrementation, dd-hh-mm

            time_end (str): last element of the list, yyyy-mm-dd-hh-mm

            vel_to_def (bool, optional): if we compute velocity to deformation or just velocity

            choice (int, optional): when computing vel_to_def, choice for which deformation to compute.

        Returns:
            (ndarray): size is (ny, nx, nt)
        """
        # Compute time array
        time_stamps = self._get_times(dt, time_end)

        # load and append all data sets into one list
        datalist = [
            self.load(
                directory=directory,
                time=t,
                expno=expno,
                datatype=datatype,
                vel_to_def=vel_to_def,
                choice=choice,
            )
            for t in time_stamps
        ]

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
        time_ini = datetime.strptime(self.time_init, "%Y-%m-%d-%H-%M")
        time_end = datetime.strptime(time_end, "%Y-%m-%d-%H-%M")

        dtlist = [int(n) for n in dt.split("-") if n.isdigit()]
        dt = timedelta(days=dtlist[0], hours=dtlist[1], minutes=dtlist[2])

        if dtlist[0]:
            time_stamps = [
                datetime.strftime(
                    time_ini + timedelta(days=x), "%Y-%m-%d-%H-%M"
                )
                for x in range(
                    0, int(abs(time_end - time_ini).days), dtlist[0]
                )
            ]

        elif dtlist[1]:
            time_stamps = [
                datetime.strftime(
                    time_ini + timedelta(hours=x), "%Y-%m-%d-%H-%M"
                )
                for x in range(
                    0,
                    int(abs(time_end - time_ini).total_seconds() / 3600),
                    dtlist[1],
                )
            ]

        elif dtlist[2]:
            time_stamps = [
                datetime.strftime(
                    time_ini + timedelta(minutes=x), "%Y-%m-%d-%H-%M"
                )
                for x in range(
                    0,
                    int(abs(time_end - time_ini).total_seconds() / 60),
                    dtlist[2],
                )
            ]

        time_stamps.append(datetime.strftime(time_end, "%Y-%m-%d-%H-%M"))

        return time_stamps

    def _coordinates(
        self, x0: np.ndarray, y0: np.ndarray, RGPS: bool = False
    ) -> np.ndarray:
        """
        Function that computes the latitude and longitude of the grid data using a moving cone inside (see book 1, page 147-148) and (see book 2, page 22 for RGPS).

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

        if not RGPS:
            lon = np.degrees(np.arctan2(y, x)) + BETA

            # angle of the cone
            tan_theta = r / (2 * R_EARTH)
            # short radius on sphere
            rs = 2 * R_EARTH * tan_theta / (1 + tan_theta ** 2)

            lat = np.degrees(np.arccos(rs / R_EARTH))

        elif RGPS:
            lon = np.degrees(np.arctan2(y, x)) + BETA_RGPS

            # small radius corresponding to plane at phi = 70
            rhat = R_EARTH * np.cos(np.pi / 2 - np.radians(PLANE_RGPS))
            lat = np.degrees(np.pi / 2 - np.arctan(r / rhat))

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

    def _deformation(self, du: np.ndarray, choice: int) -> np.ndarray:
        """
        Function that computes deformation rates from velocity derivatives.

        Args:
            du (np.ndarray): array of size (ny, nx, nt, 4) of all the data in du for many dates or time averages.
            choice (int): number 0, 1, 2 to choose between dedt, shear, divergence.

        Returns:
            np.ndarray: array of size (ny, nx, nt) of all the deformation rates in [day^-1].
        """
        # computes strain invariants (watch out for inverse derivatives 1: x-axis, 0: y-axis)
        div = du[..., 0] + du[..., 3]
        shear = np.sqrt(
            (du[..., 0] - du[..., 3]) ** 2 + (du[..., 1] + du[..., 2]) ** 2
        )

        if choice == 0:
            return np.sqrt(div ** 2 + shear ** 2)

        elif choice == 1:
            return shear

        elif choice == 2:
            return div

    def _derivative(
        self, u: np.ndarray, v: np.ndarray, scale: int = 1
    ) -> np.array:
        """
        Function that computes derivatives from velocities.

        Args:
            u (v) (np.ndarray): array of size (ny, nx, nt) of all the data in u (v) for many dates or time averages.

        Returns:
            np.ndarray: array of size (2, ny, nx, nt) of all the derivatives. 0 is d/dx (y axis in the code), 1 is d/dy (x axis in the code). I just did not choose the right name in order to match the model.
        """
        # computes mean gradients
        dudx = np.gradient(u, self.resolution * 1000 * scale, axis=1) * 86400
        dudy = np.gradient(u, self.resolution * 1000 * scale, axis=0) * 86400
        dvdx = np.gradient(v, self.resolution * 1000 * scale, axis=1) * 86400
        dvdy = np.gradient(v, self.resolution * 1000 * scale, axis=0) * 86400

        return np.stack((dudx, dudy, dvdx, dvdy), axis=-1)

    def _delta(self, du: np.ndarray, dv: np.ndarray) -> np.array:
        """
        Function that computes the delta at any given time, from the velocity derivatives.

        Args:
            du (dv) (np.ndarray): array of size (2, ny, nx, nt) of all the data in du (dv) for many dates or time averages.

        Returns:
            delta (np.array): delta function for the given velocity gradients. Shape is (ny, nx, nt)
        """
        # compute the strain tensor components
        eps11 = du[1]
        eps12 = 1 / 2 * (du[0] + dv[1])
        eps22 = dv[0]

        # computes the delta function
        delta = np.sqrt(
            (eps11 ** 2 + eps22 ** 2) * (1 + 1 / E ** 2)
            + 4 / E ** 2 * eps12 ** 2
            + 2 * eps11 * eps22 * (1 - 1 / E ** 2)
        )

        return delta

    def _pressure(self, h: np.array, A: np.array) -> np.array:
        """
        Compute the ice strenght function.

        Args:
            h (np.array): Ice thickness, shape is (ny, nx, nt)
            A (np.array): Ice concentration, shape is (ny, nx, nt)

        Returns:
            p (np.array): ice strenght function, shape is (ny, nx, nt)
        """

        # compute the pressure
        p = P_STAR * h * np.exp(-C * (1 - A))

        return p[1:-1, 1:-1]

    def _viscosity(self, p: np.array, delta: np.array) -> np.array:
        """
        Compute the maximum viscosity.

        Args:
            p (np.array): ice strenght function, shape is (ny, nx, nt)
            delta (np.array): delta function for the given velocity gradients. Shape is (ny, nx, nt)

        Returns:
            zeta (np.array): ice viscosity, shape is (ny, nx, nt)
            zeta_max (np.array): ice max viscosity, shape is (ny, nx, nt)
        """

        zeta_max = 2.5e8 * p
        zeta = zeta_max * np.tanh(p / (2 * delta * zeta_max))

        return zeta, zeta_max

    def nc_load(self, file: str, all: bool = 0, ti: int = 0, tf: int = 90):
        """
        Loads pertinent data from RGPS.

        Args:
            file (str): what is the name of the file we are loading from.
            all (bool, optional): whether to load all times or only the months of January February and March. Defaults to 0
            ti (int, optional): extent in time in days from january first. Defaults to 0.
            tf (int, optional): extent in time in days from january first. Defaults to 90.

        Returns:
            [type]: wanted data
        """
        ds = nc.Dataset(file)

        if not all:
            # indices for the period of interest
            indices = (ds["time"][:] / 60 / 60 / 24 >= ti) & (
                ds["time"][:] / 60 / 60 / 24 <= tf
            )

            div = np.flip(np.transpose(ds["divergence"][:], (1, 2, 0)), axis=0)
            div = div[..., indices]
            # div = np.where(np.abs(div) < 5e-3, np.NaN, div)

            shear = np.flip(np.transpose(ds["shear"][:], (1, 2, 0)), axis=0)
            shear = shear[..., indices]
            # shear = np.where(shear < 5e-3, np.NaN, shear)

            deps = np.sqrt(div ** 2 + shear ** 2)
            deps.mask = 0
            deps = np.where(deps == 0.0, np.NaN, deps)

            print("Done loading RGPS data.")
            print(
                "Time list is between:\n {} and {} days of the year.".format(
                    ti, tf
                )
            )

        else:
            # all months don't need to care about 0.005 treshold, just want to know where there is data this is for the mask.
            # indices for the period of interest
            indices = (ds["time"][:] / 60 / 60 / 24 >= ti) & (
                ds["time"][:] / 60 / 60 / 24 <= tf
            )

            if not indices.any():
                indices = (ds["time"][:] / 60 / 60 / 24 - 365 >= ti) & (
                    ds["time"][:] / 60 / 60 / 24 - 365 <= tf
                )

            if len(ds["time"][indices] / 60 / 60 / 24) != len(
                set(ds["time"][indices] / 60 / 60 / 24)
            ):
                same_idx = []
                for i in range(len(ds["time"][indices])):
                    for j in range(len(ds["time"][indices])):
                        if (
                            ds["time"][indices][i] / 60 / 60 / 24
                            == ds["time"][indices][j] / 60 / 60 / 24
                        ):
                            if i != j:
                                same_idx.append([i, j])
                same_idx = np.asarray(same_idx)
                idx_num = len(same_idx) // 2
                position_to_kill = same_idx[:idx_num][..., 0]
                counter = 0
                for i in range(len(indices)):
                    if indices[i] == 1:
                        break
                    counter += 1

                indices[position_to_kill + counter] = 0

            print(ds["time"][indices] / 60 / 60 / 24)
            div = np.flip(np.transpose(ds["divergence"][:], (1, 2, 0)), axis=0)
            div = div[..., indices]

            shear = np.flip(np.transpose(ds["shear"][:], (1, 2, 0)), axis=0)
            shear = shear[..., indices]

            deps = np.sqrt(div ** 2 + shear ** 2)
            deps.mask = 0
            deps = np.where(deps == 0.0, np.NaN, deps)

        return deps, div, shear

    def mask80(
        self, directory: str, year: str = "no", ti: int = 0, tf: int = 90
    ):
        """
        Function that loads all files and computes the mask80, that is the mask that represent where there is a least 80% of data in time.

        Args:
            directory (str): directory where the RGPS data is.
            year (int, optional): take only 1 year. Format is 9899 for winter 98-99.

        Returns:
            mask 80 (array): array of size (248, 248) of where there is at least 80% temporal data presence. 
        """

        if year != "no":
            # create empty array to stock data
            mask80 = np.zeros((248, 248))

            # take only the wanted year
            temp_bool = ~np.isnan(
                self.nc_load(
                    directory + "/w" + year + "n_3dys.nc", all=1, ti=ti, tf=tf
                )[0]
            )
            temp_sum = np.sum(temp_bool, axis=-1)
            temp = np.where(temp_sum / temp_bool.shape[-1] >= 0.8, 1, 0)
            mask80 = temp

        else:
            # list all .nc files in directory
            files_list = self._list_files(directory, "nc")

            # create empty array to stock data
            mask_array = np.zeros((248, 248, len(files_list)))

            i = 0
            size = 0
            # loop on file name
            for file in files_list:
                temp_bool = ~np.isnan(
                    self.nc_load(directory + "/" + file, all=1, ti=ti, tf=tf)[
                        0
                    ]
                )
                temp_sum = np.sum(temp_bool, axis=-1)
                size += temp_bool.shape[-1]
                mask_array[..., i] = temp_sum
                i += 1

            # sum all layers and divide by the number of layer to test which cells are covered for a least 80% of the time
            mask_sum = np.sum(mask_array, axis=-1)
            mask80 = np.where(mask_sum / size >= 0.8, 1, np.NaN)
            print("Done loading RGPS80 mask.")

        return mask80

    def mask80_times_RGPS(
        self, data: np.ndarray, mask80: np.ndarray
    ) -> np.ndarray:
        """
        Function that multiplies data and mask.

        Args:
            data (np.ndarray): data to mask.
            mask80 (np.ndarray): mask to apply

        Returns:
            np.ndarray: masked data.
        """
        if len(data.shape) == 3:
            data80 = np.transpose(
                np.transpose(data, (2, 0, 1)) * mask80, (1, 2, 0)
            )
        else:
            data80 = data * mask80

        return data80

    def mask80_times(self, data: np.ndarray, mask80: np.ndarray) -> np.ndarray:
        """
        Function that multiplies data and mask.

        Args:
            data (np.ndarray): data to mask.
            mask80 (np.ndarray): mask to apply

        Returns:
            np.ndarray: masked data.
        """
        # we don't really care about the exact values of the vectors, but more about their size. For example, a proper way of doing this we be to either have a grid that is one more in each directions (they correspond to corners of boxes) or to have a grid with values that are moved by half the box size (in order to be in the center). But since we don't care about the values we will do a mix.
        import scipy.interpolate as sci
        from scipy.ndimage.interpolation import rotate, shift, zoom

        theta = BETA - BETA_RGPS

        mask = np.zeros((data.shape[0], data.shape[1]))

        x0_SIM = np.arange(data.shape[1]) * self.resolution - 2500
        y0_SIM = np.arange(data.shape[0]) * self.resolution - 2250

        x0_RGPS_SIM = np.arange(mask80.shape[0]) * RES_RGPS - 2300
        y0_RGPS_SIM = np.arange(mask80.shape[1]) * RES_RGPS - 1000

        x0_RGPS_10_SIM = (
            np.arange(int(mask80.shape[0] * RES_RGPS / self.resolution))
            * self.resolution
            - 2300
        )
        y0_RGPS_10_SIM = (
            np.arange(int(mask80.shape[1] * RES_RGPS / self.resolution))
            * self.resolution
            - 1000
        )

        # +1 to match shape of x_RGPS_10
        id_xmin = np.int(
            (
                np.abs(np.min(x0_SIM) - np.min(x0_RGPS_10_SIM))
                // self.resolution
            )
        )
        id_xmax = np.int(
            (
                np.abs(np.min(x0_SIM) - np.max(x0_RGPS_10_SIM))
                // self.resolution
                + 1
            )
        )

        id_ymin = np.int(
            (
                np.abs(np.min(y0_SIM) - np.min(y0_RGPS_10_SIM))
                // self.resolution
            )
        )
        id_ymax = np.int(
            (
                np.abs(np.min(y0_SIM) - np.max(y0_RGPS_10_SIM))
                // self.resolution
                + 1
            )
        )

        mask80 = np.where(np.isnan(mask80) == 1, 0, mask80)
        interp = sci.RectBivariateSpline(y0_RGPS_SIM, x0_RGPS_SIM, mask80)
        mask_RGPS_10 = np.round(interp(y0_RGPS_10_SIM, x0_RGPS_10_SIM))
        mask_RGPS_10 = zoom(mask_RGPS_10, zoom=1.7)
        mask_RGPS_10 = np.round(shift(mask_RGPS_10, shift=(75, -55)))
        mask_RGPS_10 = np.round(
            rotate(mask_RGPS_10, angle=theta, reshape=False)
        )
        mask_RGPS_10 = np.round(shift(mask_RGPS_10, shift=(-75, 55)))
        mask_RGPS_10 = np.round(zoom(mask_RGPS_10, zoom=1 / 1.7))
        mask_RGPS_10 = np.transpose(
            np.where(mask_RGPS_10 == 0, np.NaN, mask_RGPS_10)
        )

        for i in range(id_xmin, id_xmax):
            for j in range(id_ymin, id_ymax):
                mask[j, i] = mask_RGPS_10[i - id_xmin, j - id_ymin]
        mask = np.where(mask == 0, np.NaN, mask)

        data80 = np.transpose(np.transpose(data, (2, 0, 1)) * mask, (1, 2, 0))

        return data80, np.transpose(mask_RGPS_10)

