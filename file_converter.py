import numpy as np
from os import path, listdir


def list_files(directory: str, extension: str) -> list:
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


folder = "RGPS_binary_derivatives"

filenames_DUDX = list_files(folder, "DUDX")
filenames_DUDY = list_files(folder, "DUDY")
filenames_DVDX = list_files(folder, "DVDX")
filenames_DVDY = list_files(folder, "DVDY")

dudx = np.zeros((264, 248, np.size(filenames_DUDX)))
dudy = np.zeros((264, 248, np.size(filenames_DUDY)))
dvdx = np.zeros((264, 248, np.size(filenames_DVDX)))
dvdy = np.zeros((264, 248, np.size(filenames_DVDY)))

for k in range(len(filenames_DUDX)):

    with open(path.join(folder, filenames_DUDX[k]), "rb") as file:
        data_DUDX = np.fromfile(file, dtype=">f", count=-1)
    with open(path.join(folder, filenames_DUDY[k]), "rb") as file:
        data_DUDY = np.fromfile(file, dtype=">f", count=-1)
    with open(path.join(folder, filenames_DVDX[k]), "rb") as file:
        data_DVDX = np.fromfile(file, dtype=">f", count=-1)
    with open(path.join(folder, filenames_DVDY[k]), "rb") as file:
        data_DVDY = np.fromfile(file, dtype=">f", count=-1)

    # first lines are useless
    a_DUDX = data_DUDX[24:]
    a_DUDY = data_DUDY[24:]
    a_DVDX = data_DVDX[24:]
    a_DVDY = data_DVDY[24:]

    # fortran order is needed
    b_DUDX = a_DUDX.reshape(264, 248, order="F").copy()
    b_DUDY = a_DUDY.reshape(264, 248, order="F").copy()
    b_DVDX = a_DVDX.reshape(264, 248, order="F").copy()
    b_DVDY = a_DVDY.reshape(264, 248, order="F").copy()

    for i in range(264):
        for j in range(248):
            dudx[i, j, k] = b_DUDX[i, j]
            dudy[i, j, k] = b_DUDY[i, j]
            dvdx[i, j, k] = b_DVDX[i, j]
            dvdy[i, j, k] = b_DVDY[i, j]

# swap continents and no data by NaNs
dudx = np.where(dudx >= 1e10, np.NaN, dudx)
dudy = np.where(dudy >= 1e10, np.NaN, dudy)
dvdx = np.where(dvdx >= 1e10, np.NaN, dvdx)
dvdy = np.where(dvdy >= 1e10, np.NaN, dvdy)

# final touches
dudx = np.transpose(dudx[:248, :, :], (1, 0, 2))
dudy = np.transpose(dudy[:248, :, :], (1, 0, 2))
dvdx = np.transpose(dvdx[:248, :, :], (1, 0, 2))
dvdy = np.transpose(dvdy[:248, :, :], (1, 0, 2))

np.save("RGPS_derivatives/DUDX.npy", dudx)
np.save("RGPS_derivatives/DUDY.npy", dudy)
np.save("RGPS_derivatives/DVDX.npy", dvdx)
np.save("RGPS_derivatives/DVDY.npy", dvdy)
