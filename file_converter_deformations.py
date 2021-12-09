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


folder = "RGPS_binary_deformations"

filenames_SHR = list_files(folder, "SHR")
filenames_DIV = list_files(folder, "DIV")

shear = np.zeros((264, 248, np.size(filenames_SHR)))
div = np.zeros((264, 248, np.size(filenames_DIV)))

for k in range(len(filenames_SHR)):

    with open(path.join(folder, filenames_SHR[k]), "rb") as file:
        data_SHR = np.fromfile(file, dtype=">f", count=-1)
    with open(path.join(folder, filenames_DIV[k]), "rb") as file:
        data_DIV = np.fromfile(file, dtype=">f", count=-1)

    # first lines are useless (data information)
    a_SHR = data_SHR[24:]
    a_DIV = data_DIV[24:]

    # fortran order is needed
    b_SHR = a_SHR.reshape(264, 248, order="F").copy()
    b_DIV = a_DIV.reshape(264, 248, order="F").copy()

    for i in range(264):
        for j in range(248):
            shear[i, j, k] = b_SHR[i, j]
            div[i, j, k] = b_DIV[i, j]

# swap continents and no data by NaNs
shear = np.where(shear >= 1e10, np.NaN, shear)
div = np.where(div >= 1e10, np.NaN, div)

# final touches
shear = np.transpose(shear[:248, :, :], (1, 0, 2))
div = np.transpose(div[:248, :, :], (1, 0, 2))

np.save("RGPS_derivatives/SHR.npy", shear)
np.save("RGPS_derivatives/DIV.npy", div)
