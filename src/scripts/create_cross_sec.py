"""

This python script splits simulation data into intervals of compton scatter angles and transforms it into healpix form.

For example, given a Compton resolution of 5 degrees and healpix parameters, this script will output cross sections of the data in intervals of 5 degrees and in healpix format.

"""

import numpy as np
import healpy as hp
from math import pi, sin, cos
import pickle
import os


def deg_to_rad(ang):
    return ang * 2 * pi / 360.0


def create_cross_sec(arr, NSIDE, NUMPIX, COMPTON_RESOLUTION_DEG):
    # Create map to store data for each compton scatter angle interval
    split_data = {}

    x = np.array(arr[0])

    num_vals = len(arr[1])

    curr_angle = 0
    while curr_angle < 180:
        # Create placeholder for the y array
        y_placeholder = [0] * NUMPIX

        # Convert to numpy arrays
        y_placeholder = np.array(y_placeholder)

        # Input placeholder data into split_data map
        split_data[curr_angle] = y_placeholder

        # Increment the current angle by the compton resolution
        curr_angle += COMPTON_RESOLUTION_DEG

    # print(split_data)

    # Loop through each value in the array
    for i in range(num_vals):
        energy = arr[2][i]
        theta = arr[3][i]
        phi = arr[4][i]
        angle = arr[5][i]

        # Get which compton interval this datapoint fits into
        closest_interval_mod = angle % COMPTON_RESOLUTION_DEG
        closest_interval_angle = None

        closest_interval_angle = int(angle - closest_interval_mod)

        closest_interval = split_data[closest_interval_angle]

        # Convert angles to radians
        theta = deg_to_rad(theta)
        phi = deg_to_rad(phi)
        angle = deg_to_rad(angle)

        # Find which healpix index to place datapoint into
        heal_index = hp.ang2pix(NSIDE, theta, phi)

        # The y values for the model
        y_val = energy

        # Place the x and y values into its correct compton interval array
        closest_interval[heal_index] = y_val

        if i % 100000 == 0:
            print(f"---> { i } values loaded")

    print(f"---> All values loaded.")

    # Return the x and y arrays
    return {"x": x, "y": list(split_data.values())}


def save_cross_sec_data(INPUT_DIR, OUTPUT_DIR, NSIDE, NUMPIX, COMPTON_RESOLUTION_DEG):
    # Loop through each file in the input dir
    for filename in os.listdir(INPUT_DIR):
        # Load file with pickle
        inp_path = os.path.join(INPUT_DIR, filename)

        f = open(inp_path, "rb")
        data = pickle.load(f)
        f.close()

        arr = np.array(data, dtype=object)

        print(f"Loaded file { filename }. Array shape: { arr.shape }")

        split_data = create_cross_sec(arr, NSIDE, NUMPIX, COMPTON_RESOLUTION_DEG)

        # Save data split into different cross sections
        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "wb") as handle:
            pickle.dump(split_data, handle)


if __name__ == "__main__":
    NSIDE = 128
    NUMPIX = 12 * NSIDE**2
    COMPTON_RESOLUTION_DEG = 5
    INPUT_DIR = "../../data/full-sim-data"
    OUTPUT_DIR = "../../data/cross-sec-data"

    save_cross_sec_data(INPUT_DIR, OUTPUT_DIR, NSIDE, NUMPIX, COMPTON_RESOLUTION_DEG)
