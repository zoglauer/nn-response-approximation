"""

This python script splits simulation data into intervals of compton scatter angles and transforms it into healpix form.

For example, given a Compton resolution of 5 degrees and healpix parameters, this script will output cross sections of the data in intervals of 5 degrees and in healpix format.

"""

import numpy as np
import healpy as hp
from math import pi, sin, cos
import pickle
import os
import platform
import matplotlib.pyplot as plt


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
        # Note that there was an event reconstructed to this healpix index

        # Place the x and y values into its correct compton interval array

        closest_interval[heal_index] += 1

        if i % 100000 == 0:
            print(f"---> { i } values loaded")

    print(f"---> All values loaded.")

    # Return the x and y arrays
    return {"x": x, "y": list(split_data.values())}


# Denoises the data manually
# Removes pixels whose neighbors fall below an energy threshold
def denoise(cone, THRESHOLD):
    result = []

    for cross_sec in cone:
        denoised_cross_sec = []

        for i in range(len(cross_sec) - 1):
            if cross_sec[i - 1] < THRESHOLD and cross_sec[i + 1] < THRESHOLD:
                denoised_cross_sec.append(0)
            else:
                denoised_cross_sec.append(cross_sec[i])

        # Add last element
        denoised_cross_sec.append(cross_sec[-1])

        result.append(denoised_cross_sec)

    return np.asarray(result)


# Converts a cone to cartesian coordinates
# Removes pixels whose neighbors fall below an energy threshold
def convert_to_cartesian(cone, x_dim, y_dim):
    # Disable showing of any plots
    plt.ioff()

    result = []

    for cross_sec in cone:
        # Create dummy figure so can manually close later and save memory
        fig, (ax1) = plt.subplots(ncols=1)

        plt.axes(ax1)

        cart_arr = hp.cartview(
            cross_sec, xsize=x_dim, ysize=y_dim, return_projected_map=True, hold=True
        )

        # IMPORTANT: needed for eliminating excess use of CPU memory for storing figures
        plt.close(fig)

        result.append(cart_arr)

    return np.asarray(result)


def save_cross_sec_data(
    INPUT_DIR,
    OUTPUT_DIR,
    NSIDE,
    NUMPIX,
    COMPTON_RESOLUTION_DEG,
    DENOISE,
    DENOISE_THRESHOLD,
    OVERWRITE=True,
    CARTESIAN=False,
    x_dim=None,
    y_dim=None,
):
    # If directory not existing yet, create it.
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Loop through each file in the input dir
    for filename in os.listdir(INPUT_DIR):
        # Load file with pickle
        inp_path = os.path.join(INPUT_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename)

        # If not pickle file, skip it
        if ".pkl" not in filename:
            print("Non-pickle file found in data")
            continue

        # If overwrite set to false and file already exists, skip it
        if not OVERWRITE and os.path.exists(out_path):
            print(f"SKIPPING { inp_path }")
            continue

        print(inp_path)

        f = open(inp_path, "rb")
        data = pickle.load(f)
        f.close()

        arr = np.array(data, dtype=object)

        print(f"Loaded file { filename }. Array shape: { arr.shape }")

        split_data = create_cross_sec(arr, NSIDE, NUMPIX, COMPTON_RESOLUTION_DEG)

        # Denoise cone if specified
        if DENOISE:
            split_data["y"] = denoise(split_data["y"], DENOISE_THRESHOLD)

        # Convert to cartesian before saving if specified
        if CARTESIAN:
            split_data["y"] = convert_to_cartesian(split_data["y"], x_dim, y_dim)

        # Save data split into different cross sections
        with open(out_path, "wb") as handle:
            pickle.dump(split_data, handle)


if __name__ == "__main__":
    NSIDE = 128
    NUMPIX = 12 * NSIDE**2
    COMPTON_RESOLUTION_DEG = 10

    DENOISE = False
    DENOISE_THRESHOLD = 50

    # If savio, point to scratch directory
    if platform.system() == "Linux":
        INPUT_DIR = "/global/scratch/users/akotamraju/data/big-sim-data"
        OUTPUT_DIR = (
            "/global/scratch/users/akotamraju/data/cross-sec-big-sim-data-128-healpix"
        )
    else:
        INPUT_DIR = "../../data/raw-big-sim-data"
        OUTPUT_DIR = "../../data/cross-sec-big-noisy-128-healpix"

    save_cross_sec_data(
        INPUT_DIR,
        OUTPUT_DIR,
        NSIDE,
        NUMPIX,
        COMPTON_RESOLUTION_DEG,
        DENOISE,
        DENOISE_THRESHOLD,
        OVERWRITE=True,
        CARTESIAN=False,
        x_dim=1024,
        y_dim=768,
    )
