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


# Denoises an image/cartesian grid
# Only keeps pixels who have 2 top & 2 bottom neighbors OR 2 left & 2 right neighbors
def denoise_img(img):
    new_img = []
    for r in range(len(img)):
        new_img.append([])

    for r in range(len(img)):
        for c in range(len(img[0])):
            if r >= len(img) - 2 or c >= len(img[0]) - 2:
                new_img[r].append(img[r][c])
                continue

            # if img[r][c] >= 1 and (img[r + 1][c] >= 1 and img[r - 1][c] >= 1) or (img[r][c + 1] >= 1 and img[r][c - 1] >= 1):
            #     new_img[r].append(img[r][c])
            if (
                img[r][c] >= 1
                and (
                    img[r + 1][c] >= 1
                    and img[r - 1][c] >= 1
                    and img[r + 2][c] >= 1
                    and img[r - 2][c] >= 1
                )
                or (
                    img[r][c + 1] >= 1
                    and img[r][c - 1] >= 1
                    and img[r][c + 2] >= 1
                    and img[r][c - 2] >= 1
                )
            ):
                new_img[r].append(img[r][c])
            else:
                new_img[r].append(0)
    return new_img


def denoise_cone(cone):
    output = []
    for cross_sec in cone:
        output.append(denoise_img(cross_sec))
    return np.asarray(output)


# Denoises the data manually
# Removes pixels whose neighbors fall below an energy threshold
def denoise_old(cone, THRESHOLD):
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


# NOTE: Input dir must already have the data in cartesian form
def save_denoised_data(
    INPUT_DIR,
    OUTPUT_DIR,
    OVERWRITE=True,
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

        # Get the output image
        f = open(inp_path, "rb")
        data = pickle.load(f)["y"]
        f.close()

        print(data)

        arr = np.array(data)

        print(f"Loaded file { filename }. Array shape: { arr.shape }")

        denoised_cone = denoise_cone(arr)

        print(denoised_cone)

        # input is denoised data, output is noisy data
        data = {"x": denoised_cone, "y": arr}

        # Save data split into different cross sections
        with open(out_path, "wb") as handle:
            pickle.dump(data, handle)


if __name__ == "__main__":
    # If savio, point to scratch directory
    if platform.system() == "Linux":
        INPUT_DIR = "/global/scratch/users/akotamraju/data/noisy-128-cartesian-1024-768"
        OUTPUT_DIR = (
            "/global/scratch/users/akotamraju/denoised_data/128-cartesian-1024-768"
        )
    else:
        INPUT_DIR = "../../data/128-cartesian-1024-768"
        OUTPUT_DIR = "../../denoised_data/128-cartesian-1024-768"

    save_denoised_data(
        INPUT_DIR,
        OUTPUT_DIR,
        OVERWRITE=True,
    )
