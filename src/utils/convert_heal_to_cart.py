from create_cross_sec import convert_to_cartesian
import os
import pickle


# Reads in an existing healpix cone cross section directory and creates a cartesian version
def convert_existing_healpix_to_cartesian(INPUT_DIR, OUTPUT_DIR, x_dim, y_dim):
    # If directory not existing yet, create it.
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Loop through each file in the input dir
    for filename in os.listdir(INPUT_DIR):
        f = open(filename, "rb")
        data = pickle.load(f)
        f.close()

        data["y"] = convert_to_cartesian(data["y"], x_dim, y_dim)

        # Save data split into different cross sections
        with open(OUTPUT_DIR, "wb") as handle:
            pickle.dump(data, handle)


if __name__ == "__main__":
    INPUT_PATH = (
        f"/global/scratch/users/akotamraju/data/cross-sec-big-sim-data-16-healpix"
    )
    OUTPUT_PATH = (
        f"/global/scratch/users/akotamraju/data/cross-sec-big-sim-data-16-cartesian"
    )
    x_dim = 128
    y_dim = 96
    convert_existing_healpix_to_cartesian(INPUT_PATH, OUTPUT_PATH, x_dim, y_dim)
