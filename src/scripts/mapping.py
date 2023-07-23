import healpy as hp
import numpy as np
import pickle


def save_mapping(NSIDE, x_dim, y_dim):
    healpix_arr = np.array(list(range(NSIDE**2 * 12)))

    cart_arr = hp.cartview(
        healpix_arr, xsize=2048, ysize=1536, return_projected_map=True
    )

    cart_to_heal_map = {}

    # Go through each row
    for r in range(len(cart_arr)):
        # Go through each column
        for c in range(len(cart_arr[0])):
            pos = (r, c)

            # Set the (row, column) combination to the healpix pixel value
            cart_to_heal_map[pos] = int(cart_arr[r][c])

        if r % 20 == 0:
            print("Row ", r, " completed")

    with open(f"../../mappings/NSIDE_{NSIDE}_x{x_dim}_y{y_dim}.pickle", "wb") as f:
        pickle.dump(cart_to_heal_map, f)


# Using a position map, reconstruct a cartesian array to healpix
def reconstruct_from_cartesian(cart_arr, NSIDE, cart_to_heal_map):
    NUMPIX = NSIDE**2 * 12

    reconstructed = [0] * NUMPIX

    for pos, value in cart_to_heal_map.items():
        r, c = pos
        reconstructed[value] = cart_arr[r][c]

    reconstructed = np.array(reconstructed)

    return reconstructed


if __name__ == "__main__":
    save_mapping(NSIDE=128, x_dim=2048, y_dim=1536)
