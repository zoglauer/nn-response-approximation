import random
import math

def randomAngles():
    # Generate two random values between 0 and 1 to be used for omega and theta.
    randomTheta = random.uniform(0, 1)
    randomPhi = random.uniform(0, 1)

    pi = 4 * math.atan(1)

    theta = 180 / pi * math.acos(1 - 2 * randomTheta)
    phi = 360 * randomPhi

    return str(theta) + "  " + str(phi)

if __name__ == "__main__":
    print(randomAngles())
