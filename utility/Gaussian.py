# Python Standard Libraries
import math

# Third Party Libraries
import numpy as np



class Gaussian:
    """
    This is a utility class for handling Gaussian related methods
    """

    def __init__(self):
        raise TypeError("The class 'Gaussian' should not be instantiated as " +
                        "its sole purpose is to provide helper method")

    @staticmethod
    def Gauss3D(X, x0, y0, R):
        '''
        Return a donghnut with a Gaussian shape

        Args:
          X (float, float): Tuple of the x, y position
          x0 (float): x center
          y0 (float): x center
          R (float):  radius
        '''
        ###### Unfinished/Unpolished
        x, y = X
        return np.exp(((-np.sqrt((x - x0) ** 2 + (y - y0) ** 2) - R) ** 2) / self.gSigmaR)

    @staticmethod
    def getGauss(d, sigma=1):
        '''
        Return a 1D Gaussian value

        Args:
          d (float):      Distance from 0
          sigma (float):  Sigma value of Gaussian

        '''
        return 1 / (sigma * math.sqrt(2 * np.pi)) * math.exp(-0.5 * pow(d / sigma, 2))