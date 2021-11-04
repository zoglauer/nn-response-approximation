# Base Python Libraries
import math
import random
import warnings

# Third Party Libraries
import numpy as np
import matplotlib.pyplot as plt

# Own Tools
from comptonspace.AbstractComptonSpace import AbstractComptonSpace
from utility.Gaussian import Gaussian



class CartesianSpace(AbstractComptonSpace):
    """
    This class represents a Compton Data Space that is implemented using Cartesian Coordinates
    """

    def __init__(self, gMinXY=-1, gMaxXY=1, gTrainingGridXY=30, gMinZ=0, gMaxZ=1, gTrainingGridZ=4):
        """
        The default constructor for class CartesianSpace
        ------------------------------------------------------------------------------------
        :param gMinXY: the minimum point of interval for the XY plane
        :param gMaxXY: the maximum point of interval for the XY plane
        :param gTrainingGridXY: the resolution for the XY axis
        :param gMinZ: the minimum point of interval for the Z axis
        :param gMaxZ: the maximum point of interval for the Z axis
        :param gTrainingGridZ: the resolution for the Z axis (number of zSlices in total)
        """
        # x,y grid dimension
        self.gMinXY = gMinXY
        self.gMaxXY = gMaxXY

        # x, y grid bins
        self.gTrainingGridXY = gTrainingGridXY

        # z grid dimension
        self.gMinZ = gMinZ
        self.gMaxZ = gMaxZ

        # z grid dimension - must be divisible by 4
        assert (gTrainingGridZ % 4 == 0), "Z Grid Dimension (gTrainingGridZ = {}) is not divisible by 4".format(gTrainingGridZ)
        self.gTrainingGridZ = gTrainingGridZ

        # the step size for the XY plane and the Z axis
        self.gBinSizeXY = (self.gMaxXY - self.gMinXY) / self.gTrainingGridXY
        self.gBinSizeZ = (self.gMaxZ - self.gMinZ) / self.gTrainingGridZ

        # total number of bins for this compton data space
        self.totalBinNums = self.gTrainingGridXY*self.gTrainingGridXY*self.gTrainingGridZ

        # 'GridXY' is a 1D grid for either the X axis or the Y axis
        # 'GridZ' is a 1D grid for the Z axis
        self.GridXY = self.getGridXY()
        self.GridZ = self.getGridZ()

    def getTotalBinNums(self):
        """
        This method returns the total number of data bins in this compton data space
        ------------------------------------------------------------------------------------
        :return: int
        """
        return self.totalBinNums

    def sampleSinglePointOnXY(self):
        """
        This method uniformly and randomly samples a single point (X,Y) on the XY plane

        Note: The sampled (X,Y) is continuous and might not fit exactly into a specific bins in the space
        ------------------------------------------------------------------------------------
        :return: tuple(x,y)
        """
        x = random.uniform(self.gMinXY, self.gMaxXY)
        y = random.uniform(self.gMinXY, self.gMaxXY)
        return (x, y)

    def plotFlattenedResponse(self, XSingle, YSingle, Title, FigureNumber=0, zSlices=4):
        """
        This method plots the imaging response by slicing at different Z and plots the corresponding XY plane in 2D
        ------------------------------------------------------------------------------------
        :param XSingle: the input data/measurement
        :param YSingle: the flattened imaging response (1D array)
        :param Title: the title for the entire figure
        :param FigureNumber: the figure number for the figure
        :param zSlices: how many slices of Z we would like to plot
        """
        if (zSlices <= 0 or zSlices > self.gTrainingGridZ):
            warnings.warn(f"zSlices={zSlices} is not between 1 and {self.gTrainingGridZ}(Z_BIN_NUM). " +
                          f"Thus, zSlices is replaced by the program with the following values: 1")
            zSlices = 1

        if (self.gTrainingGridZ % zSlices != 0):
            warnings.warn(f"{self.gTrainingGridZ}(Z_BIN_NUM) is not divisible by zSlices={zSlices}. " +
                          f"Thus, zSlices is automatically rounded to {int(self.gTrainingGridZ/int(self.gTrainingGridZ/zSlices))}.")
            zSlices = int(self.gTrainingGridZ/int(self.gTrainingGridZ/zSlices))

        # Figure setup
        fig = plt.figure(FigureNumber)
        plt.clf()  # clear everything on this figure
        fig.canvas.set_window_title(Title)
        fig.suptitle(f"XSingle = {XSingle}")
        plt.subplots_adjust(hspace=0.5)
        plotCols = 2
        plotRows = int(math.ceil(zSlices / plotCols))

        # Data setup
        XV, YV = np.meshgrid(self.GridXY, self.GridXY)
        Z = np.zeros(shape=(self.gTrainingGridXY, self.gTrainingGridXY))

        # Iterate through all zSlice and plot the 2D slice of the XY plane
        initialZIdx = 0
        zStep = int(self.gTrainingGridZ / zSlices)
        for plotIndex, z_Idx in enumerate(range(initialZIdx, self.gTrainingGridZ, zStep), 1):
            for x_Idx in range(self.gTrainingGridXY):
                for y_Idx in range(self.gTrainingGridXY):
                    Z[x_Idx, y_Idx] = YSingle[0, self.convertToFlattenedIndex(x_Idx, y_Idx, z_Idx)]
            ax = fig.add_subplot(plotRows, plotCols, plotIndex)
            ax.set_title("Slice through z={}".format(self.GridZ[z_Idx]))
            contour = ax.contourf(XV, YV, Z)

        plt.ion()
        plt.show()
        plt.pause(0.001)

        # Save the plots as 'PNG' files
        '''
        STILL NOT FINISHED/POLISHED FEATURES!!!
        '''
        OUTPUTPREFIX = "Run"
        if FigureNumber == 1:
            plt.savefig(OUTPUTPREFIX + "_Original.png")
        else:
            plt.savefig(OUTPUTPREFIX + "_Result.png")

    def createGaussianFlattenedResponse(self, PosX, PosY, SigmaR):
        """
        Create the response for a source at (PosX, PosY) using the Gaussian distribution model.
        The resulting imaging response is a 1D flattened array, where each element corresponds to a single point on the
        Cartesian Space
        ------------------------------------------------------------------------------------
        :param PosX: x position of the source
        :param PosY: y position of the source
        :param SigmaR: the standard deviation for the gaussian distribution
        :return: ndarray - a flattened array containing the response
        """
        Out = np.zeros(shape=(1, self.totalBinNums))
        for x_Idx in range(self.gTrainingGridXY):
            for y_Idx in range(self.gTrainingGridXY):
                for z_Idx in range(self.gTrainingGridZ):
                    centerX, centerY = self.GridXY[x_Idx], self.GridXY[y_Idx]
                    r = math.sqrt((PosX - centerX) ** 2 + (PosY - centerY) ** 2)
                    Out[0, self.convertToFlattenedIndex(x_Idx, y_Idx, z_Idx)] = Gaussian.getGauss(math.fabs(r - self.GridZ[z_Idx]), SigmaR)
        return Out

    def convertToFlattenedIndex(self, x_Idx, y_Idx, z_Idx):
        """
        This method converts the corresponding index for x,y,z to the flattened index for the 1D response array.
        The conversion is performed using a virtual indexing/mapping scheme, which is given by the formula:
                ++ =================================================================== ++
                ||  flattenedIndex = x + y * XY_BIN_NUM + z * XY_BIN_NUM * XY_BIN_NUM  ||
                ++ =================================================================== ++
        , where XY_BIN_NUM = self.gTrainingGridXY, representing the number of bins on a slice of XY plane
        ------------------------------------------------------------------------------------
        :param x_Idx: the x index, which is within: [0, 1, 2, ..., self.gTrainingGridXY]
        :param y_Idx: the y index, which is within: [0, 1, 2, ..., self.gTrainingGridXY]
        :param z_Idx: the z index, which is within: [0, 1, 2, ..., self.gTrainingGridZ]
        :return: flattenedIdx - a single number that can be used to index the 1D flattened imaging response array
        """
        if (x_Idx < 0 or x_Idx >= self.gTrainingGridXY or
                y_Idx < 0 or y_Idx >= self.gTrainingGridXY or
                z_Idx < 0 or z_Idx >= self.gTrainingGridZ):
            raise IndexError(f"The indexes (x_Idx, y_Idx, z_Idz = [{x_Idx},{y_Idx},{z_Idx}]) is " +
                             "out of bound for this cartesian space whose shape is " +
                             f"{(self.gTrainingGridXY, self.gTrainingGridXY, self.gTrainingGridZ)}")

        return x_Idx + y_Idx * self.gTrainingGridXY + z_Idx * self.gTrainingGridXY * self.gTrainingGridXY

    def convertToExpandedIndex(self, flattenedIdx):
        """
        This method converts the flattened index back into the complete 3D coordinates X, Y, Z.

        Note: This method is a reversal process of the method 'self.convertToFlattenedIndex(x_Idx, y_Idx, z_Idx)'.
        Please refer to the method :func:`~comptonspace.CartesianSpace.convertToFlattenedIndex` for more details
        ------------------------------------------------------------------------------------
        :param flattenedIdx:
        :return: (X, Y, Z)
        """
        if (flattenedIdx >= self.totalBinNums):
            raise IndexError(f"The flattened index {flattenedIdx} is out of bound for this cartesian space " +
                             f"as it exceeds or equals the total number of bins {self.totalBinNums}")
        z = flattenedIdx // (self.gTrainingGridXY*self.gTrainingGridXY)
        y = (flattenedIdx - z*self.gTrainingGridXY*self.gTrainingGridXY) // self.gTrainingGridXY
        x = flattenedIdx - z*self.gTrainingGridXY*self.gTrainingGridXY - y*self.gTrainingGridXY
        return (x,y,z)


    # def __iter__(self):
    #     """
    #     Iterator for Cartesian Space
    #
    #     Iterates through all points in this compton data space using enumeration.
    #     Each element will look like (idx, point):
    #         idx - the index for the current point
    #         point - (x,y,z) is a 1D array with 3 elements representing a single point
    #     ------------------------------------------------------------------------------------
    #     :return: iterator for a numpy array 'self.GridXYZ'
    #     """
    #     return enumerate(self.GridXYZ)
    #
    # def __len__(self):
    #     """
    #     return the total number of points for this compton data space
    #     ------------------------------------------------------------------------------------
    #     :return: totalNumOfPoints
    #     """
    #     return self.totalNumOfPoints

    def getGridXY(self):
        """
        This method generates a 1D array which contains samples of points on either the X or Y axis
        based on the resolution and interval specified in the class

        Note: since the X axis and the Y axis will be assumed to have the same interval and resolution,
              so this generated array can be used interchangeably for both the X axis and the Y axis

        Caution: This method should be avoided as it generates a new array which uses more memory. Use the 'self.GridXY'
                 if possible so that no new memory are allocated
        ------------------------------------------------------------------------------------
        :return: ndarray
        """
        gGridXY = np.zeros([self.gTrainingGridXY])
        for x in range(0, self.gTrainingGridXY):
            gGridXY[x] = self.gMinXY + (x + 0.5) * (self.gMaxXY - self.gMinXY) / self.gTrainingGridXY
        return gGridXY

    def getGridZ(self):
        """
        This method generates a 1D array which contains samples of points on the Z axis
        based on the resolution and interval specified in the class

        Caution: This method should be avoided as it generates a new array which uses more memory. Use the 'self.GridZ'
                 if possible so that no new memory are allocated
        ------------------------------------------------------------------------------------
        :return: ndarray
        """
        gGridZ = np.zeros([self.gTrainingGridZ])
        for z in range(0, self.gTrainingGridZ):
            gGridZ[z] = self.gMinZ + (z + 0.5) * (self.gMaxZ - self.gMinZ) / self.gTrainingGridZ
        return gGridZ





if __name__ == "__main__":
    # for testing purposes only
    dataSpace = CartesianSpace()


