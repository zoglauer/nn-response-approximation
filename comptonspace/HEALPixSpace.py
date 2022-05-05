# Base Python Libraries
import math
import random
import warnings
from pathlib import Path
import glob
import pickle
import logging

# Third Party Libraries
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# Own Tools
from comptonspace.AbstractComptonSpace import AbstractComptonSpace
from utility.Gaussian import Gaussian




class HEALPixSpace(AbstractComptonSpace):
    """
    This class represents a Compton Data Space that is implemented using HEALPix pixelization in spherical coordinates
    """

    def __init__(self, NSIDE=32, gMinZ = 0, gMaxZ = 1, gTrainingGridZ=4):
        """
        The default constructor for class HEALPixSpace
        ------------------------------------------------------------------------------------
        :param NSIDE: the 'NSIDE' parameter for HEALPix
        :param gMinZ: the minimum point of interval for the Z axis
        :param gMaxZ: the maximum point of interval for the Z axis
        :param gTrainingGridZ: the resolution for the Z axis (number of zSlices in total)
        """
        # NSIDE parameters for HEALPix
        self.NSIDE = NSIDE

        # colatitude axis dimension
        self.gMinLat = 0
        self.gMaxLat = math.pi

        # longitude axis dimension
        self.gMinLong= 0
        self.gMaxLong = 2*math.pi

        # z grid dimension
        self.gMinZ = gMinZ
        self.gMaxZ = gMaxZ

        # z grid dimension - must be divisible by 4
        assert (gTrainingGridZ % 4 == 0), "Z Grid Dimension (gTrainingGridZ = {}) is not divisible by 4".format(
            gTrainingGridZ)
        self.gTrainingGridZ = gTrainingGridZ

        # 'GridZ' is a 1D grid for the Z axis
        self.GridZ = self.getGridZ()

        # number of bins/pixels for this HEALPix space - the so-called "XY plane"
        self.totalPixNums = hp.nside2npix(self.NSIDE)

        # total number of bins for this HEALPix compton data space
        self.totalBinNums = hp.nside2npix(self.NSIDE) * self.gTrainingGridZ

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
        colatitude = random.uniform(self.gMinLat, self.gMaxLat)
        longitude = random.uniform(self.gMinLong, self.gMaxLong)
        return (colatitude, longitude)

    def plotFlattenedResponse(self, XSingle, YSingle, Title, FigureNumber=0, zSlices=4):
        """
        This method plots the imaging response by slicing at different Z and plots the corresponding XY plane in 2D.
        The corresponding response is plotted using 'Mollview' as provided in the 'healpy' library.

        NOTE: If this function doesn't show the plot, try calling "plt.show()" after this function
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
        plt.clf()   # clear everything on this figure
        fig.canvas.set_window_title(Title)
        fig.suptitle(f"XSingle = {np.degrees(XSingle)}")
        plt.subplots_adjust(hspace=0.5)
        plotCols = 2
        plotRows = int(math.ceil(zSlices / plotCols))

        # Iterate through all zSlice and plot the 2D slice of the XY plane
        initialZIdx = 0
        zStep = int(self.gTrainingGridZ / zSlices)
        for plotIndex, z_Idx in enumerate(range(initialZIdx, self.gTrainingGridZ, zStep), 1):
            startPixIndex = z_Idx*self.totalPixNums
            endPixIndex = z_Idx*self.totalPixNums+self.totalPixNums - 1
            currHEALPixMap = YSingle[0, startPixIndex:endPixIndex+1]

            # add another subplot for the later mollview plot
            fig.add_subplot(plotRows, plotCols, plotIndex)
            # the following mollview will be automatically plotted on the added subplot
            hp.mollview(currHEALPixMap, fig=FigureNumber, title="Slice through z={}".format(self.GridZ[z_Idx]), hold=True)
            hp.graticule()


    def createGaussianFlattenedResponse(self, PosX, PosY, SigmaR):
        """
        Create the response for a source at (PosX, PosY) using the Gaussian distribution model.
        The resulting imaging response is a 1D flattened array, where each element corresponds to a single point on the
        HEALPix space
        ------------------------------------------------------------------------------------
        :param PosX: x position of the source
        :param PosY: y position of the source
        :param SigmaR: the standard deviation for the gaussian distribution
        :return: ndarray - a flattened array containing the response
        """
        Out = np.zeros(shape=(1, self.totalBinNums))
        for pix_Idx in range(self.totalPixNums):
            for z_Idx in range(self.gTrainingGridZ):
                centerLat, centerLong = self.getLatLongFromPixIdx(pix_Idx)
                r = HEALPixSpace.angularDistanceBetween(PosX, PosY, centerLat, centerLong)
                Out[0, self.convertToFlattenedIndex(pix_Idx, z_Idx)] = Gaussian.getGauss(math.fabs(r - self.GridZ[z_Idx]), SigmaR)
        return Out

    @staticmethod
    def angularDistanceBetween(Lat_A, Long_A, Lat_B, Long_B):
        """
        This staticmethod calculates the angular distance between two points on the HEALPix Space
        ------------------------------------------------------------------------------------
        :param Lat_A: the colatitude for pointA
        :param Long_A: the longitude for pointA
        :param Lat_B: the colatitude for pointB
        :param Long_B: the longitude for pointB
        :return: the angular distance between pointA and pointB
        """
        return hp.rotator.angdist([Lat_A, Long_A], [Lat_B, Long_B])[0]

    def getLatLongFromPixIdx(self, pix_Idx, degree=False):
        """
        This method returns the corresponding (colatitude, longitude) pair given a pixel index on the HEALPix Space
        ------------------------------------------------------------------------------------
        :param pix_Idx: the pixel index
        :param degree: if True, then return results in degree rather than radians
        :return: (colatitude, longitude)
        """
        return hp.pix2ang(self.NSIDE, pix_Idx, lonlat=degree)

    def getPixIdxFromLatLong(self, latitude, longitude, degree=False):
        return hp.ang2pix(self.NSIDE, latitude, longitude, lonlat=degree)

    def getScatterAngFromZIdx(self, z_Idx):
        """
        This method returns the corresponding z value given an index for the bin on the Z axis
        ------------------------------------------------------------------------------------
        :param z_Idx: the index for the bins on the Z axis
        :return: the value of the bins
        """
        if (z_Idx < 0 or z_Idx >= self.gTrainingGridZ):
            raise IndexError(f"The z-Index {z_Idx} is out of bound on the z axis for this compton data space " +
                             f"whose 'gTrainingGridZ = {self.gTrainingGridZ}'")
        return self.GridZ[z_Idx]

    def getZIdxFromScatterAng(self, scatterAng, zWidth=-1):
        """
        TODO: OPTIMIZE THIS
        """
        if (zWidth == -1):
            z_interval = (self.gMaxZ - self.gMinZ) / self.gTrainingGridZ
            for zIdx in range(0, self.gTrainingGridZ-1):
                if (scatterAng <= self.GridZ[zIdx] + z_interval):
                    return zIdx
            return self.gTrainingGridZ-1
        else:
            for zIdx in range(0, self.gTrainingGridZ):
                if ((scatterAng >= self.GridZ[zIdx]-zWidth) and (scatterAng <= self.GridZ[zIdx]+zWidth)):
                    return zIdx
            return -1


    def convertToFlattenedIndex(self, pix_Idx, z_Idx):
        """
        This method converts the corresponding index (pixelIndex, zIndex) to the flattened index
        for the 1D response array. The conversion is performed using a virtual indexing/mapping scheme,
        which is given by the formula:
                ++ =============================================== ++
                ||  flattenedIndex = pix_Idx + z * TOTAL_PIX_NUMS  ||
                ++ =============================================== ++
        , where TOTAL_PIX_NUMS = self.totalPixNums, representing the number of pixels on one zSlice
        of the HEALPix Space
        ------------------------------------------------------------------------------------
        :param pix_Idx: the pixel index for the bin on XY Plane on HEALPix space,
                        which is within: [0, 1, 2, ..., self.totalPixNums]
        :param z_Idx: the z index, which is within: [0, 1, 2, ..., self.gTrainingGridZ]
        :return: flattenedIdx - a single number that can be used to index the 1D flattened imaging response array
        """
        return pix_Idx + z_Idx * self.totalPixNums

    def convertToExpandedIndex(self, flattenedIdx):
        """
        This method converts the flattened index back into the complete coordinates (PIX_INDEX, Z_INDEX).

        Note: This method is a reversal process of the method 'self.convertToFlattenedIndex(pix_Idx, z_Idx)'.
        Please refer to the method :func:`~comptonspace.HEALPixSpace.convertToFlattenedIndex` for more details.

        Note: PIX_INDEX is the pixel index for a point on a zSlice of the HEALPix space. To get
        the (colatitude, longitude) from the pixel index, you might want to look into the method
        'self.getLatLongFromPixIdx(pix_Idx, degree=False)'. Please refer to the method
        :func:`~comptonspace.HEALPixSpace.getLatLongFromPixIdx` for more details.
        ------------------------------------------------------------------------------------
        :param flattenedIdx:
        :return: (PIX_INDEX, Z)
        """
        if (flattenedIdx >= self.totalBinNums):
            raise IndexError(f"The flattened index {flattenedIdx} is out of bound for this HEALPix space " +
                             f"as it exceeds or equals the total number of bins {self.totalBinNums}")
        z_Idx = flattenedIdx // self.totalPixNums
        pix_Idx = flattenedIdx - (z_Idx * self.totalPixNums)
        return (pix_Idx, z_Idx)

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


    def loadSimulationData(self, simulation_file, zWidth=2.5):
        """
        TODO:
        This function was not finished!! Don't use directly!!
        """
        simulation_file = Path(simulation_file).resolve()
        if not simulation_file.is_file():
            raise Exception(f"The path {simulation_file} is not a valid file")

        Out = np.zeros(shape=(1, self.totalBinNums))
        with open(simulation_file, "rb") as simulation_file:
            incoming_dir, events_id, scatter_ang, theta, phi, collision_dist = pickle.load(simulation_file)
            incoming_dir = [float(i) / 180 * np.pi for i in incoming_dir]

            total_events_num = len(events_id)
            print(f"incoming angle: {np.array(incoming_dir) / np.pi * 180}")
            print(f"valid events: {total_events_num}")

            print(f"min theta = {min(theta)}; max theta = {max(theta)}")
            print(f"min phi = {min(phi)}; max phi = {max(phi)}")
            print(f"min collision_dist = {min(collision_dist)}; max collision_dist = {max(collision_dist)}")

            theta = np.radians(np.asarray(theta))
            phi = np.asarray(phi)
            phi = np.where(phi < 0, np.radians(phi+360), np.radians(phi))

            for i in range(total_events_num):

                pixIdx = self.getPixIdxFromLatLong(theta[i], phi[i])
                zIdx = self.getZIdxFromScatterAng(scatter_ang[i], zWidth=zWidth)
                if (zIdx == -1 or collision_dist[i] <= 0.5):
                    continue
                # print(f"scatter ang = {scatter_ang[i]} ==> ziDx = {zIdx}")
                Out[0, self.convertToFlattenedIndex(pixIdx, zIdx)] += 1

            dataspace.plotFlattenedResponse(incoming_dir, Out, Title="test", zSlices=self.gTrainingGridZ)
            plt.show()

        # phi_shifted = np.array([])
        # for p in phi:
        #     if (p < 0):
        #         phi_shifted = np.append(phi_shifted, ((p+360)/180*np.pi))
        #     else:
        #         phi_shifted = np.append(phi_shifted, (p/180*np.pi))
        # allFiles = glob.glob(f"{simulation_folder / '*'}")
        # for file in allFiles:
        #     if file.endswith("pkl"):
        #         with open(file, "rb") as simulation_file:
        #             incoming_dir, events_id, scatter_ang, theta, phi, collision_dist = pickle.load(simulation_file)
        #             theta = np.array(theta) / 180 * np.pi
        #             phi = (np.array(phi)+180) / 180 * np.pi
        #             total_events_num = len(events_id)
        #             for i in range(total_events_num):
        #                 pixIdx = self.getPixIdxFromLatLong(theta[i], phi[i])
        #                 zIdx = self.getZIdxFromScatterAng(scatter_ang[i])
        #                 Out[0, self.convertToFlattenedIndex(pixIdx, zIdx)] += 1

        return Out


# for testing purposes only
if __name__ == "__main__":
    logging.disable('WARNING')

    dataspace = HEALPixSpace(NSIDE=5, gMinZ=0, gMaxZ=180, gTrainingGridZ=4)
    print(dataspace.GridZ)

    simulation_folder = r"/home/tezktenr/Study/UCB_Capstone/simulation_data/March31_EarlyMorning"
    allFiles = glob.glob(f"{Path(simulation_folder) / '*'}")
    simulation_file = allFiles[3]

    # simulation_file = r"/home/tezktenr/Study/UCB_Capstone/simulation_data/PS_0_0.inc1.id1.tra.gz.pkl"
    out = dataspace.loadSimulationData(simulation_file, zWidth=2.5)
    print(max(max(out)))
    # print(out)





