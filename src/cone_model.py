import os
import math
import random
import warnings
import numpy as np
import healpy as hp
from tqdm import tqdm
import concurrent.futures

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from src.dataset import ApproxDataset
from src.comptonspace.healpixspace import HEALPixSpace


class HEALPixCone:
    def __init__(self, output_dir='Run', NSIDE=32, gMinZ=0, gMaxZ=1, gTrainingGridZ=4):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.compton_space = HEALPixSpace(NSIDE, gMinZ, gMaxZ, gTrainingGridZ)
        
        self.gTrainingGridXY = self.compton_space.totalPixNums
        self.gTrainingGridZ = gTrainingGridZ
        self.gMinZ = 0
        self.gMaxZ = 1
        # Width of the cone
        self.gSigmaR = 0.1
        
        self.flattened = True
        self.InputDataSpaceSize = 2
        self.totalPixNums = self.compton_space.totalPixNums
        self.OutputDataSpaceSize = self.compton_space.totalBinNums

    def CreateFullResponse(self, PosX, PosY, d=0):
        '''
        Create the response for a source at position PosX, PosY
        Args:
          PosX (float): x position of the source
          PosY (float): y position of the source
        '''
        k, b = 0.05, 0
        adjustedSigma = k * d + b + self.gSigmaR
        return self.compton_space.createGaussianFlattenedResponse(PosX, PosY, adjustedSigma)

    def create_dataset(self, dataset_size=1024):
        X, Y = self.create_data(dataset_size)
        return ApproxDataset(X, Y)

    def create_data(self, data_amount):
        X = np.zeros(shape=(data_amount, self.InputDataSpaceSize))
        Y = np.zeros(shape=(data_amount, self.OutputDataSpaceSize))

        def _gen_one_data(index):
            X = self.compton_space.sampleSinglePointOnXY()
            Y = self.CreateFullResponse(X[0], X[1])
            print(index, end='\r')
            return (index, X, Y)


        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(_gen_one_data, i) for i in range(0, data_amount)}

            for fut in concurrent.futures.as_completed(futures):
                index, X_Single, Y_Single = fut.result()
                X[index] = X_Single
                Y[index, ] = Y_Single
        
        return X, Y

    def create_data2(self, data_amount):
        X = np.zeros(shape=(data_amount, self.InputDataSpaceSize))
        if self.flattened:
            Y = np.zeros(shape=(data_amount, self.OutputDataSpaceSize))
        else:
            Y = np.zeros(shape=(data_amount, self.gTrainingGridXY, self.gTrainingGridXY, self.gTrainingGridZ))
            
        for i in tqdm(range(data_amount), desc='Generating data...'):
            # X[i] = np.random.uniform(self.gMinXY, self.gMaxXY, size=(self.InputDataSpaceSize, ))
            X[i] = self.compton_space.sampleSinglePointOnXY()
            Y[i] = self.compton_space.createGaussianFlattenedResponse(PosX=X[i, 0], PosY=X[i, 1], SigmaR=0.1)       

        return X, Y
    
    def Plot2D(self, XSingle, YSingle, figure_title='plot.png', zSlices=4):
        # print("XSingle, YSingle:", XSingle.shape, YSingle.shape)
        if (zSlices <= 0 or zSlices > self.gTrainingGridZ):
            warnings.warn(f"zSlices={zSlices} is not between 1 and {self.gTrainingGridZ}(Z_BIN_NUM). " +
                          f"Thus, zSlices is replaced by the program with the following values: 1")
            zSlices = 1

        if (self.gTrainingGridZ % zSlices != 0):
            warnings.warn(f"{self.gTrainingGridZ}(Z_BIN_NUM) is not divisible by zSlices={zSlices}. " +
                          f"Thus, zSlices is automatically rounded to {int(self.gTrainingGridZ/int(self.gTrainingGridZ/zSlices))}.")
            zSlices = int(self.gTrainingGridZ/int(self.gTrainingGridZ/zSlices))


        # Figure setup
        fig = plt.figure()
        plt.clf()   # clear everything on this figure
        # fig.canvas.set_window_title(figure_title)
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
            currHEALPixMap = YSingle[startPixIndex:endPixIndex+1]
            #print(startPixIndex:endPixIndex+1)
            # print(currHEALPixMap, currHEALPixMap.shape)
            # add another subplot for the later mollview plot
            fig.add_subplot(plotRows, plotCols, plotIndex)
            # the following mollview will be automatically plotted on the added subplot
            hp.mollview(currHEALPixMap, title="Slice through z={}".format(self.compton_space.GridZ[z_Idx]), hold=True)
            hp.graticule()


        plt.savefig(os.path.join(
            self.output_dir,
            figure_title
        ))
        plt.close()



class ToyModel3DCone:

    def __init__(self, output_dir='Run', flattened=True, filter_size=3):
        print('\nToyModel: (x,y) --> Compton cone for all  x, y in [-1, 1]')
        
        self.flattened = flattened
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # x,y grid dimension
        self.gMinXY = -1
        self.gMaxXY = +1

        # x, y grid bins
        self.gTrainingGridXY = 30

        # z grid dimension
        self.gMinZ = 0
        self.gMaxZ = 1

        # z grid dimension - must be divisible by 4
        self.gTrainingGridZ = 4

        # Width of the cone
        self.gSigmaR = 0.1

        # Derived helper variables
        self.gBinSizeXY = (self.gMaxXY - self.gMinXY)/self.gTrainingGridXY
        self.gBinSizeZ = (self.gMaxZ - self.gMinZ)/self.gTrainingGridZ

        self.gGridCentersXY = np.zeros([self.gTrainingGridXY])
        self.gGridCentersZ = np.zeros([self.gTrainingGridZ])

        for x in range(0, self.gTrainingGridXY):
            self.gGridCentersXY[x] = self.gMinXY + (x+0.5)*(self.gMaxXY-self.gMinXY)/self.gTrainingGridXY

        for z in range(0, self.gTrainingGridZ):
            self.gGridCentersZ[z] = self.gMinZ + (z+0.5)*(self.gMaxZ-self.gMinZ)/self.gTrainingGridZ

        # Set test and traing data set parameters
        self.InputDataSpaceSize = 2 
        self.OutputDataSpaceSize = self.gTrainingGridXY * self.gTrainingGridXY * self.gTrainingGridZ

        # Post-processing parameters
        self.filter_size = filter_size

    def Plot2D(self, XSingle, YSingle, figure_title):
        '''
        A function for plotting 4 slices of the model in one figure
        '''
        XV, YV = np.meshgrid(self.gGridCentersXY, self.gGridCentersXY)
        Z = np.zeros(shape=(self.gTrainingGridXY, self.gTrainingGridXY))
        
        fig = plt.figure(0)
        plt.clf()
        plt.subplots_adjust(hspace=0.5)

        # fig.canvas.set_window_title(Title)
        # print("YSingle.shape", YSingle.shape)

        for i in range(1, 5):    
            zGridElement = int((i-1)*self.gTrainingGridZ/4)
            for x in range(self.gTrainingGridXY):
                for y in range(self.gTrainingGridXY):
                    if self.flattened:
                        idx = x + y*self.gTrainingGridXY + zGridElement*self.gTrainingGridXY*self.gTrainingGridXY
                        Z[x, y] = YSingle[idx]
                    else:
                        Z[x, y] = YSingle[x][y][zGridElement]
            
            # Z = ndimage.median_filter(Z, size=self.filter_size)
            ax = fig.add_subplot(2, 2, i)
            ax.set_title('Slice through z={}'.format(self.gGridCentersZ[zGridElement]))
            contour = ax.contourf(XV, YV, Z)
            
        #Applying median filter from SciPy
        

        plt.ion()
        # plt.show()
        # plt.pause(0.001)
        
        plt.savefig(os.path.join(
            self.output_dir,
            figure_title
        ))
        plt.close()


    def getGauss(self, d, sigma = 1):
        '''
        Return a 1D Gaussian value
        
        Args:
        d (float):      Distance from 0
        sigma (float):  Sigma value of Gaussian
        
        '''
        return 1/(sigma*math.sqrt(2*np.pi)) * math.exp(-0.5*pow(d/sigma, 2))


    def CreateFullResponse(self, PosX, PosY):
        '''
        Create the response for a source at position PosX, PosY
        
        Args:
        PosX (float): x position of the source
        PosY (float): y position of the source
        
        '''
        if self.flattened:
            Out = np.zeros(shape=(self.OutputDataSpaceSize, ))
        else:
            Out = np.zeros(shape=(self.gTrainingGridXY, self.gTrainingGridXY, self.gTrainingGridZ))
        

        for x in range(0, self.gTrainingGridXY):
            for y in range(0, self.gTrainingGridXY):
                for z in range(0, self.gTrainingGridZ):
                    r = math.sqrt((PosX - self.gGridCentersXY[x])**2 + (PosY - self.gGridCentersXY[y])**2 )
                    if self.flattened:
                        idx = x + y*self.gTrainingGridXY + z*self.gTrainingGridXY*self.gTrainingGridXY
                        Out[idx] = self.getGauss(math.fabs(r - self.gGridCentersZ[z]), self.gSigmaR)
                    else:
                        Out[x][y][z] = self.getGauss(math.fabs(r - self.gGridCentersZ[z]), self.gSigmaR)
        
        return Out
    

    def create_dataset(self, dataset_size=1024):
        '''
        Generate dataset for the responses.
        
        Args:
        dataset_size: Dataset size
        '''
        X, Y = self.create_data(dataset_size)

        return ApproxDataset(X, Y)
        

    def create_data(self, data_amount):
        X = np.zeros(shape=(data_amount, self.InputDataSpaceSize))
        if self.flattened:
            Y = np.zeros(shape=(data_amount, self.OutputDataSpaceSize))
        else:
            Y = np.zeros(shape=(data_amount, self.gTrainingGridXY, self.gTrainingGridXY, self.gTrainingGridZ))
            
        for i in tqdm(range(data_amount), desc='creating data'):
            X[i] = np.random.uniform(self.gMinXY, self.gMaxXY, size=(self.InputDataSpaceSize, ))
            Y[i] = self.CreateFullResponse(PosX=X[i, 0], PosY=X[i, 1])       

        return X, Y
    
    
    def unflatten_array(self, flattened_array):
        assert flattened_array.reshape(-1).shape == (self.OutputDataSpaceSize, ), "Wrong array size is given."

        unflattened_array = np.zeros(shape=(self.gTrainingGridXY, self.gTrainingGridXY, self.gTrainingGridZ))
        for x in range(0, self.gTrainingGridXY):
            for y in range(0, self.gTrainingGridXY):
                for z in range(0, self.gTrainingGridZ):
                    idx = x + y*self.gTrainingGridXY + z*self.gTrainingGridXY*self.gTrainingGridXY
                    unflattened_array[x][y][z] = flattened_array[idx]
        
        return unflattened_array

    def flatten_array(self, unflattened_array):
        flattened_array = np.zeros(shape=(self.gTrainingGridXY * self.gTrainingGridXY * self.gTrainingGridZ, ))
        for x in range(0, self.gTrainingGridXY):
            for y in range(0, self.gTrainingGridXY):
                for z in range(0, self.gTrainingGridZ):
                    idx = x + y*self.gTrainingGridXY + z*self.gTrainingGridXY*self.gTrainingGridXY
                    flattened_array[idx] = unflattened_array[x][y][z]
        
        return flattened_array  

