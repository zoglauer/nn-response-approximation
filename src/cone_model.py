import os
import math
import random
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from src.dataset import ApproxDataset


class ToyModel3DCone:

    def __init__(self, output_dir='Run'):
        print('\nToyModel: (x,y) --> Compton cone for all  x, y in [-1, 1]')

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

        for i in range(1, 5):    
            zGridElement = int((i-1)*self.gTrainingGridZ/4)
            for x in range(self.gTrainingGridXY):
                for y in range(self.gTrainingGridXY):
                    Z[x, y] = YSingle[x + y*self.gTrainingGridXY + zGridElement*self.gTrainingGridXY*self.gTrainingGridXY]

            ax = fig.add_subplot(2, 2, i)
            ax.set_title('Slice through z={}'.format(self.gGridCentersZ[zGridElement]))
            contour = ax.contourf(XV, YV, Z)  

        plt.ion()
        plt.show()
        plt.pause(0.001)
        
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


    def CreateFullResponse(self, PosX, PosY, flattened=True):
        '''
        Create the response for a source at position PosX, PosY
        
        Args:
        PosX (float): x position of the source
        PosY (float): y position of the source
        flattened: whether the return array is flattened into 1-D array
        
        '''
        if flattened:
            Out = np.zeros(shape=(self.OutputDataSpaceSize, ))
        else:
            Out = np.zeros(shape=(self.gTrainingGridXY, self.gTrainingGridXY, self.gTrainingGridZ))
        

        for x in range(0, self.gTrainingGridXY):
            for y in range(0, self.gTrainingGridXY):
                for z in range(0, self.gTrainingGridZ):
                    r = math.sqrt((PosX - self.gGridCentersXY[x])**2 + (PosY - self.gGridCentersXY[y])**2 )
                    if flattened:
                        idx = x + y*self.gTrainingGridXY + z*self.gTrainingGridXY*self.gTrainingGridXY
                        Out[idx] = self.getGauss(math.fabs(r - self.gGridCentersZ[z]), self.gSigmaR)
                    else:
                        Out[x][y][z] = self.getGauss(math.fabs(r - self.gGridCentersZ[z]), self.gSigmaR)
        
        return Out
    

    def create_dataset(self, dataset_size=1024, flattened=True):
        '''
        Generate dataset for the responses.
        
        Args:
        dataset_size: Dataset size
        flattened: whether the return array is flattened into 1-D array
        '''
        X, Y = self.create_data(dataset_size, flattened)

        return ApproxDataset(X, Y)
        

    def create_data(self, data_amount, flattened):
        X = np.zeros(shape=(data_amount, self.InputDataSpaceSize))
        Y = np.zeros(shape=(data_amount, self.OutputDataSpaceSize))
        for i in tqdm(range(data_amount), desc='creating data'):
            X[i, 0] = random.uniform(self.gMinXY, self.gMaxXY)
            X[i, 1] = random.uniform(self.gMinXY, self.gMaxXY)
            Y[i] = self.CreateFullResponse(X[i, 0], X[i, 1], flattened)       

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
        

