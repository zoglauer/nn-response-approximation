###################################################################################################
#
# ToyModel3DConeKeras.py
#
# Copyright (C) by Andreas Zoglauer, Shivani Kishnani & contributors.
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice. 
#  
###################################################################################################

###################################################################################################


# Base Python
import random
import math
import time
import concurrent.futures


# High level python tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import tensorflow as tf

import numpy as np

# Own tools
from utility.Helpers import *
from comptonspace.CartesianSpace import CartesianSpace
from comptonspace.HEALPixSpace import HEALPixSpace

class ToyModel3DCone:
    """
    This class implements a 3D toy model response and does the training of the neural network
    """

    ###################################################################################################

    def __init__(self, OutputPrefix="Run"):
        """
        The default constructor for class ToyModel3DCone

        Args:
          OutputPrefix (string): Prefix for the output data

        """

        self.OutputPrefix = OutputPrefix
        self.UseBatchMode = False

        # self.Layout = [10, 100, 1000]

        # print("\nToyModel: (x,y) --> Compton cone for all  x, y in [-1, 1]\n")

        # Width of the cone
        self.gSigmaR = 0.1

        # The Compton Data Space of the model
        NSIDE = 32
        self.comptonDataSpace = HEALPixSpace(NSIDE)

        # Set test and traing data set parameters
        self.InputDataSpaceSize = 2
        self.OutputDataSpaceSize = self.comptonDataSpace.getTotalBinNums()

        self.SubBatchSize = 1024

        self.NTrainingBatches = 1
        self.TrainingBatchSize = self.NTrainingBatches * self.SubBatchSize

        self.NTestingBatches = 1
        self.TestBatchSize = self.NTestingBatches * self.SubBatchSize


    ###################################################################################################

    def setBatchMode(self, UseBatchMode=True):
        '''
        Toggle the batch mode, i.e., choose if to use a UI or not
        '''

        self.UseBatchMode = UseBatchMode

    ###################################################################################################

    def Plot2D(self, XSingle, YSingle, Title, FigureNumber=0):
        '''
        A function for plotting 4 slices of the model in one figure
        '''
        self.comptonDataSpace.plotFlattenedResponse(XSingle, YSingle, Title, FigureNumber)

    ###################################################################################################

    #################
    # -- The following method is moved to the module 'utility.Gaussian.Gauss3D' --
    # ============================================================================
    # def Gauss3D(self, X, x0, y0, R):
    #     '''
    #     Return a donghnut with a Gaussian shape
    #
    #     Args:
    #       X (float, float): Tuple of the x, y position
    #       x0 (float): x center
    #       y0 (float): x center
    #       R (float):  radius
    #     '''
    #     x, y = X
    #     return np.exp(((-np.sqrt((x - x0) ** 2 + (y - y0) ** 2) - R) ** 2) / self.gSigmaR)
    # ============================================================================

    #################
    # -- The following method is moved to the module 'utility.Gaussian.getGauss' --
    # =============================================================================
    # def getGauss(self, d, sigma=1):
    #     '''
    #     Return a 1D Gaussian value
    #
    #     Args:
    #       d (float):      Distance from 0
    #       sigma (float):  Sigma value of Gaussian
    #
    #     '''
    #
    #     return 1 / (sigma * math.sqrt(2 * np.pi)) * math.exp(-0.5 * pow(d / sigma, 2))
    # ============================================================================


    ###################################################################################################

    def CreateRandomResponsePoint(self, PosX, PosY):
        '''
        Create a RANDOM response data point for a source at position PosX, PosY

        Args:
          PosX (float): x position of the source
          PosY (float): y position of the source

        '''
        return (PosX + random.gauss(PosX, self.gSigma), random.uniform(self.gMinXY, self.gMaxXY))

    ###################################################################################################

    def CreateFullResponse(self, PosX, PosY):
        '''
        Create the response for a source at position PosX, PosY

        Args:
          PosX (float): x position of the source
          PosY (float): y position of the source

        TODO:
        # This way of integrating the distance between first and second compton interaction might be WRONG!!
        # ASK Andreas before proceeding
        distance_betw_first_second_interaction = 0
        k = 0.05
        b = 0
        adjustedSigma = k * distance_betw_first_second_interaction + b + self.gSigmaR
        return self.comptonDataSpace.createGaussianFlattenedResponse(PosX, PosY, adjustedSigma)
        '''
        return self.comptonDataSpace.createGaussianFlattenedResponse(PosX, PosY, self.gSigmaR)

    ###################################################################################################

    """
    def test(self):
        colatitude, longitude = self.comptonDataSpace.sampleSinglePointOnXY()
        for d in range(0, 5):
            YTrain = np.zeros(shape=(2, self.OutputDataSpaceSize))
            YTrain[0,] = self.CreateFullResponse(colatitude, longitude, d)
            self.Plot2D((colatitude, longitude), YTrain[0:1], f"Original d = {d}", d)
        plt.show()
    """

    def _gen_one_data(self, index):
        X = self.comptonDataSpace.sampleSinglePointOnXY()
        collision_distance = 5 * np.random.random_sample()
        print("progress..")
        Y = self.CreateFullResponse(X[0], X[1])
        return (index, X, Y)


    def genSimulatedData(self, totalBatchSize):
        # Meta Data
        completed = 0

        # Start Task
        X = np.zeros(shape=(totalBatchSize, self.InputDataSpaceSize))
        Y = np.zeros(shape=(totalBatchSize, self.OutputDataSpaceSize))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(self._gen_one_data, i) for i in range(0, totalBatchSize)}

            for fut in concurrent.futures.as_completed(futures):
                index, X_Single, Y_Single = fut.result()
                X[index] = X_Single
                Y[index, ] = Y_Single

                completed += 1
                if completed > 0 and completed % 128 == 0:
                    print("Data creation: {}/{}".format(completed, totalBatchSize))

        return X, Y


    def train(self):
        '''
        Perfrom the neural network training
        '''
        print("Info: Creating %i data sets" % (self.TrainingBatchSize + self.TestBatchSize))

        print("\nInfo: Generating Training Data")
        XTrain, YTrain = self.genSimulatedData(self.TrainingBatchSize)

        print("\nInfo: Generating Testing Data")
        XTest, YTest = self.genSimulatedData(self.TestBatchSize)

        print("\nInfo: Setting up neural network...")
        Model = tf.keras.Sequential()
        Model.add(tf.keras.layers.Dense(10, activation="relu", kernel_initializer='he_normal',
                                        input_shape=(self.InputDataSpaceSize,)))
        Model.add(tf.keras.layers.Dense(100, activation="relu", kernel_initializer='he_normal'))
        Model.add(tf.keras.layers.Dense(1000, activation="relu", kernel_initializer='he_normal'))
        Model.add(tf.keras.layers.Dense(self.OutputDataSpaceSize))

        Model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=1e-08), loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['mse', 'mae', 'mape'])

        Model.summary()

        print("Info: Training and evaluating the network")

        # Train the network
        Timing = time.process_time()

        XSingle = XTest[0:1]
        YSingle = YTest[0:1]
        if self.UseBatchMode == False:
            self.Plot2D(XSingle, YSingle, "Original", 1)

        TimesNoImprovement = 0
        BestMeanSquaredError = 10 ** 30  # sys.float_info.max

        for Iteration in range(0, 50000):

            # Take care of Ctrl-C -- does not work
            global Interrupted
            if Interrupted == True: break

            # Train
            history = Model.fit(XTrain, YTrain, verbose=2, batch_size=self.TrainingBatchSize,
                                validation_data=(XTest, YTest))

            # Check performance: Mean squared error
            if Iteration > 0 and Iteration % 20 == 0:
                # MeanSquaredError = sess.run(tf.nn.l2_loss(Output - YTest)/self.TestBatchSize,  feed_dict={X: XTest})
                RunOut = Model.predict(XTest)
                MeanSquaredError = math.sqrt(np.sum((YTest - RunOut) ** 2))

                print("Iteration {} - MSE of test data: {}".format(Iteration, MeanSquaredError))

                if MeanSquaredError <= BestMeanSquaredError:  # We need equal here since later ones are usually better distributed
                    BestMeanSquaredError = MeanSquaredError
                    TimesNoImprovement = 0

                    # Saver.save(sess, "model.ckpt")

                    # Test just the first test case:
                    YOutSingle = Model.predict(XSingle)

                    if self.UseBatchMode == False:
                        self.Plot2D(XSingle, YOutSingle, "Reconstructed at iteration {}".format(Iteration), 2)

                else:
                    TimesNoImprovement += 1
            else:
                if self.UseBatchMode == False:
                    plt.pause(0.001)

                # end: check performance

            if TimesNoImprovement == 100:
                print("No improvement for 100 rounds")
                break;

            # end: iterations loop

        YOutTest = sess.run(Output, feed_dict={X: XTest})

        Timing = time.process_time() - Timing
        if Iteration > 0:
            print("Time per training loop: ", Timing / Iteration, " seconds")

        input("Press [enter] to EXIT")

        return

# END
###################################################################################################
