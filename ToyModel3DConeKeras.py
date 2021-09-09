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

# High level python tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import tensorflow as tf

import numpy as np

# Own tools
from helpers import *




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
    #self.Layout = [10, 100, 1000]

    print("\nToyModel: (x,y) --> Compton cone for all  x, y in [-1, 1]\n")

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
    self.OutputDataSpaceSize = self.gTrainingGridXY*self.gTrainingGridXY*self.gTrainingGridZ

    self.SubBatchSize = 1024

    self.NTrainingBatches = 1
    self.TrainingBatchSize = self.NTrainingBatches*self.SubBatchSize

    self.NTestingBatches = 1
    self.TestBatchSize = self.NTestingBatches*self.SubBatchSize



###################################################################################################


  def Plot2D(self, XSingle, YSingle, Title, FigureNumber = 0):
    '''
    A function for plotting 4 slices of the model in one figure
    '''

    XV, YV = np.meshgrid(self.gGridCentersXY, self.gGridCentersXY)
    Z = np.zeros(shape=(self.gTrainingGridXY, self.gTrainingGridXY))
    
    fig = plt.figure(FigureNumber)
    plt.clf()
    plt.subplots_adjust(hspace=0.5)

    fig.canvas.set_window_title(Title)

    for i in range(1, 5):    

      zGridElement = int((i-1)*self.gTrainingGridZ/4)

      for x in range(self.gTrainingGridXY):
        for y in range(self.gTrainingGridXY):
          Z[x, y] = YSingle[0, x + y*self.gTrainingGridXY + zGridElement*self.gTrainingGridXY*self.gTrainingGridXY]
      
      ax = fig.add_subplot(2, 2, i)
      ax.set_title("Slice through z={}".format(self.gGridCentersZ[zGridElement]))
      contour = ax.contourf(XV, YV, Z)  

    plt.ion()
    plt.show()
    plt.pause(0.001)
    
    if FigureNumber == 1:
      plt.savefig(self.OutputPrefix + "_Original.png")
    else:
      plt.savefig(self.OutputPrefix + "_Result.png")


###################################################################################################


  def Gauss3D(self, X, x0, y0, R):
    '''
    Return a donghnut with a Gaussian shape
    
    Args:
      X (float, float): Tuple of the x, y position
      x0 (float): x center
      y0 (float): x center
      R (float):  radius
    '''
    x, y = X
    return np.exp(((-np.sqrt((x-x0)**2 + (y-y0)**2) - R)**2)/self.gSigmaR)



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


  def getGauss(self, d, sigma = 1):
    '''
    Return a 1D Gaussian value
    
    Args:
      d (float):      Distance from 0
      sigma (float):  Sigma value of Gaussian
     
    '''
    
    return 1/(sigma*math.sqrt(2*np.pi)) * math.exp(-0.5*pow(d/sigma, 2))


###################################################################################################


  def CreateFullResponse(self, PosX, PosY):
    '''
    Create the response for a source at position PosX, PosY
    
    Args:
      PosX (float): x position of the source
      PosY (float): y position of the source
    
    '''
    
    Out = np.zeros(shape=(1, self.OutputDataSpaceSize))

    for x in range(0, self.gTrainingGridXY):
      for y in range(0, self.gTrainingGridXY):
        for z in range(0, self.gTrainingGridZ):
          r = math.sqrt((PosX - self.gGridCentersXY[x])**2 + (PosY - self.gGridCentersXY[y])**2 )
          Out[0, x + y*self.gTrainingGridXY + z*self.gTrainingGridXY*self.gTrainingGridXY] = self.getGauss(math.fabs(r - self.gGridCentersZ[z]), self.gSigmaR);
    return Out


###################################################################################################


  def train(self):
    '''
    Perfrom the neural network training
    '''
        
    print("Info: Creating %i data sets" % (self.TrainingBatchSize + self.TestBatchSize))

        
    XTrain = np.zeros(shape=(self.TrainingBatchSize, self.InputDataSpaceSize))
    YTrain = np.zeros(shape=(self.TrainingBatchSize, self.OutputDataSpaceSize))
    for i in range(0, self.TrainingBatchSize):
      if i > 0 and i % 128 == 0:
        print("Training set creation: {}/{}".format(i, self.TrainingBatchSize))
      XTrain[i,0] = random.uniform(self.gMinXY, self.gMaxXY)
      XTrain[i,1] = random.uniform(self.gMinXY, self.gMaxXY)
      YTrain[i,] = self.CreateFullResponse(XTrain[i,0], XTrain[i,1])     

    XTest = np.zeros(shape=(self.TestBatchSize, self.InputDataSpaceSize)) 
    YTest = np.zeros(shape=(self.TestBatchSize, self.OutputDataSpaceSize)) 
    for i in range(0, self.TestBatchSize):
      if i > 0 and i % 128 == 0:
        print("Testing set creation: {}/{}".format(i, self.TestBatchSize))
      XTest[i, 0] = random.uniform(self.gMinXY, self.gMaxXY)
      XTest[i, 1] = random.uniform(self.gMinXY, self.gMaxXY)
      YTest[i, ] = self.CreateFullResponse(XTest[i,0], XTest[i,1])
    
    
    print("Info: Setting up neural network...")

    Model = tf.keras.Sequential()
    Model.add(tf.keras.layers.Dense(10, activation="relu", kernel_initializer='he_normal', input_shape=(self.InputDataSpaceSize,)))
    Model.add(tf.keras.layers.Dense(100, activation="relu", kernel_initializer='he_normal'))
    Model.add(tf.keras.layers.Dense(1000, activation="relu", kernel_initializer='he_normal'))
    Model.add(tf.keras.layers.Dense(self.OutputDataSpaceSize))
    
    Model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=1e-08), loss=tf.keras.losses.MeanSquaredError(), metrics=['mse', 'mae', 'mape'])
    
    Model.summary()
    
    print("Info: Training and evaluating the network")

    # Train the network
    Timing = time.process_time()

    XSingle = XTest[0:1]
    YSingle = YTest[0:1]
    self.Plot2D(XSingle, YSingle, "Original", 1)


    TimesNoImprovement = 0
    BestMeanSquaredError = 10**30 #sys.float_info.max

    for Iteration in range(0, 50000):
      
      # Take care of Ctrl-C -- does not work
      global Interrupted
      if Interrupted == True: break

      # Train
      history = Model.fit(XTrain, YTrain, verbose=2, batch_size=self.TrainingBatchSize,  validation_data=(XTest, YTest))

        
      # Check performance: Mean squared error
      if Iteration > 0 and Iteration % 20 == 0:
        #MeanSquaredError = sess.run(tf.nn.l2_loss(Output - YTest)/self.TestBatchSize,  feed_dict={X: XTest})
        RunOut = Model.predict(XTest)
        MeanSquaredError = math.sqrt(np.sum((YTest - RunOut)**2))
  
        print("Iteration {} - MSE of test data: {}".format(Iteration, MeanSquaredError))

        if MeanSquaredError <= BestMeanSquaredError:    # We need equal here since later ones are usually better distributed
          BestMeanSquaredError = MeanSquaredError
          TimesNoImprovement = 0
          
          #Saver.save(sess, "model.ckpt")
          
          # Test just the first test case:
          YOutSingle = Model.predict(XSingle)
          
          self.Plot2D(XSingle, YOutSingle, "Reconstructed at iteration {}".format(Iteration), 2)
          
        else:
          TimesNoImprovement += 1
      else:
        plt.pause(0.001)

 
        # end: check performance


      if TimesNoImprovement == 100:
        print("No improvement for 100 rounds")
        break;
      
      # end: iterations loop

    YOutTest = sess.run(Output, feed_dict={X: XTest})


    Timing = time.process_time() - Timing
    if Iteration > 0: 
      print("Time per training loop: ", Timing/Iteration, " seconds")

    input("Press [enter] to EXIT")

    return



# END  
###################################################################################################


