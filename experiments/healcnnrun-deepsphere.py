'''

This script attempts to train a CNN on healpix coordinates with the deepsphere library.

@version 11/3/22

'''


import os
import time
import pickle

import tensorflow as tf
import healpy as hp
import numpy as np


#framework used for healpix conv layers

from deepsphere import HealpyGCNN
from deepsphere import healpy_layers as hp_layer


from src.cone_model import ToyModel3DCone, HEALPixCone
from src.config import Config


config = Config()

# Config
config = Config()
config.model_type = 'conv'
config.loss_type = 'MSELoss'
config.metric_monitor = 'loss'
config.lr = 1e-3
config.dropout_rate = 0.0
config.train_batch_size = 1024
config.eval_batch_size = 1024
config.epoch = 5000
config.device = 'cpu'

# flattened = True for using fully-connected layers, False for using conv-based layers
config.flattened = False  
config.filter_size = 3
config.NSIDE = 32
config.exp_name = 'sphere_{}_{}_NSIDE{}_datasize1024'.format(
    config.model_type, config.loss_type, config.NSIDE)
config.working_dir = os.path.join('results', 
    '{}_{}'.format(config.exp_name, time.strftime('%m%d_%H-%M'))
)



cone_model = HEALPixCone(
    output_dir=os.path.join(config.working_dir, 'figs'),
    NSIDE=config.NSIDE)


"""
train_dset = cone_model.create_dataset(dataset_size=1024)
val_dset = cone_model.create_dataset(dataset_size=1024)
f = open(f"sphere_datasets_NSIDE{config.NSIDE}.pkl", "wb"); pickle.dump((train_dset, val_dset), f); f.close()
"""


f = open("sphere_datasets_NSIDE6.pkl", "rb"); train_dset, val_dset = pickle.load(f); f.close()

model1 = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(2,)),
  tf.keras.layers.Dense(12288, activation='relu'),
])



layers = [
    #tf.keras.layers.Input(shape=(2,)),
    #tf.keras.layers.Dense(12288, activation='relu'),
        hp_layer.HealpyChebyshev(K = 10, Fout=40),
        hp_layer.HealpyChebyshev(K=10, Fout=40),
        hp_layer.HealpyChebyshev(K=10, Fout=4),
         ]



indices = np.arange(hp.nside2npix(config.NSIDE))

model = HealpyGCNN(nside=config.NSIDE, indices=indices, layers=layers, n_neighbors=20)

