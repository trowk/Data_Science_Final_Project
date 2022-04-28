import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv1D, MaxPooling1D, UpSampling1D, Conv1DTranspose
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from numpy import newaxis
import csv

data = pd.read_csv("S_train_normalized.csv", header=None, skiprows=[15005])
data = pd.DataFrame(data)
ds   = data.to_numpy()

print("Input shape: \n")
print(ds.shape)

ds = ds[...,newaxis]

print('\n Input row 1: \n')
print(ds[0])

input_shape = (20,1)

encoder_input = Input(shape = input_shape)
y = Conv1D(8,3, activation = 'relu')(encoder_input) #18x8
y = Conv1D(16,3, activation = 'relu')(y) #16x16
y = Conv1D(32,3, activation = 'relu')(y) #14x32
y = Conv1D(64,3,activation = 'relu')(y) #12x64
y = Conv1D(128,3,activation = 'relu')(y) #10x128
y = Flatten()(y) #1280x1
encoder_output = Dense(3, activation = 'relu')(y) #2x1

#encoder = Model(inputs = encoder_input, outputs = encoder_output)

decoder_input = encoder_output
y = Dense(1280, activation = 'relu')(decoder_input) #1280x1
y = Reshape((10,128))(y) #10x128
y = Conv1DTranspose(64,3, activation = 'relu')(y) #12x64
y = Conv1DTranspose(32,3, activation = 'relu')(y) #14x32
y = Conv1DTranspose(16,3, activation = 'relu')(y) #16x16
y = Conv1DTranspose(8,3, activation = 'relu')(y) #18x8
decoder_output = Conv1DTranspose(1,3, activation = None)(y) #20x1

autoencoder = Model(encoder_input, decoder_output)
#autoencoder.summary()

opt = Adam(lr = 0.001, decay=1e-6)
autoencoder.compile(opt, loss = 'mse')

autoencoder.fit(ds,ds, epochs = 30, batch_size = 100, validation_split = 0.2)

example = autoencoder.predict([ds[0].reshape(-1,20,1)])

print('\n output of autoencoder for row 1: \n')
print(example)

np.savetxt('CNN_test.csv',example[0,...],delimiter=',')
