import numpy as np
from numpy import ndarray
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from datetime import datetime

iris = load_breast_cancer()
inputs: ndarray = iris.data
targets: ndarray = iris.target
print(f'{inputs.shape}, {targets.shape}')


inputs_train: ndarray
inputs_test: ndarray
targets_train: ndarray
targets_test: ndarray
inputs_train, inputs_test, targets_train, targets_test = \
    train_test_split(inputs, targets, test_size = 0.3, random_state = 2021)


model = Sequential([
    Dense(64, input_dim = 30, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


start = datetime.now()
model.fit(inputs_train, targets_train, epochs = 100, verbose = 1)
print(f'Time taken: {datetime.now() - start}')