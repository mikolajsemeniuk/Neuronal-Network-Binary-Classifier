import numpy as np
from numpy import ndarray
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


breast_cancer = load_breast_cancer()
inputs: ndarray = breast_cancer.data
targets: ndarray = breast_cancer.target.reshape(-1, 1)
print(f'inputs: {inputs.shape}, targets: {targets.shape}')


inputs_train: ndarray
inputs_test: ndarray
targets_train: ndarray
targets_test: ndarray
inputs_train, inputs_test, targets_train, targets_test = \
    train_test_split(inputs, targets, test_size = 0.3, random_state = 2021)


class Layer_Dense:
      
    def __init__(self, in_dimensions: int, out_dimensions: int) -> None:
        # gain = np.sqrt(2.0 / (1 + np.sqrt(5) ** 2))
        # std = gain / np.sqrt(out_dimensions)
        # bound = np.sqrt(3.0) * std
        # self.weights: ndarray = np.random.uniform(low=-bound, high=bound, size=(in_dimensions, out_dimensions))

        # bound = 1 / np.sqrt(out_dimensions) if out_dimensions > 0 else 0
        # self.biases: ndarray = np.random.uniform(low=-bound, high=bound, size=(1, out_dimensions))

        limit = np.sqrt(6 / (in_dimensions + out_dimensions))
        self.weights: ndarray = np.random.uniform(low=-limit, high=limit, size=(in_dimensions, out_dimensions))
        self.biases: ndarray = np.zeros((1, out_dimensions))

        # self.weights: ndarray = np.random.randn(in_dimensions, out_dimensions) * np.sqrt(3 / in_dimensions)
        # self.weights: ndarray = np.random.randn(in_dimensions, out_dimensions)
        # self.biases: ndarray = np.zeros((1, out_dimensions))
      
    def forward(self, inputs: ndarray) -> None:
        self.inputs: ndarray = inputs
        self.output: ndarray = inputs @ self.weights + self.biases
      
    def backward(self, dvalues: ndarray) -> None:
        self.dweights: ndarray = self.inputs.T @ dvalues
        self.dbiases: ndarray = np.sum(dvalues, axis = 0, keepdims=True)
        self.dinputs: ndarray = dvalues @ self.weights.T


class Activation_ReLU:

    def forward(self, inputs: ndarray) -> None:
        self.inputs: ndarray = inputs
        self.output: ndarray = np.maximum(0, inputs)
    
    def backward(self, dvalues) -> None:
        self.dinputs: ndarray = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Sigmoid:
    
    def forward(self, inputs) -> None:
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues) -> None:
        self.dinputs = dvalues * (1 - self.output) * self.output


class Loss_BinaryCrossentropy:
    def calculate(self, output, y):
        return np.mean(self.forward(output, y))

    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(sample_losses, axis = -1)

    def backward(self, dvalues, y_true):
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / len(dvalues[0])
        self.dinputs = self.dinputs / len(dvalues)

class Optimizer_SGD:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

class Optimizer_Adam:

    def __init__(self, learning_rate=0.001, epsilon=1e-08,
        beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
            
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)


dense1 = Layer_Dense(30, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 1)
activation2 = Activation_Sigmoid()
loss_function = Loss_BinaryCrossentropy()
# optimizer = Optimizer_Adam()
optimizer = Optimizer_SGD()

for epoch in range(100):
    dense1.forward(inputs_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    loss = loss_function.calculate(activation2.output, targets_train)
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == targets_train)
    
    print(f"epoch {epoch + 1}, loss : {loss:.4f}, accuracy : {accuracy:.4f}")
    
    loss_function.backward(activation2.output, targets_train)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)