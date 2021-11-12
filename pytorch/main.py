from numpy import ndarray
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch import Tensor, relu, sigmoid, FloatTensor
from torch.nn import Module, Linear, BCEWithLogitsLoss
from torch.optim import Adam, SGD
import torch

breast_cancer = load_breast_cancer()
inputs: ndarray = breast_cancer.data
targets: ndarray = breast_cancer.target
print(f'inputs: {inputs.shape}, targets: {targets.shape}')


inputs_train: ndarray
inputs_test: ndarray
targets_train: ndarray
targets_test: ndarray
inputs_train, inputs_test, targets_train, targets_test = \
    train_test_split(inputs, targets, test_size = 0.3, random_state = 2021)


class Model(Module):

    def __init__(self, inputs_dimension: int):
        super(Model, self).__init__()
        self.linear_1 = Linear(inputs_dimension, 64)
        self.linear_2 = Linear(64, 1)
        # torch.nn.init.xavier_uniform_(self.linear_1.weight)
        # torch.nn.init.xavier_uniform_(self.linear_2.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = relu(self.linear_1(inputs))
        return sigmoid(self.linear_2(inputs))


epochs: int = 100
model: Model = Model(30)
# optimizer: Adam = Adam(model.parameters(), lr=0.001)
optimizer: Adam = SGD(model.parameters(), lr=0.001)
criterion: BCEWithLogitsLoss = BCEWithLogitsLoss()

X_train = FloatTensor(inputs_train)
y_train = FloatTensor(targets_train)

for epoch in range(epochs):
    optimizer.zero_grad()

    predictions: Tensor = \
        model(X_train)

    loss: Tensor = \
        criterion(predictions, y_train.reshape(-1, 1))

    accuracy: Tensor = \
            (predictions.clone().reshape(-1).detach().numpy().round() == \
            y_train.clone().detach().numpy()).mean()

    loss.backward()

    optimizer.step()

    print(f"epoch {epoch + 1}, loss : {loss:.4f}, accuracy : {accuracy:.4f}")