from numpy import ndarray
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch import Tensor, relu, sigmoid, FloatTensor
from torch.nn import Module, Linear, BCEWithLogitsLoss
from torch.optim import Adam


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

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = relu(self.linear_1(inputs))
        return sigmoid(self.linear_2(inputs))


epochs: int = 100
model: Model = Model(30)
optimizer: Adam = Adam(model.parameters())
criterion: BCEWithLogitsLoss = BCEWithLogitsLoss()


X_train = FloatTensor(inputs_train)
y_train = FloatTensor(targets_train)


for epoch in range(epochs):
    optimizer.zero_grad()

    predictions: Tensor = \
        model(X_train)
 
    loss: Tensor = \
        criterion(predictions, y_train.reshape(-1, 1))
 
    # round
    # predicted = model(X_train)
    # acc = (predicted.reshape(-1).detach().numpy().round() == y_train.detach().numpy()).mean()
    # predicted = model(X_train)
    # acc = (predictions.reshape(-1).round() == y_train).mean()
    y_prob = predictions > 0.5
    acc = (targets_train == y_prob).sum().item() / targets_train.size(0)
 
    loss.backward()

    optimizer.step()

    print(f"epoch {epoch + 1}, loss : {loss:.4f}, accuracy : {acc:.4f}")