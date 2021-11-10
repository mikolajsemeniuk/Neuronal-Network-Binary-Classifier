# import numpy as np
# from numpy import ndarray
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from torch.nn import Module, Linear, Sigmoid, BCELoss
# from torch.nn.functional import relu
# import torch.nn.functional as F
# from torch.optim import Adam
# from torch import LongTensor, FloatTensor
# from torch import Tensor
# from datetime import datetime

# import torch

# iris = load_breast_cancer()
# inputs: ndarray = iris.data
# targets: ndarray = iris.target
# print(f'{inputs.shape}, {targets.shape}')


# inputs_train: ndarray
# inputs_test: ndarray
# targets_train: ndarray
# targets_test: ndarray
# inputs_train, inputs_test, targets_train, targets_test = \
#     train_test_split(inputs, targets, test_size = 0.3, random_state = 2021)


# class Model(Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.layer_1 = Linear(30, 64)
#         self.layer_2 = Linear(64, 1)

#     def forward(self, inputs: Tensor) -> Tensor:
#         # inputs = relu(self.layer_1(inputs))
#         # return Sigmoid()(self.layer_2(inputs))
#         inputs = torch.relu(self.layer_1(inputs))
#         inputs = torch.sigmoid(self.layer_2(inputs))
#         return inputs


# model = Model()
# criterion = torch.nn.BCELoss()
# optimizer = Adam(model.parameters(), lr = 0.1)


# X_train: Tensor = FloatTensor(inputs_train)
# X_test: Tensor = FloatTensor(inputs_test)
# y_train: Tensor = LongTensor(targets_train)
# y_test: Tensor = LongTensor(targets_test)


# start = datetime.now()

# for epoch in range(101):
#     optimizer.zero_grad()

#     predictions: Tensor = \
#         model.forward(inputs_train)

#     loss: Tensor = \
#         criterion(predictions, targets_train)

#     accuracy: Tensor = \
#         (targets_train == predictions).sum().item()

#     loss.backward()
#     optimizer.step()

#     print(f'epoch {epoch}, loss {loss.item():.4f}, accuracy: {accuracy:.4f}')

# print(f'Time taken: {datetime.now() - start}')

#importing the libraries
import torch
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
x = data['data']
y = data['target']
print("shape of x: {}\nshape of y: {}".format(x.shape,y.shape))

x_train = torch.FloatTensor(x)
y_train = torch.FloatTensor(y)

#feature scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x = sc.fit_transform(x)


#defining dataset class
# from torch.utils.data import Dataset, DataLoader
# class dataset(Dataset):
#   def __init__(self,x,y):
#     self.x = torch.tensor(x,dtype=torch.float32)
#     self.y = torch.tensor(y,dtype=torch.float32)
#     self.length = self.x.shape[0]
 
#   def __getitem__(self,idx):
#     return self.x[idx],self.y[idx]
#   def __len__(self):
#     return self.length
# trainset = dataset(x,y)
#DataLoader
# trainloader = DataLoader(trainset,batch_size=64,shuffle=False)


#defining the network
from torch import nn
from torch.nn import functional as F

class Net(nn.Module):

    def __init__(self, input_shape):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 32)
        self.fc10 = nn.Linear(32, 1)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.sigmoid(self.fc10(x))
        return x

#hyper parameters
epochs = 300
# Model , Optimizer, Loss
model = Net(30)
optimizer = torch.optim.SGD(model.parameters(), lr=.03)
loss_fn = nn.BCELoss()


for i in range(epochs):
#   for j, (x_train, y_train) in enumerate(trainloader):
    
    #calculate output
    output = model(x_train)
 
    #calculate loss
    loss = loss_fn(output, y_train.reshape(-1,1))
 
    #accuracy
    predicted = model(x_train)
    acc = (predicted.reshape(-1).detach().numpy().round() == y).mean()
    #backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))