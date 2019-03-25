import torch
import torch.nn.functional as F
import numpy as np

from data_helper import data_loader

train_x, train_y, test_x, test_y = data_loader()
train_y = np.array(train_y)
test_y = np.array(test_y)
train_y_in, train_y_out = train_y[:, 0], train_y[:, 1]
test_y_in, test_y_out = test_y[:, 0], test_y[:, 1]

train_x = torch.Tensor(train_x)
train_y = torch.Tensor(train_y)
LR = 0.01


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.hidden_1 = torch.nn.Linear(n_feature, 50)
        self.hidden_2 = torch.nn.Linear(50, 40)
        self.hidden_3 = torch.nn.Linear(40, 30)
        self.hidden_4 = torch.nn.Linear(30, 20)
        self.predict = torch.nn.Linear(20, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.relu(self.hidden_4(x))
        x = self.predict(x)  # linear output
        return x


net = Net(n_feature=5, n_output=2)
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
loss_func = torch.nn.L1Loss()


def train():
    for t in range(20000):
        prediction = net(train_x)  # input x and predict based on x

        loss = loss_func(prediction, train_y)  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if t % 5 == 0:
            print('step-{step}, loss-{loss}'.format(step=t, loss=loss))


if __name__ == '__main__':
    train()
