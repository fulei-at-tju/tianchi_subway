import torch
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np

from data_helper import data_loader2, mae

train_x, train_y, test_x, test_y = data_loader2()
train_y = np.array(train_y)
test_y = np.array(test_y)
train_y_in, train_y_out = train_y[:, 0], train_y[:, 1]
test_y_in, test_y_out = test_y[:, 0], test_y[:, 1]

train_x = torch.Tensor(train_x)
train_y = torch.Tensor(train_y)
test_x = torch.Tensor(test_x)
test_y = torch.Tensor(test_y)

LR = 0.01
BATCH_SIZE = 5

torch_dataset = Data.TensorDataset(train_x, train_y)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)


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


net = Net(n_feature=23, n_output=2)
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
loss_func = torch.nn.L1Loss()


def batch_train():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            prediction = net(batch_x)

            loss = loss_func(prediction, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                pre_test = net(test_x)
                in_num = pre_test.data.numpy()[:, 0]
                out_num = pre_test.data.numpy()[:, 1]
                score = (mae(in_num, test_y_in) + mae(out_num, test_y_in)) / 2
                loss2 = loss_func(pre_test, test_y)
                print('epoch:{epoch}  step:{step}  train:  loss-{loss}  test:  loss-{loss2}  loss-{loss3}'
                      .format(epoch=epoch, step=step, loss=loss, loss2=loss2, loss3=score))


def train():
    for t in range(20000):
        prediction = net(train_x)  # input x and predict based on x

        loss = loss_func(prediction, train_y)  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if t % 5 == 0:
            pre_test = net(test_x)
            in_num = pre_test.data.numpy()[:, 0]
            out_num = pre_test.data.numpy()[:, 1]
            score = (mae(in_num, test_y_in) + mae(out_num, test_y_in)) / 2
            loss2 = loss_func(pre_test, test_y)
            print('step-{step}  train:  loss-{loss}   test:  loss-{loss2}, loss-{loss3}'
                  .format(step=t, loss=loss, loss2=loss2, loss3=score))


if __name__ == '__main__':
    train()
