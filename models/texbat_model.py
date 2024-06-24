import torch
import torch.nn as nn
import torch.nn.functional as F





def init_hidden(batch_size, hidden_size):
    init_hidden = torch.zeros((batch_size, hidden_size))
    return init_hidden
"""
Architecture based on InfoGAN paper.
"""



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(259, 256), #用线性变换将输入映射到256维
            nn.ReLU(True),       #relu激活
            nn.Linear(256, 256), #线性变换
            nn.ReLU(True),       #relu激活
            nn.Linear(256, 1024), #线性变换
            nn.Tanh()            #Tanh激活使得生成数据分布在【-1,1】之间
        )
 
    def forward(self, x):
        x = self.gen(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),#进行非线性映射
            nn.Linear(512,256),#输入特征数为1000，输出为256
            nn.LeakyReLU(0.2),#进行非线性映射
            nn.Linear(256,256),#进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256,128),
            nn.Sigmoid()#也是一个激活函数，二分类问题中，
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )
    def forward(self, x):
        x = self.dis(x)
        return x

class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dis = nn.Linear(128, 1)
    def forward(self, x):
        output = torch.sigmoid(self.dis(x))

        return output

class QHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Linear(128, 4)
        self.mu = nn.Linear(128, 8)
        self.var = nn.Linear(128, 8)

    def forward(self, x):
        disc_logits = self.disc(x).squeeze()

        mu = self.mu(x).squeeze()
        var = torch.exp(self.var(x).squeeze())

        return disc_logits, mu, var

## the RNN models for the TEXBAT dataset
class LSTMGenerator(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.

    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Input: noise of shape (batch_size, in_dim)
    Output: sequence of shape (batch_size, out_dim)
    """
    def __init__(self, input_shape=259, output_dim=1000, rnn_hidden_dim=256):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, output_dim)

    def forward(self, noise, hidden):
        x = F.relu(self.fc1(noise))
        h_in = hidden.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        output = self.fc2(h)
        return output, h



class LSTMDiscriminator(nn.Module):
    """An LSTM based discriminator. It expects a sequence as input and outputs a probability for each element. 

    Args:
        in_dim: Input noise dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Inputs: sequence of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, 1)
    """

    def __init__(self, input_shape=1000, output_dim = 128, rnn_hidden_dim=256):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, output_dim)
        self.leakyRelu = nn.LeakyReLU(0.2)

    def forward(self, input, hidden):
        x = self.leakyRelu(self.fc1(input))
        h_in = hidden.reshape(-1, self.rnn_hidden_dim)
        h = self.leakyRelu(self.rnn(x, h_in))
        output = torch.sigmoid(self.fc2(h))
        return output, h

class LSTMDHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dis = nn.Linear(128, 1)
    def forward(self, x):
        output = torch.sigmoid(self.dis(x))

        return output

class LSTMQHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Linear(128, 4)
        self.mu = nn.Linear(128, 8)
        self.var = nn.Linear(128, 8)

    def forward(self, x):
        disc_logits = self.disc(x).squeeze()

        mu = self.mu(x).squeeze()
        var = torch.exp(self.var(x).squeeze())

        return disc_logits, mu, var


class CHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(128,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 4)
        )
        self.mu = nn.Linear(128, 8)
        self.var = nn.Linear(128, 8)

    def forward(self, x):
        disc_logits = self.disc(x).squeeze()

        mu = self.mu(x).squeeze()
        var = torch.exp(self.var(x).squeeze())

        return disc_logits, mu, var