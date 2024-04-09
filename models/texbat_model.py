import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture based on InfoGAN paper.
"""

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(264, 256), #用线性变换将输入映射到256维
            nn.ReLU(True),       #relu激活
            nn.Linear(256, 256), #线性变换
            nn.ReLU(True),       #relu激活
            nn.Linear(256, 1000), #线性变换
            nn.Tanh()            #Tanh激活使得生成数据分布在【-1,1】之间
        )
 
    def forward(self, x):
        x = self.gen(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(1000,256),#输入特征数为1000，输出为256
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
        self.disc = nn.Linear(128, 2)
        self.mu = nn.Linear(128, 8)
        self.var = nn.Linear(128, 8)

    def forward(self, x):
        disc_logits = self.disc(x).squeeze()

        mu = self.mu(x).squeeze()
        var = torch.exp(self.var(x).squeeze())

        return disc_logits, mu, var
