import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.init as init
import math
import scipy.stats as st

class NIT:

    def __init__(self, dataset = None):
        self.n_epoch = 100
        self.device = torch.device('cpu')
        self.d = dataset.shape[1]
        self.dataset = torch.FloatTensor(dataset).to(self.device)
        self.f = Test(self.d, num_layer=2, hidden_size=20).to(self.device)
        self.g = Test(self.d, num_layer=2, hidden_size=20).to(self.device)
        self.optim_f = torch.optim.Adam(self.f.parameters(), lr=1e-2)
        self.optim_g = torch.optim.Adam(self.g.parameters(), lr=1e-2)

    def nit_test(self): 
        data1 = torch.zeros(self.dataset.shape, dtype=torch.float, device=self.device)
        data2 = torch.zeros(self.dataset.shape, dtype=torch.float, device=self.device)
        data1[:, 0] = self.dataset[:, 0]
        data2[:, 1] = self.dataset[:, 1]
        for epoch in range(self.n_epoch):
            self.f.train()
            self.g.train()
            self.f.zero_grad()
            self.g.zero_grad()
            target1 = self.f(data1)
            target2 = self.g(data2)
            if epoch == 0:
                ts = get_threshold(target1, target2)
            score1 = cc_square(target1, target2)
            score2 = cc_square_sq(target1, target2)
            score3 = cc_kurt_plus(target1, target2)
            score4 = cc_kurt_minus(target1, target2)
            # score = max(score1,(score2/1.3 + max(score3 + score4-0.3,0)/100))# BEST HP IS 1,1.1,500   mark
            # score = max(score1,(score2/0.95 + max(score3 + score4-0.3,0)/200))# BEST HP IS 1,1.1,500 
            # score = max(score1,score2)
            # score = max(score1,max(score3+score4-0.6,0))
            score = max(score1,max(score3+score4-ts,0)/20)
            if score > 0.01 or epoch + 1 == self.n_epoch:
                # print('scoreNit = ',score2,score3+score4,score2/1.5+(score3+score4)/60)
                break
            loss = -(score)
            loss.backward()    
            self.optim_f.step()
            self.optim_g.step()
        return score.item()  

class Test(nn.Module):

    def __init__(self, input_size=5, num_layer=4, hidden_size=20):
        super().__init__()
        net = [
            snlinear(input_size, hidden_size, bias=False),
            nn.ReLU(True),
        ]
        for i in range(num_layer - 2):
            net.append(snlinear(hidden_size, hidden_size, bias=False))
            net.append(nn.ReLU(True))
        net.append(snlinear(hidden_size, 1, bias=False))
        self.net = nn.Sequential(*net)
        self.sample()

    def sample(self, ini_type='kaiming_unif'):
        if ini_type == 'kaiming_unif':
            self.apply(init_kaiming_unif)
        elif ini_type == 'kaiming_norm':
            self.apply(init_kaiming_norm)
        else: 
            self.apply(init_norm)

    def forward(self, x):
        return self.net(x)

def cc_square(x, y):
    cov = (x * y).mean() - x.mean() * y.mean()
    var_x = x.var(unbiased=False)
    var_y = y.var(unbiased=False)
    return cov**2 / (var_x * var_y)

def cc_square_sq(x, y):
    x = x*x
    y = y*y
    cov = (x * y).mean() - x.mean() * y.mean()
    var_x = x.var(unbiased=False)
    var_y = y.var(unbiased=False)
    return cov**2 / (var_x * var_y)

def cc_kurt_plus(x, y):
    scaleCoef = (x.std()/y.std())
    u = x + y*scaleCoef
    ku = torch.mean((u-torch.mean(u))**4) / (torch.std(u))**2
    index = torch.randperm(x.shape[0])
    z = y[index]
    v = x + z*scaleCoef
    kv = torch.mean((v-torch.mean(v))**4) / (torch.std(v))**2
    dis_kurt = abs((kv-ku)/max(abs(kv),abs(ku)))       
    return dis_kurt

def cc_kurt_minus(x, y):
    scaleCoef = (x.std()/y.std())
    u = x - y*scaleCoef
    ku = torch.mean((u-torch.mean(u))**4) / (torch.std(u))**2
    index = torch.randperm(x.shape[0])
    z = y[index]
    v = x - z*scaleCoef
    kv = torch.mean((v-torch.mean(v))**4) / (torch.std(v))**2
    dis_kurt = abs((kv-ku)/max(abs(kv),abs(ku)))       
    return dis_kurt

def get_threshold(x, y):
    k = 1000
    sM = np.zeros(k)
    for i in range(k):
        index = torch.randperm(x.shape[0])
        z = y[index]
        a = cc_kurt_plus(x, z)
        b = cc_kurt_minus(x, z)
        sM[i] = a.detach().numpy() + b.detach().numpy()
    sM = np.sort(sM)
    id = np.floor(0.99*k)
    id = sM[id.astype(int)]
    # id = max(sM) 
    return id

def snlinear(in_features, out_features, bias=False): 
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))

def init_kaiming_unif(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        init.kaiming_uniform_(m.weight)

def init_kaiming_norm(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        init.kaiming_normal_(m.weight)

def init_norm(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        init.normal_(m.weight, mean=0, std=1)

def main_nit(d1,d2): 

    data = np.zeros([2,len(d1)])
    data[0] = d1
    data[1] = d2
    test_data = data.transpose()
    bar = 0.01
    # print(bar)
    ind = True
    score = NIT(dataset=test_data)
    val = score.nit_test()
    if val > bar:
        ind = False 
    else:
        ind = True
    return ind