from re import U, template
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.init as init
import math
import scipy.stats as st

class NIT:

    def __init__(self, dataset = None):
        self.n_epoch = 10
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
        #
        k = 200
        data3 = torch.zeros([k,self.dataset.shape[0],self.dataset.shape[1]], dtype=torch.float, device=self.device)
        for i in range(k):
            index = torch.randperm(data2.shape[0])
            data3[i,:,:] = data2[index]
        #
        break_limit = 3
        break_flag = 0
        score_flag = 0
        #
        for epoch in range(self.n_epoch):
            self.f.train()
            self.g.train()
            self.f.zero_grad()
            self.g.zero_grad()
            target1 = self.f(data1)
            target2 = self.g(data2)
            kurt_xy = get_kurt(target1, target2)
            if fisher_test(target1, target2) == 0:
                score = kurt_xy * 100000000
                break
            min_kurt_xz = 10000
            max_kurt_xz = 0
            for i in range(k):
                target3 = self.g(data3[i,:,:])
                kurt_xz = get_kurt(target1, target3)
                if kurt_xz > max_kurt_xz:
                    max_kurt_xz = kurt_xz
                if kurt_xz < min_kurt_xz:
                    min_kurt_xz = kurt_xz
            score1 = min_kurt_xz - kurt_xy
            score2 = cc_square(target1, target2)
            score = score2 + max(score1,0)*10
            #
            if score > score_flag:
                score_flag = score
                break_flag = 0
            else:
                break_flag = break_flag + 1
            if score > 0.01 or fisher_test(target1, target2) == 0:
                score = score * 100000000
                break
            if break_flag == break_limit:
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

def fisher_test(x, y): 
    alpha = 0.005
    num_samples = x.shape[0]
    pcc = get_Corr(x, y)
    zpcc = 0.5*torch.log((1+pcc)/(1-pcc))
    A = math.sqrt(num_samples - 3) * torch.abs(zpcc)
    B = st.norm.ppf(1-alpha/2) # Inverse Cumulative Distribution Function of normal Gaussian (parameter : 1-alpha/2)
    if A>B:
        return 0
    else:
        return 1

def get_Corr(x, y):
    cov = (x * y).mean() - x.mean() * y.mean()
    var_x = x.var(unbiased=False)
    var_y = y.var(unbiased=False)
    return torch.sqrt(cov**2 / (var_x * var_y))

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

def kurt_k(x, y):
    scaleCoef = (x.std()/y.std())
    y = y*scaleCoef
    x = x.detach().numpy() 
    y = y.detach().numpy() 
    x = x.reshape(len(x))
    y = y.reshape(len(y))
    k = 100
    h = np.linspace(-1,1,10)
    lenh = len(h)
    u = np.zeros([1,lenh])
    v = np.zeros([k,lenh])
    sumv = u = np.zeros([1,lenh]);
    for i in range(k):
        for j in range(lenh):
            index = np.random.permutation(len(x))
            z = y[index]
            v[i,j] = get_kurtosis(x + h[j]*z)
            if i == 1:
                u[0,j] = get_kurtosis(x + h[j]*y)
        sumv = sumv + v[i,:]
    sumv = sumv/k
    cu = get_corr_square(u, sumv) 
    cv_min = 10
    cv_max = 0
    for m in range(k):
        temp = get_corr_square(v[m,:], sumv) 
        if temp > cv_max:
            cv_max = temp
        if temp < cv_min:
            cv_min = temp
    print(cu,cv_min,cv_max,cv_min - cu)        
    return cu
    
def get_kurt(x, y):
    scaleCoef = (x.std()/y.std())
    # print(scaleCoef)
    u = x + y*scaleCoef
    kurt_u = torch.mean((u-torch.mean(u))**4) / (torch.std(u))**2  
    v = x - y*scaleCoef
    kurt_v = torch.mean((v-torch.mean(v))**4) / (torch.std(v))**2  
    return abs(kurt_u)+abs(kurt_v)

def get_kurtosis(v):
    kurt = np.mean((v-np.mean(v))**4) / (np.std(v))**2
    return kurt

def get_corr_square(x, y):
    cov = (x * y).mean() - x.mean() * y.mean()
    var_x = x.var()
    var_y = y.var()
    return cov**2 / (var_x * var_y)

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
    id = np.floor(0.95*k)
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
    ind = True
    score = NIT(dataset=test_data)
    val = score.nit_test()
    if val > bar:
        ind = False 
    else:
        ind = True
    return ind