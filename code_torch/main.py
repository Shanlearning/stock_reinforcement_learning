# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 21:08:16 2022

@author: zhong
"""
import os
os.chdir("C:\\Users\\zhong\\Dropbox\\github\\stock_reinforcement _learning\\code_torch")

import numpy as np
import torch
from torch import nn
import torch.distributions as ptd
from scipy.spatial import distance
import matplotlib.pyplot as plt
import json

_input_val =  ['return_t','cci','macdh','rsi_14','kdjk','wr_14', 'atr_percent', 'cmf']
_output_val = ['return_t_plus_1']

from env import StockEnv

env = StockEnv()
###############################################################################

def sample_from_environment(env,training = True):
    X = []
    y = []
    for i in range(1024):
        env.reset(training=True)
        X.append( torch.tensor(np.asarray(env.state[_input_val])) )
        y.append( torch.tensor(np.asarray(env.state[_output_val])) )
    X = torch.stack(X)
    y = torch.stack(y)
    return X.to(device), y.to(device)

###############################################################################
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

###############################################################################

def se_kernel_fast(U,V,ell=1,tau=1.0,nugget = 0.0001):
    d = distance.cdist(U.cpu().data.numpy(),V.cpu().data.numpy(),'euclidean')
    d = np.power(d,2.0)
    kuv = np.power(tau,2.0)*np.exp(-d*np.power(ell,-2.0))
    if(np.allclose(kuv, kuv.T)):
        kuv = kuv + nugget*np.eye(len(kuv))
    return torch.tensor(kuv).float().to(device)

###############################################################################

# Define model
class my_basis_function(nn.Module):
    def __init__(self):
        super(my_basis_function, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5*8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        flatten_x = self.flatten(x)
        mean_and_logstd = self.linear_relu_stack(flatten_x)
        return mean_and_logstd

class my_policy(nn.Module):
    def __init__(self, basis_function):
        nn.Module.__init__(self)
        self.basis_function = basis_function
        self.basis_function.to(device)

    def action_distribution(self, x):
        r_and_logstd = self.basis_function(x)
        r_and_logstd = torch.split(r_and_logstd, 5, dim = 1)
        r = r_and_logstd[0]
        std = torch.exp(r_and_logstd[1])
        
        cov_matrix = []
        for i in range(x.shape[0]):
            cov_matrix.append( torch.matmul(torch.matmul( torch.diag(std[i]),se_kernel_fast(x[i],x[i]) ),torch.diag(std[i]) )  )
            
        distribution = ptd.multivariate_normal.MultivariateNormal(loc = r,covariance_matrix = torch.stack(cov_matrix) )        
        return distribution

def train(env,policy,optimizer):    
    all_log_likihood = []
    policy.train()
    # train with 1 million samples
    for batch in range(1000):
        X, y = sample_from_environment(env)
        
        # Compute prediction error
        distribution = policy.action_distribution(X.float())     
        
        negative_log_likihood = - torch.mean( distribution.log_prob(y.flatten(start_dim=1)) )
        
        optimizer.zero_grad()
        negative_log_likihood.backward()
        optimizer.step()
        
        if batch % 10 == 0:
            negative_log_likihood, current = negative_log_likihood.item(), batch * len(X)
            log_likihood = - negative_log_likihood
            print(f"log_likihood: {log_likihood:>7f}  [{current:>5d}]")
            all_log_likihood.append(log_likihood)
            plt.plot(all_log_likihood)
            plt.show()
            with open('results\\' + str(env.year) + '.json', 'w') as fp:
                json.dump(all_log_likihood, fp)  
        
def test(env,policy):
    policy.eval()
    result = []
    for batch in range(10):
        X, y = sample_from_environment(env,training = False)
        distribution = policy.action_distribution(X.float())     
        distribution_sum = ptd.Normal( loc = torch.mean(distribution.mean,dim=1), scale = torch.sqrt(torch.mean(distribution.covariance_matrix,dim = [1,2])) )
        
        # 95% prediction inerval
        _within_interval = torch.logical_and( (distribution_sum.mean - 1.96 * distribution_sum.stddev <= torch.mean(y.flatten(start_dim=1),dim=1)),
                          ( torch.mean(y.flatten(start_dim=1),dim=1) <= distribution_sum.mean + 1.96 * distribution_sum.stddev ))
        
        result.append(np.mean(_within_interval.cpu().data.numpy()))
    return result

basis_function = my_basis_function()
policy = my_policy(basis_function)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

result = []
for year in range(2002,2020):
    env.update_year(year)
    env.reset()
    train(env,policy,optimizer)
    result.append( np.mean( test(env,policy) ) )
    with open('results\\' + "prediction_interval" +str(env.year) + '.json', 'w') as fp:
        json.dump(result, fp)  

plt.plot(range(2002,2020),result)
