import torch
from torch.special import *
from Models.Model import Model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

# output between (0,1)
class ModelLogistic(Model):
    def __init__(self, inputs, oracle_mode=False, lr=0.1, tol = 1e-05, dtype=torch.float64):
        self.oracle_mode = oracle_mode
        super().__init__(lr=lr, tol=tol, dtype = dtype)
        self.model = torch.nn.Linear(inputs, 1).double()
        # self.set_params([1.0]*inputs)
    
    def get_local_loss(self):
        self.model.eval()
        loss = 0

        data, target = self.X, self.y
        
        # if torch.cuda.is_available():
        #     data = data.cuda()
        #     target = target.cuda()
        
        output = self.evaluate_with_existing_params(data)
        criterion = nn.BCELoss()
        log_loss = criterion(output, target.reshape(output.shape).double())
        loss += log_loss

        return loss

    def set_params(self, params):
        self.model.weight = (nn.Parameter(torch.Tensor(params).double().reshape(self.model.weight.shape)))

    def get_params(self):
        return (self.model.weight).detach().flatten().numpy()
    
    def evaluate(self, x):

        if self.oracle_mode:
            return x # input will be just the label
        result = torch.sigmoid(self.model.double()(x.double()))

        return result
    
    def evaluate_with_existing_params(self, x):
        if self.oracle_mode:
            return x # input will be just the label
        result = torch.sigmoid(self.model.double()(x.double()))
        return result
    
    def do_one_descent_on_local(self):
        optimizer = optim.SGD(params=self.model.parameters(), lr=0.1)
        for x in range(10):
            optimizer.zero_grad()
            data, target = self.X, self.y
            output = self.evaluate_with_existing_params(data)
            loss = nn.BCELoss()(output, target.reshape(output.shape).double())
            loss.backward()
            optimizer.step()
    
    def accuracy(self):
        with torch.no_grad():
            correct = np.sum(torch.squeeze(self.evaluate_with_existing_params(self.X)).round().detach().numpy() == self.y.detach().numpy())
            accuracy = 100 * correct/len(self.y)
        return accuracy
    
    def do_one_ascent_on_local(self):
        def my_loss(output, target):
            loss =nn.BCELoss()(output, target)
            return - loss
    
        optimizer = optim.SGD(params=self.model.parameters(), lr=0.1)
        for x in range(10):
            optimizer.zero_grad()
            data, target = self.X, self.y
            output = self.evaluate_with_existing_params(data)
            loss = my_loss(output, target.reshape(output.shape).double())
            loss.backward()
            optimizer.step()
                
    def descent_to_target_loss(self, target_loss):
        def my_loss(output, target, target_loss):
            criterion = nn.BCELoss()
            cross_entropy_loss = criterion(output, target.reshape(output.shape).double())
            return (target_loss - cross_entropy_loss)**2
        
        optimizer = optim.Adam(params=self.model.parameters(), lr=0.1)
        data, target = self.X, self.y
    
        # if torch.cuda.is_available():
        #     data = data.cuda()
        #     target = target.cuda()
        
        # number of iteration
        for x in range(50):
            optimizer.zero_grad()
            output = self.evaluate(data)
            loss = my_loss(output, target.double(), target_loss)
            loss.backward()
            optimizer.step()
    
    def set_oracle_mode(self, bool_val):
        self.oracle_mode = bool_val
    
    def random_initialize_param(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(np.random.randint(0,10000))
        s = self.model.weight
        s = np.random.uniform(-1,1,size=s.data.shape)
        self.model.weight = (nn.Parameter(torch.from_numpy(s)))
        