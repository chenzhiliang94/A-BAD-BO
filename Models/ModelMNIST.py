import torch
from torch.special import *
from Models.Model import Model
#pytorch utility imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid

#neural net imports
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np



class Net(nn.Module):
    def __init__(self, output_size):
        super(Net, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        self.linear_block = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(128*7*7, 128),
            #nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            #nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        
        return x

class ModelMNIST(Model):
    def __init__(self, local_train_loader : DataLoader, system_train_loader : DataLoader, output_size = 10, lr=0.003, tol = 1e-05, dtype=torch.float64):
        super().__init__(lr=lr, tol=tol, dtype = dtype)
        # if torch.cuda.is_available():
        #     self.conv_model = Net(output_size).to("cuda:0")
        self.conv_model = Net(output_size)
        self.local_train_loader = local_train_loader
        for batch_idx, (input, target) in enumerate(local_train_loader):
            self.X = input
            self.y = target
            break
        self.oracle_mode = False
    
    def is_nn(self):
        return True
    
    def nn_function(self):
        return
    
    def set_oracle_mode(self, bool_val):
        self.oracle_mode = bool_val

    def evaluate_score(self, x):
        return self.conv_model.forward(x)

    def evaluate(self, x):
        device = torch.device("cpu")
        if self.oracle_mode:
            return x # input will be just the label
        result = self.conv_model.forward(x.to(device)).max(1,keepdim=False)[1]
        return result
    
    def get_params(self):
        return [None]
    
    def test(self):
        print("hello")
    
    def get_local_loss(self):
        self.conv_model.eval()
        loss = 0

        data, target = self.X, self.y
        data = data.unsqueeze(1)
        
        # if torch.cuda.is_available():
        #     data = data.cuda()
        #     target = target.cuda()
        
        output = self.conv_model(data)
        
        criterion = nn.CrossEntropyLoss()
        cross_entropy_loss = criterion(output, target)
        loss += cross_entropy_loss

        return loss
    
    def do_one_descent_on_local(self):
        optimizer = optim.Adam(params=self.conv_model.parameters(), lr=0.0003)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        self.conv_model.train()
        
        data, target = self.X, self.y
        data = data.unsqueeze(1)
    
        # if torch.cuda.is_available():
        #     data = data.cuda()
        #     target = target.cuda()
        
        optimizer.zero_grad()
        output = self.conv_model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
                
    def descent_to_target_loss(self, target_loss):
        def my_loss(output, target, target_loss):
            criterion = nn.CrossEntropyLoss()
            cross_entropy_loss = criterion(output, target)
            return (target_loss - cross_entropy_loss)**2
        
        optimizer = optim.Adam(params=self.conv_model.parameters(), lr=0.005)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        self.conv_model.train()
        exp_lr_scheduler.step()
        data, target = self.X, self.y
        data = data.unsqueeze(1)
        data, target = data, target
    
        # if torch.cuda.is_available():
        #     data = data.cuda()
        #     target = target.cuda()
        
        # number of iteration
        for x in range(200):
            optimizer.zero_grad()
            output = self.conv_model(data)
            loss = my_loss(output, target, target_loss)
            loss.backward()
            optimizer.step()

    def random_initialize_param(self, seed=None):
        if seed is None:
            torch.manual_seed(np.random.randint(0,1000000))
        else:
            torch.manual_seed(seed)
        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()
                    
        self.conv_model.apply(fn=weight_reset)
    
    
