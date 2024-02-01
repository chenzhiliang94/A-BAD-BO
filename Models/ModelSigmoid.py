import torch
from torch.special import *
from Models.Model import Model

class ModelSigmoid(Model):
    def __init__(self, theta_0_=1., lr=0.01, tol = 1e-05, dtype=torch.float64):
        super().__init__(lr=lr, tol=tol, dtype = dtype)
        self.set_params([theta_0_])

    def evaluate(self, x, theta_0_):
        return expit(theta_0_ * x)