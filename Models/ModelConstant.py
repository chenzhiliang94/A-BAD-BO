import torch
from Models.Model import Model

class ModelConstant(Model):
    def __init__(self, theta_0=1, theta_1=1, lr=0.001, tol = 1e-05, dtype=torch.float64):
        super().__init__(lr=lr, tol=tol, dtype = dtype)
        self.set_params([theta_0, theta_1])

    def evaluate(self, x, theta_0_, theta_1_):
        return x * (1 + 0.0000001 * theta_0_ + 0.0000001 * theta_1_)