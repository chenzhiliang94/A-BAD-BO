import torch
from Models.Model import Model

class ModelSinCos(Model):
    def __init__(self, theta_0_=1., theta_1_=1., lr=0.01, tol = 1e-05, dtype=torch.float64):
        super().__init__(lr=lr, tol=tol, dtype = dtype)
        self.set_params([theta_0_, theta_1_])

    def evaluate(self, x, theta_0_, theta_1_):
        return theta_0_ * torch.sin(theta_1_ * x + 0.5)