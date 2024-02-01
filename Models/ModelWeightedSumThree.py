import torch
from Models.Model import Model

class ModelWeightedSumThree(Model):
    def __init__(self, inputs=3, theta_0_=1., theta_1_=1., theta_2_=1.0, lr=0.01, tol = 1e-05, dtype=torch.float64):
        super().__init__(inputs=inputs, lr=lr, tol=tol, dtype = dtype)
        self.set_params([theta_0_,theta_1_,theta_2_])

    def evaluate(self, x, theta_0_, theta_1_, theta_2):
        return theta_0_ * x[0] + theta_1_ *  x[1] + theta_2 * x[2]