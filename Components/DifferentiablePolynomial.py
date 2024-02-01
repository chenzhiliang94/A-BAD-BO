import torch
from Models.Model import Model

class DifferentiablePolynomial(Model):
    '''
    x*theta_0-theta_1*x+e^-0.3x
    '''
    def __init__(self, theta_0_=1, theta_1_=1, theta_2_=1, lr=0.01, tol = 1e-05, dtype=torch.float64):
        super().__init__(lr=lr, tol=tol, dtype = dtype)
        self.set_params([theta_0_,theta_1_,theta_2_])

    def evaluate(self, x, theta_0_, theta_1_, theta_2_):
        '''

        :param x: scalar form
        :return: function
        '''
        return -(x)*theta_0_+theta_1_*x+torch.exp(theta_2_ * x)