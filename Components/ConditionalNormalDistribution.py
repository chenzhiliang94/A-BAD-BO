import torch
from Models.Model import Model

class ConditionalNormalDistribution(Model):
    bins = []
    def __init__(self, bins=[0.3], mu=[1,2],std_dev=[0.1,0.15], dtype=torch.float64):
        assert len(bins) + 1 == len(mu)
        assert len(mu) == len(std_dev)
        super().__init__(dtype=dtype)
        self.set_params([mu, std_dev])
        self.bins = bins

    def evaluate(self, x, mu, std_dev):
        for idx, interval in enumerate(self.bins):
            if x <= interval:
                return mu[idx] + std_dev[idx]*torch.randn(x.shape[0])
        return mu[-1] + std_dev[-1]*torch.randn(x.shape[0])