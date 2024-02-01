import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

import numpy as np
# train_X = torch.rand(10, 2)
# Y = 1 - torch.norm(train_X - 0.5, dim=-1, keepdim=True)
# Y = Y + 0.1 * torch.randn_like(Y)  # add some noise
# train_Y = standardize(Y)
# print(train_X.shape)
# print(Y.shape)
# gp = SingleTaskGP(train_X, train_Y)
# mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
# fit_gpytorch_mll(mll)

np.random.seed(1)
print(np.random.uniform(1,0.05))