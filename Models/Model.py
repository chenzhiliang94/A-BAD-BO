import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def islist(obj):
    return isinstance(obj, list)

def recursive_map(fn, iterable):
    return list(map(lambda i: recursive_map(fn, i) if islist(i) else fn(i), iterable))

class Model(nn.Module):
    params = []
    X = None
    y = None
    lipschitz = 1
    oracle_mode = False
    def __init__(self, inputs=1, lr=0.01, tol = 1e-05, lipschitz=1, dtype=torch.float64):
        super().__init__()
        self.inputs = inputs
        self.lr = lr
        self.tol = tol
        self.dtype = dtype
        self.noisy_operation = (lambda y, n : y + n)
        self.lipschitz = lipschitz

    def is_nn(self):
        return False
    
    def attach_local_data(self, X, y):
        self.X = X
        self.y = y

    def get_local_loss(self):
        assert self.X != None and self.y != None, "no local data is found. Cannot get local loss!"
        output = self(self.X, grad=False, noise_mean=0.0, noise_std=0.0)
        mse = nn.MSELoss()
        loss = mse(output, self.y)
        return loss

    def to_param(self, params):
        return recursive_map(lambda p: nn.Parameter(p if torch.is_tensor(p) else torch.tensor(p, dtype=self.dtype)), params)

    def set_params(self, params):
        self.params = self.to_param(params)
        self.params_list = nn.ParameterList(self.params)

    def get_params(self):
        return recursive_map(lambda p: p.item(), self.params)

    def get_gradients_default(self, x):
        # dy/d_theta with theta from class attribute
        return self.get_gradients(x, *self.params)

    def get_gradients_combined(self, x, theta):
        # dy/d_theta
        return self.get_gradients(x, *theta)

    def get_gradients(self, x, params = None):
        if not params is None:
            assert self.check_param(params)
            params = self.to_param(params)
        else:
            params = self.params
        recursive_map(lambda p: p.grad.zero_() if p.grad else None, params)
        y = self.evaluate(x, *params)
        y.backward()
        g = recursive_map(lambda p: p.grad, params)
        return g

    def do_one_descent_on_local(self,lr=None):
        input_lr = self.lr
        if lr != None:
            input_lr = lr
        # if torch.cuda.is_available():
        #     self.X = self.X.cuda()
        #     self.y = self.y.cuda()
        optimizer = optim.Adam(self.params, lr=input_lr)
        optimizer.zero_grad()
        output = self(self.X, grad=True)
        mse = nn.MSELoss()
        loss = mse(output, self.y)
        loss.backward()
        optimizer.step()

    def do_one_ascent_on_local(self):
        # if torch.cuda.is_available():
        #     self.X = self.X.cuda()
        #     self.y = self.y.cuda()
        optimizer = optim.Adam(self.params, lr=self.lr)
        optimizer.zero_grad()
        output = self(self.X, grad=True)
        mse = nn.MSELoss()

        def my_loss(output, target):
            loss = torch.mean((output - target) ** 2)
            return - loss

        loss = my_loss(output, self.y)
        loss.backward()
        optimizer.step()
    
    def descent_to_target_loss(self, target_loss):
        # if torch.cuda.is_available():
        #     self.X = self.X.cuda()
        #     self.y = self.y.cuda()
        def my_loss(output, target, target_loss):
            criterion = nn.MSELoss()
            mse_loss = criterion(output, target)
            return (target_loss - mse_loss)**2
        
        optimizer = optim.Adam(self.params, lr=self.lr)
        for x in range(100):
            optimizer.zero_grad()
            output = self(self.X, grad=True)
            loss = my_loss(output, self.y, target_loss)
            loss.backward()
            optimizer.step()
        
        

    def random_initialize_param(self,seed=None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(np.random.randint(0,10000))
        s = self.params
        s = np.random.uniform(-2, 2, size=len(s)).astype("float32")
        self.set_params(list(s))
    
    def set_default_param(self):
        s = self.get_params()
        s = np.random.uniform(1, 1, size=len(s))
        self.set_params(list(s))

    def fit(self, X, y, itr_max = 1000):
        optimizer = optim.Adam(self.params, lr = self.lr)
        all_theta = [self.get_params()]
        mse = nn.MSELoss()
        for _ in range(itr_max):
            optimizer.zero_grad()
            output = self(X, grad=True)
            loss = mse(output, y)
            loss.backward()
            optimizer.step()
            all_theta.append(self.get_params())
            if loss < self.tol:
                break
        return all_theta

    # Recursively check that the supplied params is the same length as the original params
    def check_param(self, params):
        def recursive_check(self_param, other_param):
            self_islist = islist(self_param)
            other_islist = islist(other_param)
            if self_islist != other_islist:
                return False
            if not self_islist and not other_islist:
                return True
            if len(self_param) != len(other_param):
                return False
            for s_p, o_p in zip(self_param, other_param):
                if not recursive_check(s_p, o_p):
                    return False
            return True
        return recursive_check(self.params, params)
    
    def evaluate(self, X, *params):
        return X

    def forward(self, X, params = [], grad = False, noisy=False, noise_mean = 0.0, noise_std = 0.05, noisy_operation = None):
        if self.inputs > 1:
            assert X.shape[0] == self.inputs
        if len(params) == 0:
            params = self.params
        else:
            assert self.check_param(params)  # If params is supplied, must be equal length to number of params
            params = self.to_param(params)
        y = self.evaluate(X, *params)
        if noisy:
            noise = torch.normal(torch.full_like(y,noise_mean), torch.full_like(y,noise_std))
            np.random.seed(None)
            noise = np.random.normal(noise_mean, noise_std)
            if noisy_operation is None:
                noisy_operation = self.noisy_operation
            y = noisy_operation(y, noise)
        if not grad:
            y = y.detach()
        return y