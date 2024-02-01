from collections import namedtuple
import numpy as np
from sklearn.metrics import mean_squared_error

Component = namedtuple("Component", "item")
Model = namedtuple("Model", "item local_X local_y")

class SequentialSystem():
    '''
    Assume an ordered sequential linear system
    '''

    # ordered components; can contain model or black box models
    # each model comes with a local dataset
    # example: [(model1, X1, y1), component, (model2, X1, y1)]
    all_components = []
    global_X = None
    global_y = None

    def addModel(self, model, X_local, y_local):
        # adds a model and local dataset
        model.attach_local_data(X_local, y_local)
        self.all_components.append(Model(model, X_local, y_local))

    def addComponent(self, component):
        # adds a component
        self.all_components.append(Component(component))

    def addGlobalData(self, X, y):
        self.global_X = X
        self.global_y = y

    def fit_global_differentiable(self):
        # train composite function f.g on x,z ("global") (end to end)
        itr_max = 1000
        itr = 0

        all_theta_via_global = []

        for component_idx, component in enumerate(self.all_components):
            if isinstance(component, Component):
                continue # do not need to train components with no parameters
            while (itr < itr_max):
                theta = component.item.get_params()
                gradient_theta = np.array([0] * len(theta))

                idx = np.random.choice(np.arange(len(self.global_X)), 100, replace=False)
                for x, z in zip(self.global_X[idx], self.global_y[idx]):  # labeled data for training is using ground truth
                    d_dtheta = np.array(component.item.get_gradients(x, theta[0], theta[1])) # current
                    output = component.item.func(x)
                    grad, output = self.get_gradient_multiplication_and_forward_pass(component_idx+1, len(self.all_components), output)
                    gradient_theta = np.add((output - z) * grad * d_dtheta, gradient_theta)

                gradient_theta /= len(idx)

                if np.all(np.abs(gradient_theta)) <= 1e-03:
                    break

                theta -= 0.01 * gradient_theta
                all_theta_via_global.append(theta)
                itr += 1
                component.item.theta_0_ = theta[0]
                component.item.theta_1_ = theta[1]

                z_pred = self.predict(self.global_X.reshape(len(self.global_X), 1))
                print("error: ", mean_squared_error(z_pred, self.global_y))

        return all_theta_via_global

    def get_gradient_multiplication_and_forward_pass(self, start_component_idx, end_component_idx, input):
        # end idx exclusive
        grad_multiplication = 1
        input_next = input
        for idx in range(start_component_idx, end_component_idx):
            grad_multiplication *= self.all_components[idx].item.get_gradients_default(input_next)
            input_next = self.all_components[idx].item.func(input_next)
        return grad_multiplication, input_next

    def predict(self, X):
        output = X
        for component in self.all_components:
            if isinstance(component, Model):
                output = component.item(output)
            else:
                # assume perturbation in black box component during forward pass
                output = component.item(output, noisy = True)
        return output

    def compute_system_loss(self):
        # when computing system loss, the forward pass is noisy
        z_pred = self.predict(self.global_X.reshape(len(self.global_X), 1))
        return mean_squared_error(z_pred, self.global_y)

    def compute_local_loss(self):
        local_loss = [] # need to find a way to assign unique id to each component instead of a list
        for idx in range(0, len(self.all_components)):
            if isinstance(self.all_components[idx], Model):
                local_loss.append(self.all_components[idx].item.get_local_loss())
        return local_loss

    def assign_parameters(self, list_of_params):
        list_of_params = list_of_params.flatten()
        # list_of_params: just a list
        param_idx = 0
        for idx in range(0, len(self.all_components)):
            if isinstance(self.all_components[idx], Model):
                num_param_to_assign = len(self.all_components[idx].item.get_params())
                self.all_components[idx].item.set_params(list_of_params[param_idx:param_idx+num_param_to_assign])
                param_idx += num_param_to_assign

    def get_parameters(self):
        all_params = []
        for idx in range(0, len(self.all_components)):
            if isinstance(self.all_components[idx], Model):
                all_params.append(self.all_components[idx].item.get_params())
        return all_params
