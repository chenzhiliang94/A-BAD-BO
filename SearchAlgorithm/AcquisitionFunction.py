from botorch.acquisition import UpperConfidenceBound
import torch

from botorch.acquisition.objective import PosteriorTransform, ScalarizedPosteriorTransform
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor

from typing import Union, Optional

# class UpperConfidenceBoundWithLocalLoss(AnalyticAcquisitionFunction):
#     r"""Single-outcome Upper Confidence Bound (UCB).
#
#     Analytic upper confidence bound that comprises of the posterior mean plus an
#     additional term: the posterior standard deviation weighted by a trade-off
#     parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
#     selection of design points). The model must be single-outcome.
#
#     `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
#     posterior mean and standard deviation, respectively.
#
#     Example:
#         >>> model = SingleTaskGP(train_X, train_Y)
#         >>> UCB = UpperConfidenceBound(model, beta=0.2)
#         >>> ucb = UCB(test_X)
#     """
#
#     def __init__(
#         self,
#         model: Model,
#         modelA, modelB, X_local, y1_local, y2_local_ground_truth, z_local_ground_truth,
#         beta: Union[float, Tensor],
#         posterior_transform: Optional[PosteriorTransform] = None,
#         maximize: bool = True,
#         **kwargs,
#     ) -> None:
#         r"""Single-outcome Upper Confidence Bound.
#
#         Args:
#             model: A fitted single-outcome GP model (must be in batch mode if
#                 candidate sets X will be)
#             beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
#                 representing the trade-off parameter between mean and covariance
#             posterior_transform: A PosteriorTransform. If using a multi-output model,
#                 a PosteriorTransform that transforms the multi-output posterior into a
#                 single-output posterior is required.
#             maximize: If True, consider the problem a maximization problem.
#         """
#         super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
#         self.X_local = X_local
#         self.y1_local = y1_local
#         self.y2_local_ground_truth = y2_local_ground_truth
#         self.z_local_ground_truth = z_local_ground_truth
#         self.modelA = modelA
#         self.modelB = modelB
#         #obtain_local_loss(param, modelA, modelB, X_local, y1_local, y2_local_ground_truth, z_local_ground_truth):
#
#         self.maximize = maximize
#         if not torch.is_tensor(beta):
#             beta = torch.tensor(beta)
#         self.register_buffer("beta", beta)
#
#     @t_batch_mode_transform(expected_q=1)
#     def forward(self, X: Tensor) -> Tensor:
#         r"""Evaluate the Upper Confidence Bound on the candidate set X.
#
#         Args:
#             X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
#
#         Returns:
#             A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
#             given design points `X`.
#         """
#         self.beta = self.beta.to(X)
#         posterior = self.model.posterior(
#             X=X, posterior_transform=self.posterior_transform
#         )
#         mean = posterior.mean
#         view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
#         mean = mean.view(view_shape)
#         variance = posterior.variance.view(view_shape)
#         delta = (self.beta.expand_as(mean) * variance).sqrt()
#         local_regularizer = []
#         for param in X:
#           modelALoss, modelBLoss, system_pertubed_loss, system_loss = obtain_local_loss(param.detach().numpy(), self.modelA, self.modelB, self.X_local, self.y1_local, self.y2_local_ground_truth, self.z_local_ground_truth)
#           local_regularizer.append(modelALoss + modelBLoss)
#         local_regularizer = Tensor(local_regularizer)
#         if self.maximize:
#             return mean + delta - local_regularizer
#         else:
#             return -mean + delta - local_regularizer

class ScalarizedUpperConfidenceBound(AnalyticAcquisitionFunction):
    '''
    In Multi-task GP, there are multiple task outputs. As a result,
    we have to transform multi-output into a single number for
    sequential bayesian optimization (evaluating the next best candidate point).
    In this acquisition function
    '''
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        weights: Tensor,
        maximize: bool = True,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("weights", torch.as_tensor(weights))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the Upper Confidence Bound on the candidate set X using scalarization

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
                design points `X`.
        """
        #return 1;
        self.beta = self.beta.to(X)
        batch_shape = X.shape[:-2]
        posterior = self.model.posterior(X)
        means = posterior.mean.squeeze(dim=-2)  # b x o
        scalarized_mean = means.matmul(self.weights)  # b
        covs = posterior.mvn.covariance_matrix  # b x o x o
        weights = self.weights.view(
            1, -1, 1
        )  # 1 x o x 1 (assume single batch dimension)
        weights = weights.expand(batch_shape + weights.shape[1:])  # b x o x 1
        weights_transpose = weights.permute(0, 2, 1)  # b x 1 x o
        scalarized_variance = torch.bmm(
            weights_transpose, torch.bmm(covs, weights)
        ).view(
            batch_shape
        )  # b
        delta = (self.beta.expand_as(scalarized_mean) * scalarized_variance).sqrt()
        if self.maximize:
            return scalarized_mean + delta
        else:
            return scalarized_mean - delta