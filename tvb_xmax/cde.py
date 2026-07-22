"""Small conditional density estimators used by tvb-xMax.

This module adapts the mixture-density-network technique from ``maedoc/tvbl``
for simulation-based posterior estimation.  It intentionally has no Torch or
``sbi`` dependency: training uses :mod:`autograd` and inference uses NumPy.
"""

from __future__ import annotations

import math

import autograd.numpy as anp
from autograd import grad
from autograd.scipy.special import logsumexp
import numpy as np


class MDNEstimator:
    """Conditional full-covariance Gaussian mixture density network."""

    def __init__(self, param_dim: int, feature_dim: int, n_components: int = 8,
                 hidden_sizes: tuple[int, ...] = (64, 64)):
        if param_dim < 1 or feature_dim < 1:
            raise ValueError("param_dim and feature_dim must be positive")
        self.param_dim = param_dim
        self.feature_dim = feature_dim
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes
        self.weights = None
        self.loss_history = []
        rows, cols = np.triu_indices(param_dim, k=1)
        self._offdiag_rows = rows
        self._offdiag_cols = cols

    def _initialize_weights(self, rng):
        weights = {}
        size = self.feature_dim
        for i, out_size in enumerate(self.hidden_sizes):
            weights[f"W{i}"] = (rng.randn(size, out_size) * math.sqrt(2 / size)).astype("f")
            weights[f"b{i}"] = anp.zeros(out_size, dtype="f")
            size = out_size
        k, d = self.n_components, self.param_dim
        for name, width in (("alpha", k), ("mu", k * d),
                            ("diag", k * d),
                            ("offdiag", k * d * (d - 1) // 2)):
            weights[f"W_{name}"] = (rng.randn(size, width) * 0.01).astype("f")
            weights[f"b_{name}"] = anp.zeros(width, dtype="f")
        return weights

    def _forward(self, weights, features):
        h = features
        for i in range(len(self.hidden_sizes)):
            h = anp.tanh(h @ weights[f"W{i}"] + weights[f"b{i}"])
        n, k, d = features.shape[0], self.n_components, self.param_dim
        logits = h @ weights["W_alpha"] + weights["b_alpha"]
        alpha = anp.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        mu = (h @ weights["W_mu"] + weights["b_mu"]).reshape(n, k, d)
        log_diag = (h @ weights["W_diag"] + weights["b_diag"]).reshape(n, k, d)
        lower = anp.eye(d)[None, None, :, :] * anp.exp(log_diag)[:, :, :, None]
        offdiag = (h @ weights["W_offdiag"] + weights["b_offdiag"]).reshape(
            n, k, d * (d - 1) // 2)
        # Construct the strictly upper triangular entries without in-place
        # mutation, which keeps the expression differentiable under autograd.
        for ix, (row, col) in enumerate(zip(self._offdiag_rows, self._offdiag_cols)):
            basis = anp.zeros((d, d))
            basis = basis + anp.eye(d)[row:row + 1].T @ anp.eye(d)[col:col + 1]
            lower = lower + offdiag[:, :, ix, None, None] * basis
        return alpha, mu, lower, log_diag

    def _loss(self, weights, features, params):
        alpha, mu, precision, log_diag = self._forward(weights, features)
        delta = params[:, None, :] - mu
        z = anp.einsum("nkij,nkj->nki", precision, delta)
        logp = (-0.5 * anp.sum(z ** 2, axis=2)
                + anp.sum(log_diag, axis=2)
                - 0.5 * self.param_dim * anp.log(2 * anp.pi))
        return -anp.mean(logsumexp(anp.log(alpha + 1e-9) + logp, axis=1))

    def train(self, params, features, n_iter: int = 500,
              learning_rate: float = 1e-3, seed: int = 0, prog: bool = False):
        """Fit the conditional mixture by full-batch Adam."""
        params = np.asarray(params, dtype="f")
        features = np.asarray(features, dtype="f")
        finite = np.all(np.isfinite(params), axis=1) & np.all(np.isfinite(features), axis=1)
        params, features = params[finite], features[finite]
        if not len(params):
            raise ValueError("no finite simulation pairs available for posterior training")
        if params.shape[1] != self.param_dim or features.shape[1] != self.feature_dim:
            raise ValueError("training dimensions do not match estimator dimensions")
        self.weights = self._initialize_weights(np.random.RandomState(seed))
        mean = {k: anp.zeros_like(v) for k, v in self.weights.items()}
        var = {k: anp.zeros_like(v) for k, v in self.weights.items()}
        gradient = grad(self._loss)
        for step in range(n_iter):
            grads = gradient(self.weights, features, params)
            for name in self.weights:
                mean[name] = 0.9 * mean[name] + 0.1 * grads[name]
                var[name] = 0.999 * var[name] + 0.001 * grads[name] ** 2
                mhat = mean[name] / (1 - 0.9 ** (step + 1))
                vhat = var[name] / (1 - 0.999 ** (step + 1))
                self.weights[name] -= learning_rate * mhat / (anp.sqrt(vhat) + 1e-8)
            if prog and (step + 1) % 100 == 0:
                print(f"posterior step {step + 1}/{n_iter}")
        self.loss_history.append(float(self._loss(self.weights, features, params)))
        return self

    def sample_batched(self, shape, x, show_progress_bars: bool = False):
        """Match the ``sbi`` batch-sampling shape: ``(n, B, d_param)``."""
        if self.weights is None:
            raise RuntimeError("posterior has not been trained")
        n_samples = int(shape[0])
        features = np.asarray(x, dtype="f")
        if features.ndim == 1:
            features = features[None, :]
        alpha, mu, precision, _ = self._forward(self.weights, features)
        alpha, mu, precision = map(np.asarray, (alpha, mu, precision))
        rng = np.random.default_rng(0)
        batch, _, d = mu.shape
        components = np.stack([rng.choice(self.n_components, n_samples, p=a)
                               for a in alpha])
        output = np.empty((n_samples, batch, d), dtype=np.float32)
        for b in range(batch):
            for s, component in enumerate(components[b]):
                # precision is triangular, so inv(precision) transforms a
                # standard normal into a covariance-factor sample.
                output[s, b] = mu[b, component] + np.linalg.solve(
                    precision[b, component], rng.normal(size=d))
        return output

    def sample(self, shape, x, show_progress_bars: bool = False):
        """Single-observation counterpart compatible with SBI-style callers."""
        samples = self.sample_batched(shape, x, show_progress_bars)
        return samples[:, 0, :] if samples.shape[1] == 1 else samples


class MAFEstimator:
    """Conditional masked autoregressive flow adapted from ``tvbl.cde``."""

    def __init__(self, param_dim: int, feature_dim: int, n_flows: int = 4,
                 hidden_units: int = 64):
        if param_dim < 1 or feature_dim < 1:
            raise ValueError("param_dim and feature_dim must be positive")
        self.param_dim, self.feature_dim = param_dim, feature_dim
        self.n_flows, self.hidden_units = n_flows, hidden_units
        self.weights = None
        self.layers = []

    def _initialize(self, rng):
        weights, self.layers = {}, []
        d, c, h = self.param_dim, self.feature_dim, self.hidden_units
        for index in range(self.n_flows):
            degrees = np.arange(1, d + 1)
            hidden = rng.randint(1, max(2, d), size=h)
            mask1 = (degrees[None] <= hidden[:, None]).astype("f")
            mask2 = (hidden[None] < degrees[:, None]).astype("f")
            perm = rng.permutation(d); inverse = np.empty(d, dtype=int); inverse[perm] = np.arange(d)
            self.layers.append((mask1, mask2, perm, inverse))
            weights.update({
                f"W1y_{index}": (rng.randn(h, d) * .01).astype("f"),
                f"W1c_{index}": (rng.randn(h, c) * .01).astype("f"),
                f"b1_{index}": anp.zeros(h),
                f"W2_{index}": anp.zeros((2 * d, h)),
                f"W2c_{index}": anp.zeros((2 * d, c)),
                f"b2_{index}": anp.zeros(2 * d),
            })
        return weights

    def _made(self, y, context, index, weights):
        mask1, mask2, _, _ = self.layers[index]
        hidden = anp.tanh(anp.dot(y, (weights[f"W1y_{index}"] * mask1).T)
                          + anp.dot(context, weights[f"W1c_{index}"].T) + weights[f"b1_{index}"])
        tiled = anp.concatenate([mask2, mask2])
        output = (anp.dot(hidden, (weights[f"W2_{index}"] * tiled).T)
                  + anp.dot(context, weights[f"W2c_{index}"].T) + weights[f"b2_{index}"])
        return output[:, :self.param_dim], anp.clip(output[:, self.param_dim:], -7, 7)

    def _loss(self, weights, features, params):
        u, log_det = params, anp.zeros(params.shape[0])
        for index, (_, _, perm, _) in enumerate(self.layers):
            u = u[:, perm]
            mean, log_scale = self._made(u, features, index, weights)
            u = (u - mean) * anp.exp(-log_scale)
            log_det = log_det - anp.sum(log_scale, axis=1)
        return -anp.mean(-.5 * anp.sum(u ** 2, axis=1) - .5 * self.param_dim * anp.log(2 * anp.pi) + log_det)

    def train(self, params, features, n_iter: int = 500, learning_rate: float = 1e-3,
              seed: int = 0, prog: bool = False):
        params, features = np.asarray(params, dtype="f"), np.asarray(features, dtype="f")
        self.weights = self._initialize(np.random.RandomState(seed))
        moment = {k: anp.zeros_like(v) for k, v in self.weights.items()}
        variance = {k: anp.zeros_like(v) for k, v in self.weights.items()}
        derivative = grad(self._loss)
        for step in range(n_iter):
            grads = derivative(self.weights, features, params)
            for name in self.weights:
                moment[name] = .9 * moment[name] + .1 * grads[name]
                variance[name] = .999 * variance[name] + .001 * grads[name] ** 2
                self.weights[name] -= learning_rate * (moment[name] / (1 - .9 ** (step + 1))) / (
                    anp.sqrt(variance[name] / (1 - .999 ** (step + 1))) + 1e-8)
        return self

    def sample_batched(self, shape, x, show_progress_bars: bool = False):
        if self.weights is None:
            raise RuntimeError("posterior has not been trained")
        n_samples, features = int(shape[0]), np.asarray(x, dtype="f")
        if features.ndim == 1:
            features = features[None]
        batch, d = len(features), self.param_dim
        context = np.repeat(features, n_samples, axis=0)
        value = np.random.default_rng(0).normal(size=(len(context), d)).astype("f")
        for index in reversed(range(self.n_flows)):
            _, _, _, inverse = self.layers[index]
            out = np.zeros_like(value)
            for coordinate in range(d):
                mean, log_scale = self._made(out, context, index, self.weights)
                out[:, coordinate] = value[:, coordinate] * np.exp(log_scale[:, coordinate]) + mean[:, coordinate]
            value = out[:, inverse]
        return value.reshape(batch, n_samples, d).transpose(1, 0, 2)

    def sample(self, shape, x, show_progress_bars: bool = False):
        samples = self.sample_batched(shape, x, show_progress_bars)
        return samples[:, 0] if samples.shape[1] == 1 else samples
