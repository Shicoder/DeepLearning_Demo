import copy
import logging
import itertools

import numpy as np
from utils import randn

__version__ = "0.1"
logger = logging.getLogger(__name__)


def to_1D(x):
    x = np.array(x)
    x = x.flatten() if x.shape else np.array([x])
    return x


class AbstractGenerativeModel(object):
    def __init__(self, variables, params, updates=None):
        self._updates = updates if updates is not None else variables
        self._variables = variables
        self._params = params

    @property
    def params(self):
        return dict(zip(self._variables, self._params))

    def d_log_likelihood_approx(self, x, eps=1e-4):
        gradients = []
        likelihood = self.log_likelihood(x)
        _params = self._params.copy()
        for coo in np.arange(len(_params.flat)):
            self._params.flat[coo] += eps
            likelihood_plus = self.log_likelihood(x)
            self._params.flat[coo] = _params.flat[coo]
            gradients.append((likelihood_plus - likelihood) / eps)

        return np.array(gradients).reshape(list(_params.shape) + [-1])

    def __str__(self):
        old_formatter = np.get_printoptions().get("formatter", {})
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        str_params = "".join(" " + a+":"+str(np.round(b, 3)) for a, b in self.params.items())
        np.set_printoptions(formatter=old_formatter)
        return "<{} {}>".format(self.__class__.__name__, str_params)

    def __repr__(self):
        return str(self)

    def updated(self, delta):
        logging.debug("n_vars %s n_delta %s", len(self._variables), len(delta))
        assert len(delta) == len(self._variables)
        out = copy.deepcopy(self)

        for i in range(len(delta)):
            if self._variables[i] in self._updates:
                out._params[i] += delta[i]

        return out

    def predict_proba(self, x):
        raise NotImplementedError()

    def d_log_likelihood(self, x):
        raise NotImplementedError()

    def log_likelihood(self, x):
        raise NotImplementedError()


class GenerativeNormalModel(AbstractGenerativeModel):
    def __init__(self, mu, sigma, updates=None):
        if not updates:
            updates = ["mu", "sigma"]
        assert sigma > 0
        super(GenerativeNormalModel, self).__init__(["mu", "sigma"], np.array([float(mu), float(sigma)]), updates)

    def random(self, n=1):
        mu, sigma = self.params["mu"], self.params["sigma"]
        return mu + randn(n) * sigma

    def log_likelihood(self, x):
        mu, sigma = self.params["mu"], self.params["sigma"]
        return - ((x - mu) / sigma) ** 2.0 - np.log(sigma)

    def predict_proba(self, x):
        likelihoods = np.array(map(self.log_likelihood, x))
        return np.exp(likelihoods) / np.sqrt(np.pi)

    def d_log_likelihood(self, x):
        mu, sigma = self.params["mu"], self.params["sigma"]
        x_normed = (x - mu) / sigma
        x_normed_2 = ((x - mu) / sigma) ** 2.0

        d_mu = 2 * x_normed / sigma
        d_sigma = (2.0 * x_normed_2 - 1.0) / sigma
        return np.array([d_mu, d_sigma]).reshape((2, -1))


class GenerativeNormalMixtureModel(AbstractGenerativeModel):
    def __init__(self, mu, sigma, probas=None, updates=None):
        logger.debug("GenerativeNormalMixtureModel mu=%s sigma=%s", mu, sigma)
        if not updates:
            updates = ["mu", "sigma", "probas"]
        mu = np.array(mu, dtype=np.float32)
        sigma = np.array(sigma, dtype=np.float32)
        probas = np.array(probas, dtype=np.float32) if probas is not None \
            else np.ones(len(mu)) / len(mu)
        assert(sigma > 0).all()
        assert sigma.shape == mu.shape
        super(GenerativeNormalMixtureModel, self).__init__(
            ["mu", "sigma", "probas"],
            np.array([np.float32(mu), np.float32(sigma), np.float32(probas)]),
            updates)

    def random(self, n=1):
        mu, sigma = self.params["mu"], self.params["sigma"]
        n_gaussian = np.random.randint(len(mu), size=n)
        mu0, sigma0 = mu[n_gaussian], sigma[n_gaussian]
        return mu0 + randn(n) * sigma0

    def log_likelihood(self, x):
        x = np.array(x).reshape((-1, 1))
        mu, sigma, probas = self.params["mu"], self.params["sigma"], self.params["probas"]
        mu, sigma = mu.reshape(1, -1), sigma.reshape(1, -1)

        # matrix observation X gaussians
        log_likelihoods = - ((x - mu) / sigma) ** 2.0 - np.log(sigma)

        # We normalize because of numerical instability
        # exp(a)+exp(b) = exp(a) * (1 + exp(b-1)) better if a > b
        max_log_likelihoods = log_likelihoods.max(axis=1)
        normalized_log_likelihoods = log_likelihoods - max_log_likelihoods.reshape((-1, 1))
        normalized_likelihood = np.exp(normalized_log_likelihoods) * probas

        return max_log_likelihoods + np.log(normalized_likelihood.sum(axis=-1))

    def predict_proba(self, x):
        return np.exp(self.log_likelihood(x)) / np.sqrt(np.pi)

    def d_log_likelihood(self, x):
        x = to_1D(x)
        n_obs, n_components = len(x), len(self.params["mu"])

        # n_obs X 1
        x = x.reshape((n_obs, 1))

        # 1 x n_components
        mu = self.params["mu"].reshape(1, n_components)
        sigma = self.params["sigma"].reshape(1, n_components)
        probas = self.params["probas"].reshape(1, n_components)

        x_normed = (x - mu) / sigma
        assert x_normed.shape == (n_obs, n_components)

        x_normed_2_min = (x_normed ** 2).min(axis=-1).reshape((n_obs, 1))

        exp_x_normed_2_normalized = np.exp(- x_normed ** 2 + x_normed_2_min)
        assert exp_x_normed_2_normalized.shape == (n_obs, n_components)

        p_exp = (probas * exp_x_normed_2_normalized)
        assert p_exp.shape == (n_obs, n_components)

        sum_p_exp = p_exp.sum(axis=1).reshape((-1, 1))

        # n_obs x n_components
        d_mu = 2.0 * x_normed / sigma * p_exp / sum_p_exp
        assert d_mu.shape == (n_obs, n_components)

        d_sigma = (2.0 * x_normed ** 2 - 1.0) / sigma * p_exp / sum_p_exp
        assert d_sigma.shape == (n_obs, n_components)

        d_p = exp_x_normed_2_normalized / sum_p_exp
        assert d_p.shape == (n_obs, n_components)

        return np.array([d_mu, d_sigma, d_p]).transpose([0, 2, 1])
