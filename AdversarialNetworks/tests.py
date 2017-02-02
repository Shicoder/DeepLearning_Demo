import unittest
import logging

import numpy as np
from sklearn import pipeline, preprocessing, linear_model
from competition import AdversarialCompetition
from gradient_descent import GradientDescent
from models import GenerativeNormalModel, GenerativeNormalMixtureModel
from utils import randn

logging.basicConfig(level=logging.INFO)
__version__ = "0.1"
logger = logging.getLogger(__name__)


class TestDistribution(unittest.TestCase):
    def test_normal(self):
        return self._test(GenerativeNormalModel(1, 2))

    def test_normal_mixture(self):
        return self._test(GenerativeNormalMixtureModel([0, 1], [0.5, 2]))

    def _test(self, model):
        step = 0.01
        x = np.arange(-100, 100, step)
        y = model.predict_proba(x)
        integral = np.sum(y) * step
        self.assertLess(np.abs(integral - 1), 0.01)


class TestGradient(unittest.TestCase):
    def test_normal(self):
        return self._test(GenerativeNormalModel(1, 2))

    def test_normal_mixture(self):
        return self._test(GenerativeNormalMixtureModel([-3, 3], [1, 1]))

    def _test(self, model):
        np.random.seed(0)
        logger.info("Starting test gradient for model %s" % model)
        np.random.seed(1)
        x = np.round(randn(1), 1)
        grad = model.d_log_likelihood(x)
        grad_approx = model.d_log_likelihood_approx(x)
        np.testing.assert_allclose(grad, grad_approx, 0.01)


class TestConvergence(unittest.TestCase):
    def test_normal_10(self):
        np.random.seed(0)
        size_batch = 10
        adversarials = AdversarialCompetition(
            size_batch=size_batch,
            true_model=GenerativeNormalModel(1, 2),
            discriminative=pipeline.make_pipeline(
                preprocessing.PolynomialFeatures(4),
                linear_model.LogisticRegression()),
            generative=GenerativeNormalModel(0, 1, updates=["mu", "sigma"]),
            gradient_descent=GradientDescent(
                0.1, inertia=0.9, annealing=1000, last_learning_rate=0.01),
        )
        for i in range(500):
            adversarials.iteration()
        params = adversarials.generatives[-1]._params
        true_params = adversarials.true_model._params
        np.testing.assert_allclose(params, true_params, 0, 0.05)

    def test_normal_100(self):
        np.random.seed(0)
        size_batch = 100
        adversarials = AdversarialCompetition(
            size_batch=size_batch,
            true_model=GenerativeNormalModel(1, 2),
            discriminative=pipeline.make_pipeline(
                preprocessing.PolynomialFeatures(4),
                linear_model.LogisticRegression()),
            generative=GenerativeNormalModel(
                0, 1, updates=["mu", "sigma"]),
            gradient_descent=GradientDescent(
                0.03, inertia=0.0, annealing=100),
        )
        for i in range(1000):
            adversarials.iteration()
        params = adversarials.generatives[-1]._params
        true_params = adversarials.true_model._params
        np.testing.assert_allclose(params, true_params, 0, 0.02)

    def test_normal_1000(self):
        np.random.seed(0)
        size_batch = 1000
        adversarials = AdversarialCompetition(
            size_batch=size_batch,
            true_model=GenerativeNormalModel(1, 2),
            discriminative=pipeline.make_pipeline(
                preprocessing.PolynomialFeatures(4),
                linear_model.LogisticRegression()),
            generative=GenerativeNormalModel(0, 1, updates=["mu", "sigma"]),
            gradient_descent=GradientDescent(0.03, 0.9),
        )
        for i in range(200):
            adversarials.iteration()
        params = adversarials.generatives[-1]._params
        true_params = adversarials.true_model._params
        np.testing.assert_allclose(params, true_params, 0, 0.02)

    def test_normal_mixture(self):
        np.random.seed(0)
        size_batch = 1000
        competition = AdversarialCompetition(
            size_batch=size_batch,
            true_model=GenerativeNormalMixtureModel([-3, 3], [1, 1]),
            discriminative=pipeline.make_pipeline(
                preprocessing.PolynomialFeatures(4),
                linear_model.LogisticRegression()),
            generative=GenerativeNormalMixtureModel(
                [-1, 1], [1, 1], updates=["mu", "sigma"]),
            gradient_descent=GradientDescent(
                0.1, inertia=0.9, annealing=1000, last_learning_rate=0.01),
        )
        for i in range(2000):
            competition.iteration()
        params = competition.generatives[-1]._params
        true_params = competition.true_model._params
        np.testing.assert_allclose(params, true_params, 0, 0.1)

    def test_normal_mixture_hard(self):
        np.random.seed(0)
        size_batch = 1000
        competition = AdversarialCompetition(
            size_batch=size_batch,
            true_model=GenerativeNormalMixtureModel(
                np.arange(-3, 4), np.random.uniform(1, 2, 7).round(2)),
            discriminative=pipeline.make_pipeline(
                preprocessing.PolynomialFeatures(4),
                linear_model.LogisticRegression()),
            generative=GenerativeNormalMixtureModel(
                np.arange(-3, 4) * 0.1, np.ones(7), updates=["mu", "sigma"]),
            gradient_descent=GradientDescent(
                np.array([0.3, 0.1, 0.3]).reshape((-1, 1)), inertia=0.9,
                annealing=2000, last_learning_rate=0.001),
        )
        for i in range(5000):
            competition.iteration()
        params = competition.generatives[-1]._params
        print params.shape
        true_params = competition.true_model._params
        np.testing.assert_allclose(params, true_params, 0, 0.2)


if __name__ == '__main__':
    unittest.main()
