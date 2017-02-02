import logging

import pandas as pd
import numpy as np
from sklearn import metrics
import sklearn.utils
import matplotlib.pyplot as plt

__version__ = "0.1"
logger = logging.getLogger(__name__)


class AdversarialCompetition(object):
    def __init__(self, true_model, discriminative, generative,
                 gradient_descent, size_batch=100):
        self.l_auc = []
        self.true_model = true_model
        self.generatives = [generative, ]
        self.discriminative = discriminative
        self.size_batch = size_batch
        self.gradient_descent = gradient_descent
        l_auc = []

    def plot(self, n=-1):
        xplot = np.arange(-10, 10, 0.1).reshape((-1, 1))
        generative = self.generatives[n]
        plt.figure()
        plt.plot(xplot, self.true_model.predict_proba(xplot), c="black")
        try:
            plt.plot(xplot, self.discriminative.predict_proba(xplot)[:, 0])
        except sklearn.utils.validation.NotFittedError:
            logger.info("Discriminative model have never been fitted")
            pass
        plt.plot(xplot, generative.predict_proba(xplot), c="red")
        plt.title("{model}\nIteration {n_iter} AUC {auc}".format(
            model=generative, n_iter=len(self.generatives) - 1, auc=self.l_auc[-1] if self.l_auc else "?"))

    def plot_params(self):
        lp = np.array([m._params for m in self.generatives])
        lp2 = lp.reshape([lp.shape[0], -1])
        plt.plot(lp2)
        comps = range(lp.shape[2]) if len(lp.shape) > 2 else [""]
        plt.legend(["%s %s" % (self.generatives[-1]._variables[var], comp)
                    for var in range(lp.shape[1])
                    for comp in comps])

    def plot_auc(self):
        plt.plot(self.l_auc)

    def iteration(self):
        np.random.seed(len(self.generatives))
        generative = self.generatives[-1]

        x_true = self.true_model.random(self.size_batch).reshape(-1, 1)
        x_generative = generative.random(self.size_batch).reshape(-1, 1)
        x = np.concatenate((x_true, x_generative))
        y = np.concatenate((np.ones((self.size_batch,)),
                            np.zeros((self.size_batch,))))
        logger.debug("Using x: %s to y: %s", x.shape, y.shape)
        self.discriminative.fit(x, y)

        x_true = self.true_model.random(self.size_batch).reshape(-1, 1)
        x_generative = generative.random(self.size_batch).reshape(-1, 1)
        yhat_true= self.discriminative.predict(x_true)
        yhat_generative = self.discriminative.predict(x_generative)
        yhat = np.concatenate((yhat_true, yhat_generative))

        logger.debug("Computing AUC %s %s", y.shape, yhat.shape)
        self.l_auc.append(metrics.roc_auc_score(y,yhat))
        logger.debug("AUC %s", self.l_auc[-1])

        logger.debug("Computing gradient")
        d_grad = generative.d_log_likelihood(x_generative)

        logger.debug("Computing update")
        update = self.gradient_descent.transform(d_grad * (2 * yhat_generative - 1))

        new_generative = generative.updated(update)
        self.generatives.append(new_generative)

    def __str__(self):
        return ("Adversarial Models Competition\n"
                "    True model {true_model}\n"
                "    Generative model {generative}\n"
                "    Discriminative {discriminative}\n"
                "    Gradient descent {gradient_descent}\n"
                "    Batch size {size_batch}").format(
            true_model=self.true_model,
            generative=self.generatives[-1],
            discriminative=self.discriminative,
            gradient_descent=self.gradient_descent,
            size_batch=self.size_batch)
