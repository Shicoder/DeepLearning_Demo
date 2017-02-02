import logging

import numpy as np

logger = logging.getLogger(__name__)


class GradientDescent(object):
    def __init__(self, learning_rate, inertia=0.0, annealing=np.Inf, last_learning_rate=None):
        self.last_learning_rate = last_learning_rate
        self.learning_rate = learning_rate
        self.inertia = inertia
        self.annealing = annealing
        self.last_update = None
        self.n_iter = 0

    def transform(self, grad):
        """Individuals must be on the LAST dimensions"""
        p_anneal = self.n_iter / self.annealing
        if self.last_learning_rate is None:
            learning_rate = self.learning_rate / (1.0 + p_anneal)
        else:
            g = (self.last_learning_rate / self.annealing) ** min(p_anneal, 1.0)
            learning_rate = self.learning_rate * g

        self.n_iter += 1.0
        raw_update = learning_rate * grad.mean(axis=-1)
        if self.last_update is None:
            self.last_update = raw_update
        self.last_update = (self.inertia * self.last_update +
                            (1 - self.inertia) * raw_update)
        return self.last_update

    def __str__(self):
        return ("<GradientDescent"
                " learning_rate={learning_rate}"
                " last_learning_rate={last_learning_rate}"
                " inertia={inertia}"
                " annealing={annealing}").format(**self.__dict__)
