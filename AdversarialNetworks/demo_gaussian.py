import logging

import numpy as np
from sklearn import linear_model, preprocessing, pipeline
from pylab import plt

from competition import AdversarialCompetition
from models import GenerativeNormalModel
from gradient_descent import GradientDescent

logging.basicConfig(format="[%(filename)s:%(lineno)s - %(funcName)15s() %(asctime)-15s ] %(message)s", level=logging.DEBUG)

np.random.seed(0)
size_batch = 1000

competition = AdversarialCompetition(
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

print(competition)

for i in range(200):
    if i % 50 == 0:
        competition.plot()
        plt.show()
        pass
    competition.iteration()

print("final model %s" % competition.generatives[-1])

competition.plot_params()
plt.show()

competition.plot_auc()
plt.show()
