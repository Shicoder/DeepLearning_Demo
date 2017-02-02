import logging

import numpy as np
from sklearn import linear_model, preprocessing, pipeline
from pylab import plt

from competition import AdversarialCompetition
from models import GenerativeNormalMixtureModel
from gradient_descent import GradientDescent

logging.basicConfig(format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s", level="INFO")

np.random.seed(0)
size_batch = 1000

competition = AdversarialCompetition(
    size_batch=size_batch,
    true_model=GenerativeNormalMixtureModel([-3, 3], [1, 1]),
    discriminative=pipeline.make_pipeline(preprocessing.PolynomialFeatures(4),
                                          linear_model.LogisticRegression()),
    generative=GenerativeNormalMixtureModel([-1, 1], [1, 1], updates=["mu", "sigma"]),
    gradient_descent=GradientDescent(0.01, 0.5),
)

print(competition)

for i in range(1000):
    if i % 200 == 0:
        competition.plot()
        plt.show()
        pass
    competition.iteration()

print("final model %s" % competition.generatives[-1])

competition.plot_params()
plt.show()

competition.plot_auc()
plt.show()
