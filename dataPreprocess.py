import pymc3 as pm
import numpy as np
import theano.tensor as tt
import scipy.stats as stats
import functools as ft


def lagData(data, lags, includeCurrent=False):
    if includeCurrent:
        ranger = (0, lags)
    else:
        ranger = (1, lags + 1)

    lagsList = [np.roll(data, lag, 0)
            for lag in range(*ranger)]

    return ft.reduce(
        lambda x, y: np.concatenate([x, y], 1),
        lagsList
    )[lags:, :]

class RadialBasis(object):
    def __init__(self, numBasis):
        self.numBasis = numBasis

    def fit(self, data):
        self.nObs, self.nVars = data.shape

        self.mvNormal = stats.multivariate_normal(
            np.zeros(self.nVars), 
            np.eye(self.nVars))

        self.kernelMatrix = np.quantile(
            data, 
            np.arange(0, 1.1, 1/(self.numBasis - 1)), 
            axis = 0)

    def transform(self, data):
        def helper(array):
            return np.array([
                self.mvNormal.logpdf(array - self.kernelMatrix[x, :])
                for x in range(self.numBasis)
            ])

        return np.array([
            helper(data[x, :]) for x in range(data.shape[0])
        ])

    def fitTransform(self, data):
        return self.fit(data).transform(data)


def changeBasis(data, numBasis):
    def helper(array):
        return np.array([
            mvNormal.logpdf(array - kernelMatrix[x, :])
            for x in range(numBasis)
        ])
    
    nObs, nVars = data.shape

    mvNormal = stats.multivariate_normal(
        np.zeros(nVars), 
        np.eye(nVars))

    kernelMatrix = np.quantile(
        data, 
        np.arange(0, 1.1, 1/(numBasis - 1)), 
        axis = 0)

    return np.array([
        helper(data[x, :]) for x in range(data.shape[0])
    ])
