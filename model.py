import pymc3 as pm
import numpy as np
import theano.tensor as tt
import scipy.stats as stats
import functools as ft
from theano import shared
from dataPreprocess import lagData, RadialBasis
from sklearn.preprocessing import StandardScaler


class Narx(object):
    def __init__(self, numLags, numBasis):
        self.numLags = numLags
        self.numBasis = max(numBasis, 3)
        self.model = pm.Model()
        self.scaler = StandardScaler()
        self.yScaler = StandardScaler(with_std=False)
        self.fitted = False

    def fit(self, data, adviIterations):
        self.data = data
        self.yScaler.fit(data)
        laggedData = lagData(data, self.numLags)
        # changing basis
        # set basis
        self.radialBasis = RadialBasis(self.numBasis)
        self.radialBasis.fit(laggedData)
        # changeBasis
        changedBasis = self.radialBasis.transform(laggedData)
        # scaling for numeric funzies
        self.scaler.fit(changedBasis)
        changedBasis = self.scaler.transform(changedBasis)
        # set model predictors as shared so we can do the forecasting
        self.sharedPredictors = shared(changedBasis)
        # pymc model
        with self.model:
            theta = pm.Normal('theta', 0, 1,
                              shape = (self.numBasis, data.shape[1]))

            fX = pm.math.matrix_dot(self.sharedPredictors, theta)
            pm.Deterministic('fX', fX)
    
            yVec = pm.MvNormal('yVec', fX, 
                               tau = np.eye(data.shape[1]),
                               observed=self.yScaler.transform(
                                   data[self.numLags:, :]))

            advi = pm.ADVI()
            self.approx = pm.fit(n = adviIterations, method = advi)
        
        print('variational inference concluded')

        print(
            '''
            The sin which is unpardonable is knowingly and willfully to reject truth,
            to fear knowledge lest that knowledge pander not to thy prejudices.
            ''')

        self.fitted = True

    def predict(self, newData, traceSamples, samples):
        def changeBasisAndScale(data):
            return self.scaler.transform(
                self.radialBasis.transform(data))

        newDataShape = len(newData.shape)

        if newDataShape == 2:
            print("reshaping data for prediction")
            newData = np.array([newData])
        elif newDataShape != 3:
            raise "new data has wrong dimension"

        numSamples, numObs, numVars = newData.shape

        newData = np.array(
            [np.vstack(
                [self.data[self.data.shape[0] - self.numLags:],
                 newData[k]])
             for k in range(numSamples)])

        print("lagging data for prediction")
        newLaggedData = np.array(
            [lagData(newData[k], self.numLags, True)
             for k in range(numSamples)])

        newNewBasis = np.array(
            [changeBasisAndScale(newLaggedData[k])
             for k in range(numSamples)])

        newNewBasis = newNewBasis.reshape(numSamples, newNewBasis.shape[2])
        
        self.sharedPredictors.set_value(np.array([newNewBasis[newNewBasis.shape[0] - 1]]))

        ppc = pm.sample_posterior_predictive(
            self.approx.sample(traceSamples), 
            model=self.model, 
            samples = samples)['yVec']
        
        return np.array([
            self.yScaler.inverse_transform(ppc[k])
            for k in range(samples)
        ])
