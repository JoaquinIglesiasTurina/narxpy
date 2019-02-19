import matplotlib
matplotlib.use('Agg')
import pymc3 as pm
import numpy as np
import theano.tensor as tt
from theano import shared
import matplotlib.pyplot as plt
import scipy.stats as stats
import functools as ft
from sklearn.preprocessing import StandardScaler

from data import dataGenerator
from dataPreprocess import lagData, RadialBasis
from model import Narx


def plotSeries(dataSet, name, number):
    plt.plot(dataSet[:, number])
    plt.title('plot of ' + name)
    plt.savefig(name + 'plot.png')
    plt.close()

# parametros, yo tampoco se que significan
sigma = 16.0
r = 45.92
b = 4.0
# este si se lo que significa, h es la magnitud del pasocon el que se generan
# los datos
h = 0.01
# este es la fila 0 de los datos
init = np.array([-1, 0, 1])
# cuantas obvservaciones
numObs = 2000

numBasis = 3
numLags = 2
# generamos los datos
generator = dataGenerator(sigma, r, b)
data = generator(init, numObs, h) 

# [plotSeries(data, *x)
#  for x in zip(['x', 'y', 'z'], [0, 1, 2])]

narx = Narx(numLags, numBasis)
fittedNarx = narx.fit(data, 20000)

newData = generator(data[numObs - 1], 3, h)[1:]

newData = newData.reshape(2, 1, 3)

prediction = narx.predict(newData, 2000, 2000)
prediction
prediction

np.roll(data[:10], 0, 0)

predOutput = np.array([np.vstack([data[data.shape[0] - numLags:], newData, prediction[k]])
                       for k in range(prediction.shape[0])])
predOutput
laggedOutput = np.array([lagData(predOutput[k], numLags)
                         for k in range(prediction.shape[0])])

laggedOutput

fullOutput = np.array([
    np.concatenate((newData, prediction[k]), 0)
    for k in range(prediction.shape[0])
])

asdf['yVec'].apply(lambda x: newData.concat)


# laggedData = lagData(data, numLags)
# newBasis = changeBasis(laggedData, numBasis)
# # [plotSeries(newBasis, *x)
# #  for x in zip(['basis0', 'basis1', 'basis2'], [0, 1, 2])]
# scaler = StandardScaler()
# newBasis = scaler.fit_transform(newBasis)
# data = scaler.fit_transform(data)
# # preprocessingDone
# print('new basis computed:\n',
#       newBasis[0:10, 0:4])
# print('''The sin which is unpardonable is knowingly and willfully to reject truth,
# to fear knowledge lest that knowledge pander not to thy prejudices.''')

# sharedPredictors = shared(newBasis)

# with pm.Model() as model:
#     theta = pm.Normal('theta', 0, 1,
#                       shape = (numBasis, data.shape[1]))

#     fX = pm.math.matrix_dot(sharedPredictors, theta)
#     pm.Deterministic('fX', fX)
    
#     yVec = pm.MvNormal('yVec', fX, 
#                        tau = np.eye(data.shape[1]),
#                        observed=data[numLags:, :])

# with model:
#     advi = pm.ADVI()
#     approx = pm.fit(n = 20000, method = advi)

# data[numObs - 1]


# newData = generator(data[numObs - 1], numObs, h)
# newLaggedData = lagData(newData, numLags)
# newNewBasis = changeBasis(newLaggedData, numBasis)
# newNewBasis = scaler.fit_transform(newNewBasis)
# sharedPredictors.set_value(newNewBasis)

# ppc = pm.sample_posterior_predictive(approx.sample(2000), model=model, samples = 10)
# ppc['yVec'].shape




# plt.plot(advi.hist)
# plt.savefig('trace.png')
# plt.close()

# # generating fitting plots
# fig = plt.figure()
# xPlot = fig.add_subplot(311)
# xPlot.plot(data[numLags:, 0])


# samples = approx.sample(50)['fX']
# xSamples = [sample[:, 0] for sample in samples]


# [plt.plot(sample, 'ro', alpha = 0.009) 
#  for sample in xSamples]
# plt.plot(data[numLags:, 0])
# plt.savefig('samples.png')
# plt.close()


# [plotSeries(asdf['fX'][0], *x)
#  for x in zip(['xHat', 'yHat', 'zHat'], range(0, 3))]


