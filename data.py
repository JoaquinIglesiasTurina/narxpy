import numpy as np


class dataGenerator(object):
    def __init__(self, sigma, r, b):
        self.sigma = sigma
        self.r = r
        self.b = b

    def __call__(self, init, nObs, h):
        def xDot(x, y, z):
            return self.sigma * (y - x)
        
        def yDot(x, y, z):
            return (x * (self.r - z))  - y
        
        def zDot(x, y, z):
            return (x * y)  - (self.b * z)
        
        def newEntry(x, y, z):
            return np.array([
                f(x, y, z)
                for f in [xDot, yDot, zDot]
            ])

        output = np.zeros((nObs, 3))
        output[0] = init

        counter = 1
        while (counter <= (nObs - 1)):
            prevPeriod = output[counter - 1]
            output[counter] = (prevPeriod + (h * newEntry(*prevPeriod)))
            counter += 1

        return output
