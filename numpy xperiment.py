import numpy as np
import math
import pandas as pd


def sample(paramArray,nSamples):
    l = math.floor(len(paramArray)/nSamples)
    outArray = np.zeros(l+1)
    print(np.shape(paramArray),np.shape(outArray))
    for i in range(l+1):
        outArray[i] = paramArray[i * nSamples]
    print(paramArray,outArray)
    return outArray
a =np.arange(1,10)
print(a,sample(a,2))