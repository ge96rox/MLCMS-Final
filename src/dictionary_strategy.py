import numpy as np
from scipy import special
import pandas as pd

import copy
import mpl_toolkits.mplot3d.axes3d as p3
import sklearn.manifold as manifold
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import LocalRegressionSelection



class Strategy(object):
    def __init__(self):
        pass


class Herimite_strategy(Strategy):

    def __init__(self):
        H0 = special.hermite(0, monic=True)
        H1 = special.hermite(1, monic=True)
        H2 = special.hermite(2, monic=True)
        H3 = special.hermite(3, monic=True)
        H4 = special.hermite(4, monic=True)
        self.D = lambda x, y: [H0(x) * H0(y), H1(x) * H0(y), H2(x) * H0(y), H3(x) * H0(y), H4(x) * H0(y),
                          H0(x) * H1(y), H1(x) * H1(y), H2(x) * H1(y), H3(x) * H1(y), H4(x) * H1(y),
                          H0(x) * H2(y), H1(x) * H2(y), H2(x) * H2(y), H3(x) * H2(y), H4(x) * H2(y),
                          H0(x) * H3(y), H1(x) * H3(y), H2(x) * H3(y), H3(x) * H3(y), H4(x) * H3(y),
                          H0(x) * H4(y), H1(x) * H4(y), H2(x) * H4(y), H3(x) * H4(y), H4(x) * H4(y)]

    def dictionary(self, X):
        dictionary = []
        for xm in X:
            dictionary.append( self.D(xm[0], xm[1]))

        return np.array(dictionary)
    
    
class dmap_strategy(Strategy):
    
    def _init_(self):
        pass
    
    def dictionary(self, X):
        X_pcm = pfold.PCManifold(X)
        X_pcm.optimize_parameters()
        dmap = dfold.DiffusionMaps(
            kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon),
            n_eigenpairs=5,
            dist_kwargs=dict(cut_off=X_pcm.cut_off),
        )
        dmap = dmap.fit(X_pcm)
    
        return dmap.eigenvectors_[:,:]
