import numpy as np
from scipy import special
from scipy.spatial import distance
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


class Hermite_strategy(Strategy):

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
    
    def ini(self, X):
        pass
    
    
class dmap_strategy(Strategy):
    
    def __init__(self, n, e):
        
        # number of dictionary functions
        self.n = n
        # bandwidth = e * average distance
        self.e = e
    
    def g_kernel(self, data, x_center, eps):
        
        # Gaussian kernel
        return np.exp(-(distance.cdist(data,x_center) / self.eps)**2)
    
    def ini(self, X):
        
        # dmap
        self.X_pcm = pfold.PCManifold(X)
        self.X_pcm.optimize_parameters()
        dmap = dfold.DiffusionMaps(
            kernel=pfold.GaussianKernel(epsilon=self.X_pcm.kernel.epsilon),
            n_eigenpairs=self.n,
            dist_kwargs=dict(cut_off=self.X_pcm.cut_off),
        )
        dmap = dmap.fit(self.X_pcm)
        # n dmap eigenvectors in columns
        v = dmap.eigenvectors_[:,:]
        
        # rbf
        self.rand_id = np.random.permutation(self.X_pcm.shape[0])[0:self.n]
        self.eps = np.average(distance.cdist(self.X_pcm,self.X_pcm)) * self.e
        
        phi = self.g_kernel(self.X_pcm,self.X_pcm[self.rand_id,:],self.eps)
        self.c,_,_,_ = np.linalg.lstsq(phi, v, rcond = None)
              
    def dictionary(self, X):
        
        phi = self.g_kernel(X,self.X_pcm[self.rand_id,:],self.eps)
    
        return phi@self.c
