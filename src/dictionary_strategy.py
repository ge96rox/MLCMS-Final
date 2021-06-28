import numpy as np
from scipy import special


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
