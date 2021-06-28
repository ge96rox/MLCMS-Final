import numpy as np
import pandas as pd
import scipy.linalg
from scipy import special


class simpleEDMD:

    def __init__(self, X, Y, dictionary_strategy):

        self.X = X
        self.Y = Y
        self.dictionary_strategy = dictionary_strategy
        # self.dictionary

    def compute_koopman_operator(self):
        # dimension and data format specificed to 4.1.1
        # needs GENERALIZATION

        G = np.zeros((25, 25))
        A = np.zeros((25, 25))

        for m in range(len(self.X)):
            psi_xm = self.dictionary_Hermite_poly(self.X[m])
            psi_ym = self.dictionary_Hermite_poly(self.Y[m])

            G += psi_xm.T @ psi_xm
            A += psi_xm.T @ psi_ym
        G /= len(self.X)
        A /= len(self.X)

        # compute koopman matrix
        self.koopman_matrix = np.linalg.pinv(G) @ A

        # compute koopman eigenvalues
        self.koopman_eigenvalues, self.left_eigenvectors, self.right_eigenvectors = self.sort_eig(self.koopman_matrix)
        # self.left_eigenvectors = np.linalg.pinv(self.right_eigenvectors)  # w_star.T

        # compute koopman eigenfunctions

        # compute koopman modes

    def dictionary_Hermite_poly(self, xm):

        # dictionary = []
        #
        # for j in range(0, 5):
        #     Hx2 = special.hermite(j, monic=True)
        #     for i in range(0, 5):
        #         Hx1 = special.hermite(i, monic=True)
        #         dictionary.append(Hx1(xm[0]) * Hx2(xm[1]))
        # return np.array([dictionary])

        return self.dictionary_strategy.dictionary(xm)

    def sort_eig(self, matrix):

        eig_vals, eig_lvecs, eig_rvecs = scipy.linalg.eig(matrix, left=True, right=True)
        ind = eig_vals.argsort()[::-1]
        return (eig_vals[ind], eig_lvecs[:, ind].T, eig_rvecs[:, ind])
