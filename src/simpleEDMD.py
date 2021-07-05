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

        #G = np.zeros((25, 25))
        #A = np.zeros((25, 25))

        #for m in range(len(self.X)):
            #psi_xm = self.dictionary_Hermite_poly(self.X[m])
            #psi_ym = self.dictionary_Hermite_poly(self.Y[m])

            #G += psi_xm.T @ psi_xm
            #A += psi_xm.T @ psi_ym
        
        Psi_X = self.dictionary_strategy.dictionary(self.X)
        Psi_Y = self.dictionary_strategy.dictionary(self.Y)
        
        G = Psi_X.T @ Psi_X /len(self.X)
        A = Psi_X.T @ Psi_Y /len(self.X)
            
    
        # compute koopman matrix
        self.koopman_matrix = np.linalg.pinv(G) @ A

        # compute koopman eigenvalues
        self.koopman_eigenvalues, self.levecs, self.right_eigenvectors = self.sort_eig(self.koopman_matrix)
        self.left_eigenvectors = np.linalg.pinv(self.right_eigenvectors).T # scaled w_star
        
        # compute koopman eigenfunctions
        # see method compute_koopman_eigenfunctions
        
        # compute B
        self.B = np.linalg.pinv(Psi_X) @ self.X

        # compute koopman modes
        self.koopman_modes = (self.left_eigenvectors.T @ self.B).T
    
    def compute_koopman_eigenfunctions(self, test_X):
        
        Psi_test_X = self.dictionary_strategy.dictionary(test_X)
        return Psi_test_X@self.right_eigenvectors
        

    def predict_next_timestep(self, X_initial):
        
        Phi = self.compute_koopman_eigenfunctions(X_initial) # 100x25
        Mu  = np.diag(self.koopman_eigenvalues) # 25x25
        V = self.koopman_modes #2x25
        
        
        X_next = Phi @ Mu @ V.T #100x2
        
        return X_next
    
    def predict_n_timestep(self, X_initial, n):
            
        Phi = self.compute_koopman_eigenfunctions(X_initial) # 100x25
        Mu  = np.diag(self.koopman_eigenvalues) # 25x25
        V = self.koopman_modes #2x25
        
        
        X_n = Phi @ np.linalg.matrix_power(Mu,n) @ V.T #100x2
        
        return X_n
    

    '''
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
    '''

    def sort_eig(self, matrix):

        eig_vals, eig_lvecs, eig_rvecs = scipy.linalg.eig(matrix, left=True, right=True)
        ind = eig_vals.argsort()[::-1]
        return (eig_vals[ind], eig_lvecs[:, ind].T, eig_rvecs[:, ind])
