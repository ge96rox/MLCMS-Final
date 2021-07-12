import numpy as np
import pandas as pd
import scipy.linalg
from scipy import special


class simpleEDMD:
    """
       simpleEDMD(X, Y, dictionary_strategy)
       A class represents a simple EMDM algorithm

       Parameters
       ----------
       X : np.ndarray
           input data at time t in [M, N] format, where M is the number of input data,
           N is the dimension of data

       Y : np.ndarray
           input data at time t+1 in [M, N] format, where M is the number of input data,
           N is the dimension of data

       dictionary_strategy: class Strategy
            a class presents chosen dictionary

       Attributes
       ----------
       X : np.ndarray
           input data at time t in [M, N] format, where M is the number of input data,
           N is the dimension of data

       Y : np.ndarray
           input data at time t+1 in [M, N] format, where M is the number of input data,
           N is the dimension of data

       dictionary_strategy: class Strategy
            a class presents chosen dictionary

       koopman_matrix: np.ndarray
            approximated koopman operator K

       left_eigenvectors: np.ndarray
            left eigenvector of approximated K

       koopman_modes: np.ndarray
            koopman modes

       PsiX: np.ndarray
            Psi(X)
       PsiY: np.ndarray
            Psi(Y)
    """

    def __init__(self, X, Y, dictionary_strategy):

        self.X = X
        self.Y = Y
        self.dictionary_strategy = dictionary_strategy
        # self.dictionary

    def compute_koopman_operator(self):
        """function that get approxiamted koopman operator
        """
        self.dictionary_strategy.ini(self.X)
        
        Psi_X = self.dictionary_strategy.dictionary(self.X)
        Psi_Y = self.dictionary_strategy.dictionary(self.Y)
        
        self.PsiX = Psi_X
        self.PsiY = Psi_Y
        
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
        """function that calculate koopman eigenfunctions

        Parameters
        ----------
        test_X : np.ndarray
            data of test input
        Returns
        -------
        koopman eigenfunctions

        """
        Psi_test_X = self.dictionary_strategy.dictionary(test_X)
        return Psi_test_X@self.right_eigenvectors
        

    def predict_next_timestep(self, X_initial):
        """function that predicts dynamical system for next time step

        Parameters
        ----------
        X_initial : np.ndarray
            initial data at time t
        Returns
        -------
        data at time t+1

        """
        Phi = self.compute_koopman_eigenfunctions(X_initial) # 100x25
        Mu  = np.diag(self.koopman_eigenvalues) # 25x25
        V = self.koopman_modes #2x25
        
        
        X_next = Phi @ Mu @ V.T #100x2
        
        return X_next
    
    def predict_n_timestep(self, X_initial, n):
        """function that predicts dynamical system for next n steps

        Parameters
        ----------
        X_initial : np.ndarray
            initial data at time t
        Returns
        -------
        data at next n steps

        """
        Phi = self.compute_koopman_eigenfunctions(X_initial) # 100x25
        Mu  = np.diag(self.koopman_eigenvalues) # 25x25
        V = self.koopman_modes #2x25
        
        
        X_n = Phi @ np.linalg.matrix_power(Mu,n) @ V.T #100x2
        
        return X_n

    def sort_eig(self, matrix):
        """function that sort eigenvalues and eigenvectors

        Parameters
        ----------
        matrix : np.ndarray
            the matrix that we want to sort the eigenvalues and eigenvectors
        Returns
        -------
        tuple:
            (eigenvalues, left eigenvector, right eigenvector)
        """
        eig_vals, eig_lvecs, eig_rvecs = scipy.linalg.eig(matrix, left=True, right=True)
        ind = eig_vals.argsort()[::-1]
        return (eig_vals[ind], eig_lvecs[:, ind].T, eig_rvecs[:, ind])
