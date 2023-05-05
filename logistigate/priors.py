'''
Contains prior classes for use with Bayesian inference methods
'''

import numpy as np
import scipy.special as sps
import scipy.stats as spstat

class prior_normal_assort:
    """
    Defines the class instance of an assortment of independent normal priors, with associated mu (mean)
    and scale vectors in the logit-transfomed [0,1] range, and the following methods:
        rand: generate random draws from the distribution
        lpdf: log-likelihood of a given vector
        lpdf_jac: Jacobian of the log-likelihood at the given vector
        lpdf_hess: Hessian of the log-likelihood at the given vector
    beta inputs may be a Numpy array of vectors
    """
    def __init__(self, mu=sps.logit(0.1), covar=np.array([1.]).reshape((1,1))):
        self.mu = mu
        self.covar = covar
    def rand(self, n=1):
        return np.random.multivariate_normal(self.mu, self.covar, n)
    def expitrand(self, n=1): # transformed to [0,1] space
        return sps.expit(np.random.multivariate_normal(self.mu, self.covar, n))
    def lpdf(self, beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        return np.squeeze(spstat.multivariate_normal.logpdf(beta, self.mu, self.covar))
    def lpdf_jac(self, beta): # ONLY TO BE USED WITH INDEPENDENT NORMALS
        if beta.ndim == 1:  # reshape to 2d
            beta = np.reshape(beta, (1, -1))
        jac = -(1/np.diag(self.covar)) * (beta - self.mu)
        return np.squeeze(jac)
    def lpdf_hess(self, beta):
        if beta.ndim == 1:  # reshape to 2d
            beta = np.reshape(beta, (1, -1))
        (k, n) = beta.shape
        hess = np.tile(np.zeros(shape=(n, n)), (k, 1, 1))
        for i in range(k):
            hess[i] = np.diag(-(1 / np.diag(self.covar)))
        return np.squeeze(hess)