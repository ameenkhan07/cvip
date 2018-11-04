import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy.stats import multivariate_normal


def save_elliptical_plot():
    """
    """
    pass


def gmm_expectation_maximization(X, Mu, epsilon, pi):
    """2 Step algo
    Expectation : Calculate probability for all datapoints to be in 
    all clusters (Mu, epsilon)
    """
    # Calculating membership of all points in all clusters
    # E Step of GMM Algo, where mu and epsilon are cluster
    p = [[pi[j]*multivariate_normal.pdf(i, Mu[j], epsilon[j])
          for j in range(3)] for i in X]
    # print(p)

    # Recompute Gaussian parameters
    # M step : Maximum likelihood expectation of params(pi , Mu, Epsilon)
    new_Mu = []
    # new_Epsilon = []
    for j, _ in enumerate(Mu):
        # 1. Compute new means
        mu_x = [ sum(   [i_[0]*p[i][j] for i, i_ in enumerate(X)]) /
                 sum([p[i][j] for  i, _ in enumerate(X)])]
        mu_y = [ sum(   [i_[1]*p[i][j] for i, i_ in enumerate(X)]) /
                 sum([p[i][j] for  i, _ in enumerate(X)])]
        new_Mu.append([mu_x, mu_y])
        # 2. Compute new variances
        ## Todo
    
    print('Recomputed Mu : ', new_Mu)
    # print('Recomputed Epsilon : ', new_Epsilon)


if __name__ == '__main__':

    # To test the implemented gmm functions here

    # Part 1 : GMM on given data
    X = np.asarray([
        [5.9, 3.2], [4.6, 2.9], [6.2, 2.8], [4.7, 3.2], [5.5, 4.2], [
            5.0, 3.0], [4.9, 3.1], [6.7, 3.1], [5.1, 3.8], [6.0, 3.0]
    ])

    # Centroids, and centroid-color mapping
    Mu = {
        0: [6.2, 3.2],
        1: [6.6, 3.7],
        2: [6.5, 3.0]
    }
    # Covariance Metrix
    epsilon = [[[0.5, 0], [0, 0.5]], [
        [0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.5]]]
    pi = [1/3, 1/3, 1/3]
    gmm_expectation_maximization(X, Mu, epsilon, pi)
