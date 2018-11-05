import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from gaussian_mixture_model import *
from old_faithful import *
import random

UBIT = 'ameenmoh'
np.random.seed(sum([ord(c) for c in UBIT]))
random.seed(sum([ord(c) for c in UBIT]))

OUTPUT_DIR = "outputs/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# For Color Quantization task
img_name = "./baboon.jpg"


def _show_plot(points, Mu, Mu_color, color_map, _save=True, filename='temp.png'):
    """Plot Graphs
    """

    fig = plt.figure(figsize=(5, 5))

    # Scatter Points
    plt.scatter(points[0], points[1], facecolor=color_map,
                marker='^', edgecolor='k')
    # Add coodinates to the scatter points
    for i, j in zip(points[0], points[1]):
        plt.text(i, j-0.08, f'({i},{j})', fontsize=4.5)

    # Set Centroids and add annoted coordinate text
    for i in Mu.keys():
        plt.scatter(*Mu[i], color=Mu_color[i])
        plt.text(Mu[i][0], Mu[i][1]-0.08,
                 f'({Mu[i][0]},{Mu[i][1]})', fontsize=4.5)

    plt.xlim(4, 7)
    plt.ylim(2, 5)
    if _save:
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
    else:
        plt.show()


def _save(filename, img):
    """Saves the image with filename in output dir 
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # filename = filename+'.png'
    filename = os.path.join(OUTPUT_DIR, filename)
    # print(filename, img.shape)
    cv.imwrite(filename, img)


def euclidian_distance(p):
    """Euclidian Distance of 2 points
    Works for 2D and 3D
    """
    return(np.sqrt(sum([(p[0][i]-p[1][i])**2 for i, _ in enumerate(p)])))


def _assignment(points, Mu):
    """Calculate Euclidean distance from provided Centroids
    """
    # Calculate Distance matrix for the points to the given centroids Mu
    distance = [euclidian_distance([points, Mu[i]]) for i in Mu.keys()]
    closest = []
    # print(distance)
    # Get the closest centroid for every point using distance matrix
    for i in range(len(points[0])):
        _MIN, _INDEX = 10000, 100
        for j in range(len(Mu.keys())):
            # print(distance[j][i])
            if distance[j][i] < _MIN:
                _MIN = distance[j][i]
                _INDEX = j
        closest.append(_INDEX+1)
    return (closest)


def _update_Mu(Mu, closest, points):
    """
    """
    new_Mu = {}
    for i in Mu.keys():
        x = round(np.mean([points[0][j1]
                           for j1, j2 in enumerate(closest) if i == j2]), 2)
        y = round(np.mean([points[1][j1]
                           for j1, j2 in enumerate(closest) if i == j2]), 2)
        # print(x,y)
        new_Mu[i] = [x, y]
    # print(new_Mu)
    return new_Mu


def kmeans_quantization(img, K, filename):
    """
    """
    print(
        f'\n\nKMEANS quantization for cluster Size : {K}, Output File : {filename} (Running...)\n\n')
    # Random initialization of Mu
    from pprint import pprint
    Mu = [img[random.sample(range(img.shape[0]), K)[i]][random.sample(
        range(img.shape[1]), K)[i]] for i in range(K)]
    # pprint(Mu)

    # Kmeans Clustering until convergence
    for ranger in range(40):
        print(f'----------------{ranger}--------------------')
        ranger += 1
        # Classify the img matrix to respective classes
        dist = [[[euclidian_distance([p, m]) for m in Mu]
                 for p in row]for row in img]
        # Get the index of the closest point
        closest = [[np.argmin(p) for p in row]for row in dist]
        #  Recompute Centroids for next iteration
        new_Mu = []
        for m, ele in enumerate(Mu):
            # Get relevant points lying inside cluster m
            temp = [img[r][c]
                    for r, rows in enumerate(closest) for c, col in enumerate(rows) if m == col]
            R_new, G_new, B_new = np.mean([v[0] for v in temp]), np.mean([
                v[1] for v in temp]), np.mean([v[2] for v in temp])
            new_Mu.append([R_new, G_new, B_new])
        # pprint(new_Mu)

        # Update Mu
        if np.array_equal(Mu, new_Mu):
            break
        Mu = deepcopy(new_Mu)
    print(f' Final Centroids for {K} clusters : \n', Mu)
    # Quantize all points using centroid colour and final classificiation
    remapped_points = [[Mu[np.argmin(p)] for p in row]for row in dist]
    # Save Quantized image matrix
    _save(filename, np.asarray(remapped_points))
    print('\nImage saved !!!\n')


if __name__ == '__main__':
    # Cartesian points to be clusters
    points = np.asarray([
        [5.9, 4.6, 6.2, 4.7, 5.5, 5.0, 4.9, 6.7, 5.1, 6.0],
        [3.2, 2.9, 2.8, 3.2, 4.2, 3.0, 3.1, 3.1, 3.8, 3.0]
    ])

    # Centroids, and centroid-color mapping
    Mu = {
        1: [6.2, 3.2],
        2: [6.6, 3.7],
        3: [6.5, 3.0]
    }
    Mu_color = {1: 'r', 2: 'g', 3: 'b'}

    # Part 1 : Kmeans Classification, points according to the initial centroids
    closest = _assignment(points, Mu)
    color_map = [Mu_color[i] for i in closest]
    print('\nClassification Vector (First Iteration): \n',
          np.vstack([points, color_map]))
    _show_plot(points, Mu, Mu_color, color_map=color_map,
               filename='task3_iter1_a.png')

    # Part 2 : Update Centroids
    # Use color map from previous step
    Mu = _update_Mu(Mu, closest, points)
    print('\nUpdated Centroids : \n', Mu)
    _show_plot(points, Mu, Mu_color, color_map=color_map,
               filename='task3_iter1_b.png')

    # Part 3.a : Kmeans Classification second iteration
    # Recompute coloring based on new centroids
    closest = _assignment(points, Mu)
    color_map = [Mu_color[i] for i in closest]
    print('\nClassification Vector (Second Iteration) : \n',
          np.vstack([points, color_map]))
    _show_plot(points, Mu, Mu_color, color_map=color_map,
               filename='task3_iter2_a.png')

    # Part 3.b : Update Centroids for the second time
    # Used color map from previous step
    Mu = _update_Mu(Mu, closest, points)
    print('\nUpdated Centroids : \n', Mu)
    _show_plot(points, Mu, Mu_color, color_map=color_map,
               filename='task3_iter2_b.png')
    print('\n----------------------------------------------\n')

    # Part 4 : Colour Quantization
    img = cv.imread(img_name)
    K = [(3, 'task3_baboon_3.jpg'), (5, 'task3_baboon_5.jpg'),
         (10, 'task3_baboon_10.jpg'), (20, 'task3_baboon_20.jpg')]

    for k in K:
        # kmeans_quantization_alt(img, k[0], k[1])
        kmeans_quantization(img, k[0], k[1])

    # Part 5 GMM , import from another file and run here
    # Part a : GMM on given data
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
    # Mixing Coefficients
    pi = [1/3, 1/3, 1/3]

    print('\nPART 3.5 : GMM EM Algorithm : \n', Mu)
    gmm_expectation_maximization(X, Mu, epsilon, pi)
