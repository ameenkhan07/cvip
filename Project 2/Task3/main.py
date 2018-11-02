import matplotlib.pyplot as plt
import numpy as np

UBIT = 'ameenmoh'
np.random.seed(sum([ord(c) for c in UBIT]))

# Requirements of projects
points = np.asarray([
    [5.9, 4.6, 6.2, 4.7, 5.5, 5.0, 4.9, 6.7, 5.1, 6.0],
    [3.2, 2.9, 2.8, 3.2, 4.2, 3.0, 3.1, 3.1, 3.8, 3.0]
])

# Centroids
Mu = {
    1: [6.2, 3.2],
    2: [6.6, 3.7],
    3: [6.5, 3.0]
}
Mu_color = {1: 'r', 2: 'g', 3: 'b'}


def _show_plot(points, Mu, Mu_color, color_map=[]):
    """
    """
    if len(color_map):
        k = color_map
    else:
        k = '1'
    fig = plt.figure(figsize=(5, 5))
    
    plt.scatter(points[0], points[1], color=k,
                marker='^', edgecolor='b')
    # Set Centroids
    for i in Mu.keys():
        plt.scatter(*Mu[i], color=Mu_color[i])
    plt.xlim(4, 7)
    plt.ylim(2, 5)
    plt.show()


def _assignment(points, Mu, Mu_color):
    """Calculate Euclidean distance from provided Centroids
    """
    # Calculate Distance matrix for the points to the given centroids Mu
    distance = [np.sqrt(
                (points[0] - Mu[i][0]) ** 2
                + (points[1] - Mu[i][1]) ** 2
                ) for i in Mu.keys()]
    closest, color = [], []
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
    color = [Mu_color[i] for i in closest]
    # print(Mu_color, closest)
    return (closest, color)


_show_plot(points, Mu, Mu_color)
closest, color_map = _assignment(points, Mu, Mu_color)
_show_plot(points, Mu, Mu_color, color_map=color_map)
