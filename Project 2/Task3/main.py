import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def _show_plot(points, Mu, Mu_color, color_map=[], _scatter=True, _save=True, filename='temp.png'):
    """
    """
    if len(color_map):
        k = color_map
    else:
        k = '1'
    fig = plt.figure(figsize=(5, 5))

    if _scatter:
        plt.scatter(points[0], points[1], facecolor=k,
                    marker='^', edgecolor='k')
    # Set Centroids
    for i in Mu.keys():
        plt.scatter(*Mu[i], color=Mu_color[i])
    plt.xlim(4, 7)
    plt.ylim(2, 5)
    if _save:
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
    else:
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


def _update_Mu(Mu, closest, points):
    """
    """
    # print(points)
    # print(closest)
    # print(Mu)
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

    # Initial Plot : No Classification
    # _show_plot(points, Mu, Mu_color)

    # Kmeans Classification, points according to the initial centroids
    closest, color_map = _assignment(points, Mu, Mu_color)
    print('Classification Vector (First Iteration): ',
          np.vstack([points, color_map]))
    _show_plot(points, Mu, Mu_color, color_map=color_map,
               filename='task3_iter1_a.png')

    # Update Centroids
    Mu = _update_Mu(Mu, closest, points)
    print('Updated Centroids : ', Mu)
    _show_plot(points, Mu, Mu_color, color_map=color_map,
               _scatter=False, filename='task3_iter1_b.png')

    # Kmeans Classification second iteration
    # Recompute coloring based on new centroids
    closest, color_map = _assignment(points, Mu, Mu_color)
    print('Classification Vector (Second Iteration) : ',
          np.vstack([points, color_map]))
    _show_plot(points, Mu, Mu_color, color_map=color_map,
               filename='task3_iter2_a.png')

    # Update Centroids for the second time
    Mu = _update_Mu(Mu, closest, points)
    print('Updated Centroids : ', Mu)
    _show_plot(points, Mu, Mu_color, color_map=color_map,
               _scatter=False, filename='task3_iter2_b.png')
