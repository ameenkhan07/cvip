import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

UBIT = 'ameenmoh'
np.random.seed(sum([ord(c) for c in UBIT]))

OUTPUT_DIR = "outputs/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# For Color Quantization task
img_name = "./baboon.jpg"


def _show_plot(points, Mu, Mu_color, color_map=[], _save=True, filename='temp.png'):
    """
    """
    if len(color_map):
        k = color_map
    else:
        k = '1'
    fig = plt.figure(figsize=(5, 5))

    # Scatter Points
    plt.scatter(points[0], points[1], facecolor=k,
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


def _quantization(points, K, filename):
    """Classify points according to the centroids
    """
    #  Centroid Matrix
    Mu = [[round(np.random.uniform(0, 255), 2)
           for i in range(3)] for j in range(K)]
    # print('Centroid POINTS', Mu, ' for ', K)
    # Populate Distance matrix of K, 3 -> R, G, and B
    dist = [[[_rgb_euclidian_distance(p, m) for m in Mu]
             for p in row]for row in points]
    # Get minimum value for each each distance
    remapped_points = [[Mu[np.argmin(p)] for p in row]for row in dist]
    mu_classified = [[np.argmin(p) for p in row]for row in dist]
    # print(remapped_points)
    # print(mu_classified)
    _save(filename, np.asarray(remapped_points))


def _rgb_euclidian_distance(p1, p2):
    """Euclidian Distance of R, G, B values of 2 points
    """
    return(round(
        np.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[0] - p2[0]) ** 2 +
            (p1[0] - p2[0]) ** 2
        ), 2
    ))


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
    # _show_plot(points, Mu, Mu_color, 'task3_iter0.png')

    # Part 1 : Kmeans Classification, points according to the initial centroids
    closest, color_map = _assignment(points, Mu, Mu_color)
    print('Classification Vector (First Iteration): ',
          np.vstack([points, color_map]))
    _show_plot(points, Mu, Mu_color, color_map=color_map,
               filename='task3_iter1_a.png')

    # Part 2 : Update Centroids
    Mu = _update_Mu(Mu, closest, points)
    print('Updated Centroids : ', Mu)
    _show_plot(points, Mu, Mu_color, color_map=color_map,
               filename='task3_iter1_b.png')

    # Part 3.a : Kmeans Classification second iteration
    # Recompute coloring based on new centroids
    closest, color_map = _assignment(points, Mu, Mu_color)
    print('Classification Vector (Second Iteration) : ',
          np.vstack([points, color_map]))
    _show_plot(points, Mu, Mu_color, color_map=color_map,
               filename='task3_iter2_a.png')

    # Part 3.b : Update Centroids for the second time
    Mu = _update_Mu(Mu, closest, points)
    print('Updated Centroids : ', Mu)
    _show_plot(points, Mu, Mu_color, color_map=color_map,
               filename='task3_iter2_b.png')

    # Part 4 : Colour Quantization
    img = cv.imread(img_name)
    K = [(3, 'task3_baboon_3.jpg'), (5, 'task3_baboon_5.jpg'),
         (10, 'task3_baboon_10.jpg'), (20, 'task3_baboon_20.jpg')]

    for i, k in enumerate(K):
        _quantization(img, k[0], k[1])
