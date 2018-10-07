def get_neigborhood(x, y, mat1, mat2, mat3):
    """Return array of elements in the neighborhood
    of the input coordinates (x,y)
    3 Arrays provided for a 3D representation of the neighborhood

    Parameters:
    -----------
        x : int, point in x axis
        y : int, point in y axis
        mat1 : List[List], Matrice at current scale
        mat2 : List[List], Matrice at previous scale
        mat2 : List[List], Matrice at another scale

    Returns:
    --------
        neighbors : List, all element neighboring the array
    """
    neighbors = []
    X, Y = mat1.shape

    neighbor_coord = lambda x, y : [(x_, y_) for x_ in range(x-1, x+2)
    for y_ in range(y-1, y+2)if ((0 <= x_ <= X-1) and (0 <= y_ <= Y-1) and
    (x != x_ or y != y_))]

    neighbors.extend([mat1[x[0]][x[1]] for x in neighbor_coord(x, y)])
    neighbors.extend([mat2[x[0]][x[1]] for x in neighbor_coord(x, y)])
    neighbors.extend([mat3[x[0]][x[1]] for x in neighbor_coord(x, y)])

    return (neighbors)
