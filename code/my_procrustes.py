from math import atan
from math import cos
from math import sin

import numpy as np
from scipy.linalg import norm


def procrustes(X, Y, scaling=False, reflection=False):
    """
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows).

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.0).sum()
    ssY = (Y0**2.0).sum()

    # centered Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != "best":
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {"rotation": T, "scale": b, "translation": c}

    return d, Z, tform


# #########################################


def get_translation(shape):
    """
    Calculates a translation for x and y
    axis that centers shape around the
    origin
    Args:
      shape(2n x 1 NumPy array) an array
      containing x coodrinates of shape
      points as first column and y coords
      as second column
     Returns:
      translation([x,y]) a NumPy array with
      x and y translationcoordinates
    """

    mean_x = np.mean(shape[::2]).astype(int)
    mean_y = np.mean(shape[1::2]).astype(int)

    return np.array([mean_x, mean_y])


def translate(shape):
    """
    Translates shape to the origin
    Args:
      shape(2n x 1 NumPy array) an array
      containing x coodrinates of shape
      points as first column and y coords
      as second column
    """
    mean_x, mean_y = get_translation(shape)
    shape[::2] -= mean_x
    shape[1::2] -= mean_y


def get_rotation_scale(reference_shape, shape):
    """
    Calculates rotation and scale
    that would optimally align shape
    with reference shape
    Args:
        reference_shape(2nx1 NumPy array), a shape that
        serves as reference for scaling and
        alignment

        shape(2nx1 NumPy array), a shape that is scaled
        and aligned

    Returns:
        scale(float), a scaling factor
        theta(float), a rotation angle in radians
    """

    a = np.dot(shape, reference_shape) / norm(reference_shape) ** 2

    # separate x and y for the sake of convenience
    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]

    x = shape[::2]
    y = shape[1::2]

    b = np.sum(x * ref_y - ref_x * y) / norm(reference_shape) ** 2

    scale = np.sqrt(a**2 + b**2)
    theta = atan(b / max(a, 10**-10))  # avoid dividing by 0

    return round(scale, 1), round(theta, 2)


def get_rotation_matrix(theta):
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])


def procrustes_analysis(reference_shape, shape):
    """
    Reference shape ~ ground truth
    Shape ~ prediction
    """
    temp_P0 = []
    temp_P = []

    for px, py in zip(reference_shape[:, 0], reference_shape[:, 1]):
        temp_P0 += [px, px]
    P0 = np.array(temp_P0)
    # P = np.array(temp_P.append[px, px] for px, py in zip(shape[:, 0], shape[:, 1]))
    for px, py in zip(shape[:, 0], shape[:, 1]):
        temp_P += [px, py]
    P = np.array(temp_P)

    # translate(P0)
    # translate(P)

    scale, theta = get_rotation_scale(P0, P)

    R = get_rotation_matrix(theta)

    return theta, R
