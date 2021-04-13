import numpy as np


def pairwise_difference(a, b):
    '''
    Calculates the pairwise difference between two vectors of points.

    Parameters
    ----------
    a, b : numpy.ndarray
        The 2*N, 2*M vectors.
    
    Returns
    -------
    dx, dy : numpy.ndarray
        A N*M matrix with the pairwise difference on the given axis.
    '''

    # TODO flip?
    dx = a[0, :, None] - b[0, :, None].T
    dy = a[1, :, None] - b[1, :, None].T

    return dx, dy

