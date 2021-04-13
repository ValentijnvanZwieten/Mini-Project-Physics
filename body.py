import numpy as np


class Body():
    '''
    TODO doc
    '''

    def __init__(self, mass, position):
        self.m = mass
        self.c = position
        self.v = np.zeros(2)

