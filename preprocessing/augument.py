import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations

def rotate_3D(cube):
    lstCube = []
    for x in permutations([0, 1, 2], 3):
        lstCube.append(cube.transpose(x))
    return lstCube