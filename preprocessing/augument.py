import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations

def permute_3D(cube):
    for x in permutations([0, 1, 2], 3):
        yield cube.transpose(x)