import pathlib

import numpy as np


def load_dataset(filename):
    X = np.loadtxt(pathlib.Path(__file__).parent / 'clustering-classes' / f"{filename}_X.csv", delimiter=",")
    y = np.loadtxt(pathlib.Path(__file__).parent / 'clustering-classes' / f"{filename}_y.csv", delimiter=",")
    return X, y



