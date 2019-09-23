import numpy as np
from numpy import matlib

def inicializarMatrizAleatoria(linhas, colunas):
    matrix = np.array((np.random.uniform(low=0.5, high=-0.5, size=(linhas,colunas))))
    return matrix