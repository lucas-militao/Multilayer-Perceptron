from Dataset import Dataset as dt
from Functions import InicializarPesos as ip
import numpy as np
from numpy import matlib
from simpy import *
from MultiLayerPerceptron.MultiLayerPerceptron import MultiLayerPerceptron as mlp

taxaAp = 0.01
prec = pow(10, -6)
neuroniosEsc = 2

def main():

    arquivoXLarge = open("../Xlarge.txt", "r")
    arquivoXsmall = open("../Xsmall.txt", "r")

    Xlarge = dt.getData(arquivoXLarge)
    Xsmall = dt.getData(arquivoXsmall)

    MLP = mlp(Xlarge, Xsmall, taxaAp, prec, neuroniosEsc)

    # print(MLP.treinamento())

    x = np.array(((5,5),(5,5)))
    print(np.sum(x))
    # print(1/(np.cosh(x)**2))

main()