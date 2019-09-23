import numpy as np
import re


def __readFile(arquivo): #Lê o documento completo e o separa por linhas
    return arquivo.readlines()

def __convertDataLine(row): #Distribui os elementos de uma linha em um vetor e os converte de String para Int
    return list(map(int,row.split()))

def getData(arquivo): #Retorna uma matriz com todos os elementos do documento

    data = __readFile(arquivo)

    matrix = np.ndarray

    for line in data:
        convertedLine = __convertDataLine(line)
        try:
            matrix = np.vstack((matrix, [convertedLine]))
        except:
            matrix = np.array(convertedLine)

    return matrix


