import numpy as np
import numpy.matlib
from Functions import InicializarPesos as ip

class MultiLayerPerceptron:

    def __init__(self, amostrasX, saidasDesejadasD, taxaAprendizagem, precisao, neuroniosEscondidos):
        self.amostrasX = np.insert(amostrasX, 0, -1, axis=1)
        self.saidasDesejadasD = saidasDesejadasD
        self.taxaAprendizagem = taxaAprendizagem
        self.precisao = precisao
        self.neuroniosEscondidos = neuroniosEscondidos

#inicializa os pesos
    def inicializarPesos(self, numeroNeuronios, qtdEntradas):
        pesos = ip.inicializarMatrizAleatoria(numeroNeuronios, qtdEntradas)
        return pesos

#Calcula a saída
    def __calculateI(self, pesos, entradas):
        I = pesos.dot(entradas)
        return I

#Cálcula a entrada para os neurônios da camada seguinte
    def __calculateY1(self, saida):
        saida = np.tanh(saida)
        saida = np.insert(saida, 0, -1)
        return saida

#Cálcular a saída da camada de saída
    def __calculateY2(self, saida):
        saida = np.tanh(saida)
        return saida

#Calcula o gradiente da camada de saída
    def __localGragient2(self, saidaD, saidaY, entradaI):
        return (saidaD - saidaY) * (1/(np.cosh(entradaI)**2))
#Atualiza os pesos na camada de saída
    def __atualizarPesos2(self, peso, taxaDeAprendizagem, localGradient, saidaDaCamadaAnterior):
        return peso + taxaDeAprendizagem * localGradient * saidaDaCamadaAnterior

    def __localGradient1(self, gradienteSaida, pesosSaida, saidaI):
        return np.sum(gradienteSaida*pesosSaida) * (1/(np.cosh(saidaI)**2))

    def __calcularErro(self, quantidadeEntradas, saidasD, saidasY):
        return (np.power((np.subtract(saidasD, saidasY))))/2

    def __calcularErroMédio(self, erros, quantidadeEntradas):
        return np.sum(erros)/quantidadeEntradas
#------------------------------------------------------------------------------------------

#Realiza o treinamento mas está faltando a parte do backwards, está realizando apenas o feedforward
    def treinamento(self):
        Em = 0
        EmAtual = 1
        EmAnterior = Em

        w1 = self.inicializarPesos(self.neuroniosEscondidos, np.size(self.amostrasX[0,:])) #pesos da camada escondida
        w2 = self.inicializarPesos(np.size(self.saidasDesejadasD[0,:]), self.neuroniosEscondidos + 1) #pesos da camada de saída
        #É somado mais um self.neuroniosEscondidos + 1(número de colunas) pois é calculado o peso do limiar

        I1 = np.zeros((np.size(self.amostrasX[:,0]), self.neuroniosEscondidos)) #inicializa a matriz que irá receber a saída da camada escondida
        I2 = np.zeros((np.size(self.amostrasX[:,0]), np.size(self.saidasDesejadasD[0,:]))) #inicializa a matriz que irá receber a saída da camada de saída

        Y1 = np.zeros((np.size(self.amostrasX[:,0]), self.neuroniosEscondidos + 1)) #inicializa a matriz que irá calcular a entrada da camada de saída
        Y2 = np.zeros((np.size(self.amostrasX[:,0]), np.size(self.saidasDesejadasD[0,:])))
        #ele soma mais um na self.neuroniosEscondidos + 1 (consiste no número de colunas) pois é acrescentado o valor -1 do limiar

        g1 = np.zeros((np.size(self.amostrasX[:,0]), self.neuroniosEscondidos))
        g2 = np.zeros((np.size(self.saidasDesejadasD[:,0]), np.size(self.saidasDesejadasD[0,:])))

        while(abs(EmAtual - EmAnterior) > self.precisao):

            #Forward
            for i in range(np.size(self.amostrasX[:,0])):
                I1[i] = self.__calculateI(w1, self.amostrasX[i])
                Y1[i] = self.__calculateY1(I1[i])
                I2[i] = self.__calculateI(w2, Y1[i])
                Y2[i] = self.__calculateY2(I2[i])


            #Backward
            for j in range(np.size(g2[:,0])):
                g2[j] = self.__localGragient2(self.saidasDesejadasD[j], Y2[j], I2[j])

            for k in range(np.size(w2[:,0])):
                for l in range(np.size(w2[0,:])):
                    w2[k,l] = self.__atualizarPesos2(w2[k, l], self.taxaAprendizagem, g2[k][l], Y1[k][l])

            # for m in range(g1[:,0]):
            #     for n in range(g2[0,:]):
            #         g2[m,n] = self.__localGragient1()



            EmAtual = 1/100000000000000000000 #coloquei esse valor para realizar apenas uma interação e testar a saída
#-----------------------------------------------------------------------------------

        return w2

