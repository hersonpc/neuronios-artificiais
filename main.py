# Exemplo do prof. Altino Dandas
# Referência: https://gist.github.com/altinodantas/2959bf535ade107ed8a95ef3781d8ea9#file-atividade_single_neuron-py

import numpy as np
import matplotlib.pyplot as plt

def plotar(w1,w2,bias,title):
    xvals = np.arange(-1, 3, 0.01)     
    newyvals = (((xvals * w2) * - 1) - bias) / w1
    plt.plot(xvals, newyvals, 'r-')    
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.axis([-1,2,-1,2])
    plt.plot([0,1,0],[0,0,1], 'b^')
    plt.plot([1],[1], 'go')
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.show()
    
def main():
    X = [[1,1],[1,0],[0,1],[0,0]]   # Dados de entrada parta lógica AND
    d = [1,-1,-1,-1]                # Saídas esperadas 
    
    # Implemente a função Percepton que deve retornar o vetor de pesos e o bias, respectivamente.
    w, bias = perceptron(max_it=100, E=1, alpha=.1, X=X, d=d)
    plotar(w[0],w[1],bias,"Porta lógica AND com Perceptron")

    # Implemente a função Adaline que deve retornar o vetor de pesos e o bias, respectivamente.
    w, bias = adaline(max_it=100, Epsilon=.0000001, alpha=.1, X=X, d=d)
    plotar(w[0],w[1],bias,"Porta lógica AND com Adaline")
    
if __name__ == '__main__':
    main()