import random

###
# Ref.: https://gist.github.com/marcoscastro/491bd5837815fe11181dce6c50f457ee
#       https://waldirbertazzijr.com/index.php/2019/05/15/redes-neurais-3-implementacao-de-um-perceptron-em-python/
###


class Perceptron():

    def __init__(self, amostras, saidas, taxa_aprendizado=0.1, epocas=1000, limiar=-1):
        self.amostras = amostras  # todas as amostras
        self.saidas = saidas  # saídas respectivas de cada amostra
        # taxa de aprendizado (num. entre 0 e 1)
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas  # número de épocas para o treinamento
        self.limiar = limiar  # limiar
        self.num_amostras = len(amostras)  # quantidade de amostras
        # quantidade de elementos por amostra
        self.num_amostra = len(amostras[0])
        self.pesos = []  # vetor de pesos

    def treinar(self):
        # adiciona -1 para cada uma das amostras
        for amostra in self.amostras:
            amostra.insert(0, -1)

        # inicia o vetor de pesos com valores aleatórios
        for i in range(self.num_amostra):
            self.pesos.append(random.random())

        # insere o limiar no vetor de pesos
        self.pesos.insert(0, self.limiar)

        # inicia o contador de epocas
        num_epocas = 0

        while True:
            # inicializar o erro como inexiste
            erro = False

            # para todas as amostras de treinamento
            for i in range(self.num_amostras):
                u = 0

                # realiza o somatório, o limite (self.amostra + 1)
                # é porque foi inserido o -1 para cada amostra

                for j in range(self.num_amostra + 1):
                    u += self.pesos[j] * self.amostras[i][j]

                    # obtém a saída da rede utilizando a função de ativação
                y = self.sinal(u)

                # verifica se a saída da rede é diferente da saída desejada
                if y != self.saidas[i]:

                    # calcula o erro: subtração entre a saída desejada e a saída da rede
                    erro_aux = self.saidas[i] - y

                    # faz o ajuste dos pesos para cada elemento da amostra
                    for j in range(self.num_amostra + 1):
                        self.pesos[j] = self.pesos[j] + self.taxa_aprendizado * \
                            erro_aux * self.amostras[i][j]

                    erro = True  # ainda existe erro

                # incrementa o número de épocas
                num_epocas += 1

                # critério de parada é pelo número de épocas ou se não existir erro
                if num_epocas > self.epocas or not erro:
                    break

    def testar(self, amostra, classe1, classe2):
        pass

# def perceptron(max_it, E, a, X, d):

#     # Inicialização de variáveis
#     w = 0
#     b = 0

#     t = 1
#     while t < max_it and E > 0:
#         for i in
