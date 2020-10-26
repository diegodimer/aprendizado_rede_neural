import numpy as np

class NeuralNetwork():
    
    activation = None # lista de valores de ativação. activation[x] é uma matriz com os valores de ativação dos neuronios na camada x
    theta_list = None # lista de np.matrix pros pesos de cada layer. theta_list[x] é uma matriz de pesos para a layer x
    bias = None # lista de pesos dos neuronios de bias. bias[x] é uma matriz dos pesos de bias na layer x

    def __init__(self, reg_factor, *layers):
        """
        reg_factor = factor de regularização (lambda)
        layers = número de neuronios em cada camada da rede neural, no mínimo 2 números diferentes (n de neuronios de entrada e de saída)
        """
        #TO-DO: passar os initial_weights aqui também, então provavelmente trocar esse *layers por um *kwargs
        pass

    def predict(self, inference_data):
        """
        retorna uma lista com os valores de ativação da última camada
        """
        number_of_entries = len(inference_data.index)

        self.activation[0] = inference_data.to_numpy().reshape(number_of_entries,1) # preenche os primeiros neuronios (os de entrada)

        for i in range(1,layers):
            self.calculate_layer_activation(i) #calcula a ativação da rede toda

        return self.activation[-1] # retorna os pesos na camada de saída

    def calculate_layer_activation(self, layer):
        """
        Calcula a função de ativação dos neurônios na camada {layer}
        layer: número da camada 
        pra camada 1 não calcula, porque os valores de ativação são as próprias entradas
        theta_list[x] = são os pesos da camada x, da forma  θ11 θ12 θ13
                                                            θ21 θ22 θ23
                                                            θ31 θ32 θ33
        ativation[x] é os valores de ativação na camada x na forma  a1
                                                                    a2
                                                                    a3
        """
        if layer > 1:
            self.activation[layer] = self.theta_list[layer-1] * self.activation[layer-1]
            self.activation[layer] = 1 / (1 + np.exp(-(self.bias[layer] + self.activation[layer]))

    def calculate_cost_funcion(self, test_set):
        """
        Calcula função de custo J para o conjunto de treinamento {test_set}
        J(theta) = 1/n * [ ( (-y_k) * (log( f(x_k) ) - (1 - y_k) * (log(1 - f(x_k)) ) ) ) ]para todos os neuronios de saída, e para todos exemplos + (lambda / 2*n)* (soma de todos os pesos)
        n: tamanho do {test_set}
        y_k = rotulo correto da entrada k
        x_k = entrada k
        f(x_k) = rotulo predito para entrada x_k
        """
        n = len(test_set.index) # número de linhas
        cost_function = 1/n
        for entry in test_set:
            y_k = []
            for i in self.output_layers:
                y_k.append(entry[i])
            f_k = self.predict(entry.drop(self.label_column))
            # y_k e f_k são listas (com as saídas esperadas pra cada neuronio na camada de saída)
            cummulative_sum = 0
            for i in range(len(f_k)):
                cummulative_sum += (-y_k[i] * log(f_k[i])) - ((1-y_k[i]) * log(1-f_k[1]))
            cost_function += cummulative_sum

        theta_sum = 0
        for i in self.theta_list:
            theta_sum += i.sum() # cada elemento da lista de thetas é uma np.matrix

        cost_function += (self.reg_factor / 2*n) * theta_sum

        return cost_function
