import numpy as np
'''
    ****substitui np.matrix por np.array (aparentemente funciona exatamente igual)*****

    nota na documentação de numpy.matrix:

    It is no longer recommended to use this class, even for linear algebra. 
    Instead use regular arrays. The class may be removed in the future.
'''
class NeuralNetwork():
    
    activation = None # lista de valores de ativação. activation[x] é uma matriz com os valores de ativação dos neuronios na camada x
    theta_list = None # lista de np.matrix(array) pros pesos de cada layer. theta_list[x] é uma matriz de pesos para a layer x
    bias = None # lista de pesos dos neuronios de bias. bias[x] é uma matriz (array?) dos pesos de bias na layer x

    '''
        options['regularization']: guarda fator de regularização
        options['neurons_per_layer']: lista de neurônios por camada, contando a primeira (número de features) e a última (número de outputs)
        options['bias']: lista de matrizes unidimensionais de peso para cada camada
        options['thetas']: lista de matrizes de peso para cada camada
        options['df']: dataframe contendo todas as instâncias, incluindo colunas de output
        options['output_columns']: lista de nomes das colunas de output. Garantido que os nomes são strings e não números
        options['learning_rate']: o learning rate alpha para atualizar os pesos
    '''
    def __init__(self, options):
        """
        reg_factor = factor de regularização (lambda)
        layers = número de neuronios em cada camada da rede neural, no mínimo 2 números diferentes (n de neuronios de entrada e de saída)
        """
        self.reg_factor = options['regularization']
        self.layers = options['neurons_per_layer']
        self.bias = options['bias']
        self.theta_list = options['thetas']
        self.output_columns = options['output_columns']
        self.learning_rate = options['learning_rate']

        self.activation = [None]*len(self.layers)
        

    def predict(self, inference_data, debug = False):
        """
        retorna uma lista com os valores de ativação da última camada
        """
        number_of_entries = len(inference_data.index)

        self.activation[0] = inference_data.to_numpy().reshape(number_of_entries,1) # preenche os primeiros neuronios (os de entrada)

        if debug:
            entrada = ['%.5f' % n for n in  inference_data.to_numpy()]
            print('\t\ta1: [1.00000 ' + ' '.join(entrada)+ ']')

        for i in range(1,len(self.layers)): 
            self.calculate_layer_activation(i, debug) #calcula a ativação da rede toda

        return self.activation[-1] # retorna os pesos na camada de saída

    def calculate_layer_activation(self, layer, debug):
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
        if layer > 0:
            #print('computing: '+str(self.theta_list[layer-1]) + ' times '+ str(self.activation[layer-1]))
            self.activation[layer] = np.matmul(self.theta_list[layer-1],self.activation[layer-1])
            #print('result')
            #print(self.activation[layer])
            if debug:
                b4_sigmoid = ['%.5f' % n for n in (self.bias[layer-1] + self.activation[layer])]
                print('\n\t\tz' + str(layer+1) + ': [' + ' '.join(b4_sigmoid)+ ']')

            self.activation[layer] = 1 / (1 + np.exp(-(self.bias[layer-1] + self.activation[layer])))
            #print('result2')
            #print(self.activation[layer])
            if debug:
                after_sigmoid = ['%.5f' % n for n in self.activation[layer]]
                if layer != len(self.layers) - 1:
                    print('\t\ta' + str(layer+1) + ': [1.00000 ' + ' '.join(after_sigmoid)+ ']')
                else: 
                    print('\t\ta' + str(layer+1) + ': [' + ' '.join(after_sigmoid)+ ']') # última camada não tem neurônio de bias
                    print('\n\t\tf(x): [' + ' '.join(after_sigmoid)+ ']')


    def calculate_cost_function(self, test_set):
        """
        Calcula função de custo J para o conjunto de treinamento {test_set}
        J(theta) = 1/n * [ ( (-y_k) * (log( f(x_k) ) - (1 - y_k) * (log(1 - f(x_k)) ) ) ) ]para todos os neuronios de saída, e para todos exemplos + (lambda / 2*n)* (soma de todos os pesos)
        n: tamanho do {test_set}
        y_k = rotulo correto da entrada k
        x_k = entrada k
        f(x_k) = rotulo predito para entrada x_k
        """
        n = len(test_set.index) # número de linhas
        cost_function = 0 
        
        for _,entry in test_set.iterrows():
            y_k = []
            for i in self.output_columns:
                y_k.append(entry[i])
            f_k = self.predict(entry.drop(self.output_columns)) #need to change this for output_layers as well
            # y_k e f_k são listas (com as saídas esperadas pra cada neuronio na camada de saída)
            cummulative_sum = 0
            for i in range(len(f_k)):
                cummulative_sum += (-y_k[i] * np.log(f_k[i])) - ((1-y_k[i]) * np.log(1-f_k[i]))

            cost_function += cummulative_sum[0]
            
        cost_function = cost_function/n
      
        theta_sum = 0
        for i in self.theta_list:
            theta_sum += i.sum() # cada elemento da lista de thetas é uma np.matrix

        cost_function += (self.reg_factor / 2*n) * theta_sum
        
        return cost_function


    def backpropagation(self, test_set, debug = False):
        '''
            Atualiza os pesos da rede neural com base nas instâncias de treino

            1. calcular os deltas de cada neurônio:
                para a camada de saída: delta = (f_x - y)
                para as camadas ocultas: delta = sum(theta_(k+1)*delta_(k+1))*ativação_k*(1-ativação_k), onde k é a camada atual
            2. calcular os gradientes dos pesos:
                d/d(theta_ijk) = ativação_jk*delta_ik+1, onde theta_ijk é o peso entre o neurÔnio j na camada k e o neurônio i na camada k+1
            3. atualizar cada peso com o valor do gradiente:
                theta_ijk = theta_ijk -learning_rate*d/d(theta_ijk)
        '''
        L = len(self.activation)


        for _,entry in test_set.iterrows():
            y_k = []
            for i in self.output_columns:
                y_k.append(entry[i])
            f_k = self.predict(entry.drop(self.output_columns))

            #1. calcular os deltas de cada neurônio:
            deltas = [None]*L

            # caso especial da camada de saída
            deltas[L - 1] = f_k - y_k

            if debug:
                saida = ['%.5f' % n for n in deltas[L-1]]
                print('\t\tdelta'+str(L) + ': [' + ' '.join(saida) + ']')

            # resto das camadas
            for i in reversed(range(1,L - 1)): # primeira camada não tem ativação
                deltas[i] = []
                #print('theta_list[i]: '+str(self.theta_list[i]))
                first = deltas[i+1].T.dot(self.theta_list[i])
          
                second = np.subtract(self.activation[i], np.square(self.activation[i]))

                deltas[i] =  np.multiply(first.T,second) # element-wise multiplication
                if debug:
                    #print('deltas[i]: ' +str(deltas[i]))
                    print_camada = ['%.5f' % n for n in deltas[i]]
                    print('\t\tdelta'+str(i+1) + ': [' + ' '.join(print_camada) + ']')


            #2. calcular os gradientes dos pesos:
            gradients = [None] * (L-1)
            bias_gradients = [None]*(L-1) # n sei se é L
            for i in reversed(range(len(self.theta_list))):
                gradients[i] = self.activation[i].dot(deltas[i+1].T).T + self.reg_factor*self.theta_list[i]
              
                bias_gradients[i] = np.array(1*deltas[i+1]) # bias não leva regularização
                if debug:
                    print('\t\tGradientes de Theta'+str(i+1)+':')
                    for j in range(len(gradients[i])):
                        print_gradients = ['%.5f' % n for n in gradients[i][j]]
                        print('\t\t\t' + '%.5f ' % bias_gradients[i][j] +' '.join(print_gradients))
                    print(' ')
            

            if not debug:
                # 3. atualizar cada peso com o valor do gradiente:
                for i in range(len(self.theta_list)):
                    #print('trying to' + str(self.theta_list[i]) + ' + 0.00001 * '+ str(gradients[i]))
                    #print('type of gradients:')
                    #print(type(gradients[i]))
                    self.theta_list[i] = self.theta_list[i] - self.learning_rate*gradients[i]

                for i in range(len(self.bias)):
                    #print('trying to' + str(self.bias[i]) + ' + 0.00001 * '+ str(bias_gradients[i]))
                    #print('type of bias gradients:')
                    #print(type(bias_gradients[i]))
                    self.bias[i] = self.bias[i] - self.learning_rate*bias_gradients[i]

        return (bias_gradients, gradients) # só usado no debug











