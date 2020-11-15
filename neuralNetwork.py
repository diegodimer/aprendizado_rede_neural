import numpy as np
import re

np.set_printoptions(precision=5, formatter={'all':lambda x: f'\t{x:.05f}'})
np.printoptions(precision=5)

class NeuralNetwork():

    activation = None # lista de valores de ativação. activation[x] é uma matriz com os valores de ativação dos neuronios na camada x
    theta_list = None # lista de np.matrix(array) pros pesos de cada layer. theta_list[x] é uma matriz de pesos para a layer x
    df = None
    reg_factor = None
    layers = None
    output_columns = None
    learning_rate = None

    '''
        options['regularization']: guarda fator de regularização
        options['neurons_per_layer']: lista de neurônios por camada, contando a primeira (número de features) e a última (número de outputs)
        options['bias']: lista de matrizes unidimensionais de peso para cada camada
        options['thetas']: opcional, lista de matrizes de peso para cada camada
        options['df']: dataframe contendo todas as instâncias, incluindo colunas de output
        options['output_columns']: lista de nomes das colunas de output. Garantido que os nomes são strings e não números
        options['learning_rate']: o learning rate alpha para atualizar os pesos
        options['min_improvement']: opcional, melhora minima para determinar a parada do backpropagation
        options['mini_batch_size']: opcional, tamanho do batch para backprop. Se é None usa o dataset inteiro (batch). Se 1 é totalmente estocástico
    '''
    def train(self, options, debug=False):
        self.reg_factor = options['regularization']
        self.layers = options['neurons_per_layer']
        self.output_columns = options['output_columns']
        self.learning_rate = options['learning_rate']
        self.df = options['df']
        self.theta_list = options.get('thetas', self.gen_random_thetas())
        self.activation = []
        self.mini_batch_size = options.get("mini_batch_size", len(self.df.index))
        min_improvement = options.get('min_improvement', 0.0005)
        for i in self.layers:
            self.activation.append(np.ones((i+1,1))) #bias nas camadas ocultas/inicial. tem na última mas não retorna

     
        if not debug:
            stop = False
            J,_,_ = self.backpropagation(self.df)
            i = 0
            while not stop:
                new_J,_,_ = self.backpropagation(self.df)

                if np.abs(new_J - J) < min_improvement: #melhorou menos que o minimo: para
                    stop = True
                else:
                    J = new_J
                # if i%5 == 0:
                #     # print('new_J: '+str(new_J))
                # i += 1

        return self

    def predict(self, inference_data, debug=False):
        """
        retorna uma lista com os valores de ativação da última camada
        """
        number_of_entries = len(inference_data.index) + 1

        entries = inference_data.tolist()
        entries.insert(0, 1) ## inserir o bias (por isso o mais 1 ali no número de entradas)
        self.activation[0] = np.array(entries).reshape(number_of_entries,1) # preenche os primeiros neuronios (os de entrada)


        if debug:
            entrada = inference_data.to_numpy()
            print('\t\ta1:\t1.00000\t' + re.sub(r"[\[\]]", r"", str(entrada)) + '\n')

        for i in range(1,len(self.layers)):
            self.calculate_layer_activation(i, debug) #calcula a ativação da rede toda

        return self.activation[-1][1:] # retorna os pesos na camada de saída. Tirando o bias

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

            try:
                self.activation[layer][1:] = np.matmul(self.theta_list[layer-1], self.activation[layer-1])
            except Exception:
                raise Exception(f"Could not multiply matrixes with shapes {self.theta_list[layer-1].shape} and {self.activation[layer-1].shape}")
            if debug:
                a_without_regularization = f"\t\tz{layer+1}: " + str(self.activation[layer][1:].T)
                print(re.sub(r"[\[\]]", r"", a_without_regularization))

            self.activation[layer][1:] = 1 / (1 + np.exp(-(self.activation[layer][1:])))

            # if layer < len(self.layers)-1:
            #     self.activation[layer] = np.append(np.array([[1]]), self.activation[layer], axis=0) #adiciono o neuronio de bias pra próxima camada

            if debug:
                if layer == len(self.layers)-1:
                    b4_sigmoid = f"\t\ta{layer+1}: " + str(self.activation[layer][1:].T) + '\n'
                    print(re.sub(r"[\[\]]", r"", b4_sigmoid))
                else:
                    b4_sigmoid = f"\t\ta{layer+1}: " + str(self.activation[layer].T) + '\n'
                    print(re.sub(r"[\[\]]", r"", b4_sigmoid))

    def calculate_cost_function(self, test_set, theta_list, regularizar=True):
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
            f_k = self.predict(entry.drop(self.output_columns)) 
            # y_k e f_k são listas (com as saídas esperadas pra cada neuronio na camada de saída)
            cummulative_sum = 0
            for i in range(len(f_k)):
                cummulative_sum += (-y_k[i] * np.log(f_k[i])) - ((1-y_k[i]) * np.log(1-f_k[i]))

            cost_function += cummulative_sum

        cost_function = cost_function/n

        theta_sum = 0
        for i in theta_list:
            theta_sum += (np.square(i[:,1:])).sum() # não somo o bias, tiro a primeira coluna

        if regularizar:
            cost_function += (self.reg_factor / (2*n)) * theta_sum

        return float(cost_function)


    def gen_random_thetas(self): # primeira camada não tem thetas entrando nela
        thetas = []
        for i, element in enumerate(self.layers[1:]):
            thetas.append(np.random.rand(element, self.layers[i]+1)*2 - 1.00001)
        return thetas

    def backpropagation(self, test_set, debug=False):
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
        gradients = [None] * (L-1)
        mini_batch_index = 0
        cost_function = 0

        for _,entry in test_set.sample(frac=1).iterrows():
            y_k = []
            for i in self.output_columns:
                y_k.append(entry[i])
            y_k = np.matrix(y_k).reshape(len(y_k),1) # transforma a saída esperada numa matriz coluna também
            f_k = self.predict(entry.drop(self.output_columns))

            if not debug: #calcula a função de custo pra entrada logo depois de prever ela
                for i in range(len(f_k)):
                    cost_function += (-y_k[i] * np.log(f_k[i])) - ((1-y_k[i]) * np.log(1-f_k[i]))

            #1. calcular os deltas de cada neurônio:
            deltas = [None]*L

            # caso especial da camada de saída
            deltas[L - 1] = f_k - y_k

            if debug:
                saida = re.sub(r"[\[\]]", r"", str(deltas[L-1].T))
                print('\t\tdelta'+str(L) + ':' + saida.replace(",", "   "))

            # resto das camadas
            for i in reversed(range(1,L - 1)): # primeira camada não tem ativação
                deltas[i] = []

                first = deltas[i+1].T.dot(self.theta_list[i][:,1:])

                second = np.subtract(self.activation[i][1:], np.square(self.activation[i][1:]))

                deltas[i] =  np.multiply(first.T,second) # element-wise multiplication
                if debug:
                    camada = str(deltas[i].T)
                    print_camada = re.sub(r"[\[\]]", r"", camada) #['%.5f' % n for n in deltas[i]]
                    print('\t\tdelta'+str(i+1) + ':' + print_camada.replace(",", "   ") )


            #2. calcular os gradientes dos pesos:

            for i in reversed(range(len(self.theta_list))):
                if mini_batch_index == 0:
                    gradients[i] = self.activation[i].dot(deltas[i+1].T).T
                else:
                    gradients[i] += self.activation[i].dot(deltas[i+1].T).T
                
                if debug:
                    print('\n\t\tGradientes de Theta'+str(i+1)+':')
                    gr = str(gradients[i])
                    print(re.sub(r"[\[\]]", r"",gr))

            mini_batch_index+= 1

            if mini_batch_index == self.mini_batch_size:
                mini_batch_index = 0
                if not debug:
                    # 3. atualizar cada peso com o valor do gradiente:
                    for i in range(len(self.theta_list)):
                        self.theta_list[i][:,1:] = self.theta_list[i][:,1:] - self.learning_rate * ( (gradients[i][:,1:]/self.mini_batch_size) + (self.reg_factor/len(test_set.index))*self.theta_list[i][:,1:])              
                        self.theta_list[i][:,1] = self.theta_list[i][:,1] - self.learning_rate * ((np.asarray(gradients[i])[:,1])/self.mini_batch_size) # bias, np.asarray to correct bug from multiplying np.matrix with np.array

        n = len(test_set.index)
        cost_function = cost_function/n

        theta_sum = 0
        for i in self.theta_list:
            theta_sum += (np.square(i[:,1:])).sum() # não somo o bias, tiro a primeira coluna

        cost_function += (self.reg_factor / (2*n)) * theta_sum
        return float(cost_function), gradients, self.theta_list # só usado no debug
