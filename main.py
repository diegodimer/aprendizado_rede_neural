'''
	comando para rodar:

		python backpropagation.py network.txt initial_weights.txt dataset.txt

	arquivos e formato de saida como descrito no enunciado

'''
# testa backpropagation como descrito no enunciado
import numpy as np
import pandas as pd
import sys

from neuralNetwork import NeuralNetwork

if __name__ == '__main__':
	network = sys.argv[1]
	initial_weights = sys.argv[2]
	dataset = sys.argv[3]

	# get regularization factor and number of neurons
	f = open(network,'r')
	content = f.readlines()
	regularization = float(content[0])
	neurons_per_layer = []
	for i in range(1,len(content)):
		neurons_per_layer.append(int(content[i]))

	f.close()

	# get initial weight values
	f = open(initial_weights,'r')
	content = f.readlines()
	bias = []
	thetas = []
	for i in range(len(content)):
		neurons = content[i].split(';')
		b = []
		t = []
		for j in range(len(neurons)):
			weights = neurons[j].split(',')
			b.append(float(weights[0])) # first value is always bias
			t.append([float(k) for k in weights[1:]])
		b = np.array(b).reshape(len(b),1) # matriz unidimensional
		bias.append(b)
		t= np.array(t)
		thetas.append(t)
	
	f.close()

	#get data and output columns
	f = open(dataset,'r')
	content = f.readlines()
	data = []
	features_and_outputs = content[0].split(';')
	number_of_features = len(features_and_outputs[0].split(','))
	number_of_outputs = len(features_and_outputs[1].split(','))
	for i in range(len(content)):
		data.append([float(k) for k in content[i].replace(';',',').split(',')])

	output_columns = [str(x) for x in range(number_of_features,number_of_features+number_of_outputs)]
	df = pd.DataFrame.from_records(data)
	df.columns = [str(x) for x in range(len(data[0]))] # column name is string values, for consistency
	
	f.close()

	'''
		no fim desse pré-processamento:

		regularization: guarda fator de regularização
		neurons_per_layer: lista de neurônios por camada, contando a primeira (número de features) e a última (número de outputs)
		bias: lista de matrizes unidimensionais de peso para cada camada
		thetas: lista de matrizes de peso para cada camada
		df: dataframe contendo todas as instâncias, incluindo colunas de output
		output_columns: lista de nomes das colunas de output. Garantido que os nomes são strings e não números
	'''
	print('|--------------------------------------------------------------------------------|')
	print("Parametro de regularizacao lambda = "+ '%.5f' % regularization)
	print("\nInicializando rede com a seguinte estrutura de neuronios por camadas: " + str(neurons_per_layer))

	for i in range(len(neurons_per_layer)-1):
		print("\nTheta"+str(i+1) + " inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):")
		for j in range(neurons_per_layer[i+1]): # camada de features não tem pesos
			remaining_weights = ['%.5f' % n for n in  thetas[i][j]]
			print('\t ' + "%.5f" % bias[i][j] + ' ' + ' '.join(remaining_weights))
	print('\nConjunto de treinamento')
	for index,row in df.iterrows():
		print('\tExemplo '+str(index+1))
		y_counter = 1
		for number,name in enumerate(df.columns, 1):
			if name in output_columns:
				print('\t\ty' + str(y_counter) + ": " + '%.5f' % row[name])
				y_counter += 1
			else:
				print('\t\tx' + str(number) + ": " + '%.5f' % row[name])

	print('|--------------------------------------------------------------------------------|')


	options = {
		'regularization': regularization,
		'neurons_per_layer': neurons_per_layer,
		'bias': bias,
		'thetas': thetas,
		'df': df,
		'output_columns': output_columns,
		'learning_rate': 0.00001
	}

	nn = NeuralNetwork(options)
	'''
	print(bias)
	print(thetas)
	print(df)
	for entry in df:
		print(entry)
		print(type(entry))
	'''
	print('Calculando erro/custo J da rede')

	for i,row in df.iterrows():
		print('\tProcessando exemplo de treinamento '+str(i+1))
		entrada = ['%.5f' % n for n in  row.drop(output_columns).to_numpy()]

		print('\tPropagando entrada ' + '[' + ' '.join(entrada)+ ']')
		prediction = nn.predict(row.drop(output_columns), debug = True) # retorna [[n]] porque é uma matriz Lx1 ao invés de um array
		prediction = ['%.5f' % n for n in prediction]

		print('\tSaida predita para o exemplo '+ str(i+1) + ': [' + ' '.join(prediction)+']')
		expected = ['%.5f' % n for n in row[output_columns]]
		print('\tSaida esperada para o exemplo '+ str(i+1) + ': [' + ' '.join(expected)+']')
		result = nn.calculate_cost_function(df.iloc[[i]])
		print('\tJ do exemplo '+ str(i+1) + ': ' + '%.3f' % result+'\n')

	print('J total do dataset (com regularizacao): ' + '%.5f' % nn.calculate_cost_function(df) + '\n')

	print('|--------------------------------------------------------------------------------|')
	print('Rodando backpropagation')

	b = []
	g = []
	for i,row in df.iterrows():
		print('\tCalculando gradientes com base no exemplo '+str(i+1))
		r1,r2 = nn.backpropagation(df.iloc[[i]], debug=True)
		b.append(r1)
		g.append(r2)
	print('\tDataset completo processado. Calculando gradientes regularizados')
	avg_b = np.mean(b, axis=0)
	avg_g = np.mean(g, axis=0)
	#print(avg_b)
	#print(avg_g)
	for i in range(len(avg_b)):
		print('\t\tGradientes finais para Theta' + str(i+1) + '(com regularizacao):')
		for j in range(len(avg_b[i])):
			print_average_gradients = ['%.5f' % n for n in avg_g[i][j]]
			print('\t\t\t'  + '%.5f ' % avg_b[i][j] + ' '.join(print_average_gradients))
		
	print('|--------------------------------------------------------------------------------|')




