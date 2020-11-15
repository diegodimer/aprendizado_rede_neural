'''
	comando para rodar:

		python main.py network.txt initial_weights.txt dataset.txt
		./backpropagation network.txt initial_weights.txt dataset.txt
	arquivos e formato de saida como descrito no enunciado

'''
# testa backpropagation como descrito no enunciado
import numpy as np
import pandas as pd
import sys
import re

import warnings
warnings.filterwarnings('ignore')

from neuralNetwork import NeuralNetwork

np.set_printoptions(precision=5, formatter={'all':lambda x: f'\t{x:.05f}'})
np.printoptions(precision=5)

def _normalize_df(df):
	for column in df:
		max_col = df[column].max()
		min_col = df[column].min()
		if max_col == min_col: #divisão por 0 faz dar NaN
			df[column] = 1.0
		else:
			df[column] = (df[column] - min_col) / (max_col - min_col)
		print(column)

def read_initial_weights(initial_weights):
	with open(initial_weights,'r') as f:
		content = f.readlines()
		thetas = []
		for i in content:
			thetas.append(np.asarray(np.matrix(i)))
	return thetas

def read_dataset(dataset):
	with open(dataset,'r') as f:
		content = f.readlines()
		data = []
		features_and_outputs = content[0].split(';')
		number_of_features = len(features_and_outputs[0].split(','))
		number_of_outputs = len(features_and_outputs[1].split(','))
		for i in content:
			data.append([float(k) for k in i.replace(';',',').split(',')])

		output_columns = [str(x) for x in range(number_of_features,number_of_features+number_of_outputs)]
		df = pd.DataFrame.from_records(data)
		df.columns = [str(x) for x in range(len(data[0]))] # column name is string values, for consistency
	return df, output_columns

def read_network_file(network):
	with open(network, 'r') as f:
		content = f.readlines()
		regularization = float(content.pop(0))
		neurons_per_layer = []
		for i in content:
			neurons_per_layer.append(int(i))
	return regularization, neurons_per_layer

if __name__ == '__main__':
	network = sys.argv[1]
	initial_weights = sys.argv[2]
	dataset = sys.argv[3]

	# get regularization factor and number of neurons
	regularization, neurons_per_layer = read_network_file(network)

	# get initial weight values
	thetas = read_initial_weights(initial_weights)

	#get data and output columns
	df, output_columns = read_dataset(dataset)
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
		print( re.sub(r"[\[\] ]", r"\t", str(thetas[i])) )
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
		'thetas': thetas,
		'df': df,
		'output_columns': output_columns,
		'learning_rate': 0.00001
	}

	nn = NeuralNetwork().train(options, True)

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
		result = nn.calculate_cost_function(df.iloc[[i]], thetas, regularizar=False)
		print('\tJ do exemplo '+ str(i+1) + ': ' + '%.3f' % result+'\n')

	print('J total do dataset (com regularizacao): ' + '%.5f' % nn.calculate_cost_function(df, thetas) + '\n')

	print('|--------------------------------------------------------------------------------|')
	print('Rodando backpropagation')

	b = []
	g = []
	for i,row in df.iterrows():
		print('\t\nCalculando gradientes com base no exemplo '+str(i+1))
		_, gradient, theta_list = nn.backpropagation(df.iloc[[i]], debug=True)
		
		for i in range(len(theta_list)):
			gradient[i][:,1:] = gradient[i][:,1:] + (regularization/len(df.index))*theta_list[i][:,1:]

		g.append(gradient)
	print('\t\nDataset completo processado. Calculando gradientes regularizados')
	avg_g = np.mean(g, axis=0)

	for i, g in enumerate(avg_g):
		print('\n\t\tGradientes finais para Theta' + str(i+1) + '(com regularizacao):')
		print(re.sub(r"[\[\]]", r"", str(g)))
		
	print('--------------------------------------------')

	print('Rodando verificacao numerica de gradientes (epsilon=0.0000010000)')
	epsilon = 0.0000010000

	erro = [0] * len(thetas)
	for i, theta in enumerate(thetas):
		print(f'\t\nGradiente numerico de Theta{str(i+1)}:')
		linhas, colunas = theta.shape
		for j in range(linhas):
			for k in range(colunas):
				theta[j,k]+= epsilon # aumenta epsilon
				t1 = nn.calculate_cost_function(df, thetas)
				theta[j,k]-= epsilon # aumenta epsilon
				theta[j,k]-= epsilon # aumenta epsilon
				t1 -= nn.calculate_cost_function(df, thetas)
				theta[j,k]+= epsilon # corrige
				gradiente = t1/(2*epsilon)
				print(f'\t\t{gradiente:.5f}', end='')
				erro[i] += abs(avg_g[i][j,k] - gradiente)
			print('')

	print("Verificando corretude dos gradientes com base nos gradientes numericos:")
	for i in range(len(thetas)):
		lin, col = thetas[i].shape
		den = lin*col
		print(f"\tErro entre gradiente via backprop e gradiente numerico para Theta{i}: {'%.15f' % (erro[i]/den)}")
