from neuralNetwork import NeuralNetwork
import numpy as np
import pandas as pd
from KFoldValidation import KFoldValidation

def gen_random_thetas(neurons_per_layer): # primeira camada não tem thetas entrando nela
    thetas = []
    for i, element in enumerate(neurons_per_layer[1:]):
        thetas.append(np.random.rand(element, neurons_per_layer[i]+1))
    return thetas

def read_dataset(df_name):
    return pd.read_csv(df_name, sep='\t')

df =  read_dataset("datasets/house-votes-84.tsv")

output_columns = ['target_0', 'target_1']

options = {
    'regularization': 0.25,
    'neurons_per_layer': [16,10,10,2],
    'df': df,
    'output_columns': output_columns,
    'learning_rate': 0.001, 
    'task': 'classification',
    'train_algorithm': NeuralNetwork(),
    'num_folds': 5,
    'label_column': 'target'
}

# nn = NeuralNetwork().train(options)
import time
start = time.time()
kfold = KFoldValidation()

kfold.train_with_kfold(options)
print(f'{time.time()-start}')