from neuralNetwork import NeuralNetwork
import numpy as np
import pandas as pd
from KFoldValidation import KFoldValidation
from main import _normalize_df
from copy import deepcopy

def _get_statistics(population):
        population_sum = sum(population)
        population_size = len(population)

        mean = population_sum/population_size

        result = 0
        for i in population:
            result += (i-mean)**2
        result = result/population_size
        standard_deviation = result**1/2

        return mean, standard_deviation

def gen_random_thetas(neurons_per_layer): # primeira camada não tem thetas entrando nela
    thetas = []
    for i, element in enumerate(neurons_per_layer[1:]):
        thetas.append(np.random.rand(element, neurons_per_layer[i]+1))
    return thetas

def read_dataset(df_name):
    return pd.read_csv(df_name, sep='\t')

df =  read_dataset("datasets/wine-recognition.tsv")
_normalize_df(df)
output_columns = ['target_0.0', 'target_0.5', 'target_1.0']

options = {
    'regularization': 0.1,
    'neurons_per_layer': [13, 20, 20, 3],
    'df': df,
    'output_columns': output_columns,
    'learning_rate': 0.25, 
    'task': 'classification',
    'train_algorithm': NeuralNetwork(),
    'num_folds': 10,
    'label_column': 'target',
    'mini_batch_size': 5,
    'task': 'classification'
}


# nn = NeuralNetwork().train(options)
acc_list = []
for i in range(1):
    kfold = KFoldValidation()
    options['df'] = deepcopy(df)
    acc_list.append(kfold.train_with_kfold(options))
    del kfold
print(f"Acurácia média rg 0.1: {_get_statistics(acc_list)}")

options['regularization'] = 0.5
acc_list = []
for i in range(1):
    kfold = KFoldValidation()
    options['df'] = deepcopy(df)
    acc_list.append(kfold.train_with_kfold(options))
    del kfold
print(f"Acurácia média rg 0.5: {_get_statistics(acc_list)}")

options['regularization'] = 1
acc_list = []
for i in range(1):
    kfold = KFoldValidation()
    options['df'] = deepcopy(df)
    acc_list.append(kfold.train_with_kfold(options))
    del kfold
print(f"Acurácia média rg 1: {_get_statistics(acc_list)}")

options['regularization'] = 0.25
options['learning_rate'] = 0.1
acc_list = []
for i in range(1):
    kfold = KFoldValidation()
    options['df'] = deepcopy(df)
    acc_list.append(kfold.train_with_kfold(options))
    del kfold
print(f"Acurácia média lr 0.1: {_get_statistics(acc_list)}")

options['learning_rate'] = 0.25
acc_list = []
for i in range(1):
    kfold = KFoldValidation()
    options['df'] = deepcopy(df)
    acc_list.append(kfold.train_with_kfold(options))
    del kfold
print(f"Acurácia média lr 0.25: {_get_statistics(acc_list)}")

options['learning_rate'] = 1
acc_list = []
for i in range(1):
    kfold = KFoldValidation()
    options['df'] = deepcopy(df)
    acc_list.append(kfold.train_with_kfold(options))
    del kfold
print(f"Acurácia média lr 1: {_get_statistics(acc_list)}")


# import time

# regularization = [ 0.1, 0.25, 0.5, 0.50, 0.75 ]
# regularization_2 = [ '0.1', '0.25', '0.5', '0.50', '0.75' ]
# acc_list = []
# time_list = []

# for i in regularization:
#     kfold = KFoldValidation()
#     options['regularization'] = i
#     options['df'] = deepcopy(df)
#     start = time.time()
#     acc_list.append(kfold.train_with_kfold(options))
#     print(f' with {i} batch_size took {time.time()-start}s')
#     time_list.append((time.time()-start)/60)
#     del kfold

# import matplotlib.pyplot as plt
# plt.plot(regularization_2, acc_list)
# for i,j in zip(regularization_2, acc_list):
#     plt.annotate(f"{j:.2f}", xy=(i,j))
# plt.xlabel("Fator de Regularização")
# # plt.xticks(rotation=45)
# plt.ylabel("Acurácia média no Kfold com 5 folds")
# plt.savefig("regularization_acc.png", bbox_inches='tight')

# plt.cla()
# plt.plot(regularization_2, time_list)
# for i,j in zip(regularization_2, time_list):
#     plt.annotate(f"{j:.2f}", xy=(i,j))
# plt.xlabel("Fator de Regularização")
# plt.ylabel("Tempo para treinamento do Kfold com 5 folds")
# # plt.xticks(rotation=45)
# plt.savefig("regularization_time.png", bbox_inches='tight') 

