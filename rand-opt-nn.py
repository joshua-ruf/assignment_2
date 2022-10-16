# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import mlrose
import os
import time
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

os.chdir('../assignment_1')

from helpers import load_data

# +
X, y, _ = load_data(ebert=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)

# +
"""
uses log loss (cross-entropy)
"""
algorithms = [
    'random_hill_climb',
    'simulated_annealing', 
    'genetic_alg',
    'gradient_descent'
]

random_states = range(8)
training_sizes = np.arange(0.3, 1.01, 0.1)
hidden_nodes_list = [[8]]
max_iters_list = range(100, 401, 100)

out = []
for i, (alg, rs, training_size, nodes, max_iters) in enumerate(product(algorithms, random_states, training_sizes, hidden_nodes_list, max_iters_list)):
    nn = mlrose.NeuralNetwork(algorithm=alg,
                              hidden_nodes=nodes,
                              activation='sigmoid',
                              early_stopping=True,
                              clip_max=5,
                              random_state=rs,
                              max_iters=max_iters,
                              curve=True)
    
    np.random.seed(rs)
    sample = np.random.choice(X_train.shape[0], size=int(X_train.shape[0]*training_size), replace=False)
    X_train_sub = X_train[sample]
    y_train_sub = y_train.iloc[sample]
    
    t0 = time.time()
    nn.fit(X_train_sub, y_train_sub)
    duration = time.time() - t0
    out.append({
        'algorithm': alg,
        'duration': duration,
        'nn': nn,
        'random_state': rs,
        'train_accuracy': nn.score(X_train_sub, y_train_sub),
        'test_accuracy': nn.score(X_test, y_test),
        'f1_train': f1_score(nn.predict(X_train_sub), y_train_sub),
        'f1_test': f1_score(nn.predict(X_test), y_test),
        'fitness_curve': len(nn.fitness_curve),
        'sample_size': X_train_sub.shape[0],
        'hidden_nodes': nodes,
        'max_iters': max_iters,
    })
    
    if i % 10 == 0:
        print(i)
       

# -

df = pd.DataFrame(out)
df.to_csv('../assignment_2/20221015-nn-2.csv')




