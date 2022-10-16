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

import pandas as pd
import numpy as np
import mlrose
import time
import random
from itertools import product


def run_and_time_knapsack(algorithm, N=10, **kwargs):
    
    np.random.seed(0)
    weights = np.random.choice(range(1, 6), size=N)
    values = np.arange(1, N+1)
    fitness = mlrose.Knapsack(weights, values)
    
    problem = mlrose.DiscreteOpt(length=N, fitness_fn=fitness, max_val=N)
    
    t0 = time.time()
    _, best_fitness, curve = algorithm(problem, curve=True, **kwargs)
    duration = time.time() - t0
    
    return {
        'fitness': fitness.__class__.__name__,
        'algorithm': algorithm.__name__,
        'input_size': N,
        'number_of_iterations': len(curve),
        'best_fitness': best_fitness,
        'time': duration,
        **kwargs,
    }



# +

algorithms = [
    mlrose.random_hill_climb,
    mlrose.simulated_annealing,
    mlrose.genetic_alg,
    mlrose.mimic,
]

lengths = range(5, 21, 5)

random_states = [0]

iterations = [float('inf')]

out = []
for i, (N, alg, rs, max_iters) in enumerate(product(lengths, algorithms, random_states, iterations)):
    if i % 10 == 0:
        print(i)
    
    if alg == mlrose.mimic:
        max_iters = 20
    
    temp = run_and_time_knapsack(alg, N=N, random_state=rs, max_iters=max_iters)
    out.append(temp)

df = pd.DataFrame(out)
df.to_csv(f'20221015-rand-opt{i}-knapsack.csv', index=False)
# -

out




