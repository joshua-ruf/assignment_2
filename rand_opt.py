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
from itertools import product
from tqdm import tqdm


def run_and_time(fitness, algorithm, N=100, **kwargs):
    
    problem = mlrose.DiscreteOpt(length=N, fitness_fn=fitness)

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
fitness_functions = [
    mlrose.OneMax(),
    mlrose.FlipFlop(),
    mlrose.FourPeaks(),
    mlrose.SixPeaks(),
    mlrose.ContinuousPeaks(),
]

algorithms = [
    mlrose.random_hill_climb,
    mlrose.simulated_annealing,
    mlrose.genetic_alg,
    mlrose.mimic,
]

lengths = range(10, 81, 10)

random_states = [0]

iterations = [float('inf')]

out = []
for i, (N, fit, alg, rs, max_iters) in enumerate(product(lengths, fitness_functions, algorithms, random_states, iterations)):
    if i % 10 == 0:
        print(i)
    
    if alg == mlrose.mimic:
        max_iters = 20
    
    temp = run_and_time(fit, alg, N=N, random_state=rs, max_iters=max_iters)
    out.append(temp)


df = pd.DataFrame(out)
df.to_csv(f'20221014-rand-opt{i}.csv', index=False)

# +
# df.to_csv('20221013-rand-opt-240.csv', index=False)
# -

x = pd.DataFrame(out)
x.to_csv(f'20221014-rand-opt-{len(x)}.csv', index=False)







