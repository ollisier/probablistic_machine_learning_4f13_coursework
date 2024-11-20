import os

import numpy as np
import matplotlib.pyplot as plt

from utils import load_all
import A
import B
import C
import D
import E

def main(use_cache=True):
    W, G, M, N, gibbs_samples, mean_player_skills, precision_player_skills = load_all(use_cache)
    
    problems = [A, B, C, D, E]
    
    for problem in problems:
        print(f'----------Running {problem.__name__}----------')
        problem.main(W, G, M, N, gibbs_samples, mean_player_skills, precision_player_skills)
    
if __name__ == '__main__':
    main()
    plt.show()