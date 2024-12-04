import os

import numpy as np
import matplotlib.pyplot as plt

from utils import load_data
# import A
# import B
# import C
# import D
# import E

def main(use_cache=True):
    A, B, V, gibbs_samples, mean_player_skills, precision_player_skills = load_all(use_cache)
    
    problems = [A, B, C, D, E]
    
    for problem in problems:
        print(f'----------Running {problem.__name__}----------')
        problem.main(A, B, V)
    
if __name__ == '__main__':
    # main()
    # plt.show()
    
    load_data()