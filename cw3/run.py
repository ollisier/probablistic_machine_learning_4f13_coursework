import os

import numpy as np
import matplotlib.pyplot as plt

from utils import load_all
import A
import C
import D
import E

def main(use_cache=True):
    A_, B_, V = load_all(use_cache)
    
    problems = [A, C, D, E]
    
    for problem in problems:
        print(f'----------Running {problem.__name__}----------')
        problem.main(A_, B_, V)
    
if __name__ == '__main__':
    main()
    plt.show()
