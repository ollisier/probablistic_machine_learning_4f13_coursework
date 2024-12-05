import numpy as np
import matplotlib.pyplot as plt


from utils import load_all
from bmm import BMM

def main(A, B, V):
    K = 20
    T = 50
    alpha = 1
    gamma = 0.1
    seeds = [0, 10, 100]

    for seed in seeds:
        np.random.seed(seed)
        perplexity, swk, sk_docs = BMM(A, B, K, T, alpha, gamma)
        theta = (sk_docs + alpha) / np.sum(sk_docs + alpha, axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(4, 3))
        
        idx = np.argsort(-theta[-1, :])
        
        for k in range(K):
            ax.plot(np.arange(T+1), theta[:, idx[k]])
            
        ax.set_xlabel('Gibbs iteration')
        ax.set_ylabel('Topic proportion')
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(f'cw3/figures/D/mixing_proportions_seed_{seed}.pdf')
        
        print(f'Perplexity for seed {seed}: {perplexity}')
    
if __name__ == '__main__':
    main(*load_all())
    plt.show()
