import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from utils import load_all
from lda import LDA

def main(A, B, V):
    K = 20
    T = 50
    alpha = 1
    gamma = 0.1
    
    cache = 'cw3/cache/lda_samples.npy'
    params = (K, T, alpha, gamma)
    
    recompute = False
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            params_loaded, data = pickle.load(f)
        if params_loaded != params:
            recompute = True
    else:
        recompute = True
        
    if recompute:
        data = LDA(A, B, K, T, alpha, gamma)
        with open(cache, 'wb') as f:
            pickle.dump((params, data), f)
    
    perplexity, swk, skd = data
    
    theta = (skd + alpha) / np.sum(skd + alpha, axis=1)[:, np.newaxis, :]
    
    documents = [0, 10, 20]
    for d in documents:
        fig, ax = plt.subplots(figsize=(6, 4))
        idx = np.argsort(-theta[-1, :, d])
        ax.plot(np.arange(T+1), theta[:, idx, d])
        ax.set_xlabel('Gibbs iteration')
        ax.set_ylabel('Topic proportion')
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(f'cw3/figures/E/mixing_proportions_doc_{d+1}.pdf')
        
    theta_avg = np.mean(theta, axis=2)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.argsort(-theta_avg[-1, :])
    ax.plot(np.arange(T+1), theta_avg[:, idx])
    
    beta = (swk+gamma) / np.sum(swk+gamma, axis=1)[:, np.newaxis, :]
    entropy_beta = np.sum(-beta*np.log(beta), axis=1)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(T+1), entropy_beta)
    ax.set_xlabel('Gibbs iteration')
    ax.set_ylabel('Entropy (nats)')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f'cw3/figures/E/entropy.pdf')
    
    print(f'Test perplexity = {perplexity}')
    
    
if __name__ == '__main__':
    main(*load_all())
    plt.show()
