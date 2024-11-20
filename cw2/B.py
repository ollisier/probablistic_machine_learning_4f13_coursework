import os

import numpy as np
import matplotlib.pyplot as plt

from utils import load_all


def main(W, G, M, N, gibbs_samples, mean_player_skills, precision_player_skills):
    fig, ax = plt.subplots(2, 1, figsize=(10, 4))

    plot_iters = 40

    players = np.arange(0,5)
    for player in players:
        ax[0].plot(mean_player_skills[player, :plot_iters], label=W[player])
        ax[1].plot(precision_player_skills[player, :plot_iters], label=W[player])
        
    
    ax[0].set_ylabel('Player Skill Mean')
    ax[1].set_ylabel('Player Skill Precision')
    ax[1].set_xlabel('Iteration')
    
    ax[1].legend(loc='lower right', prop={'size': 8})
    ax[0].grid(True)
    ax[1].grid(True)

    fig.tight_layout()
    plt.legend
    fig.savefig('cw2/figures/B/samples.pdf')
    
    mean_convergence = np.max(np.abs(np.diff(mean_player_skills, axis=1)), axis=0)
    precision_convergence = np.max(np.abs(np.diff(precision_player_skills, axis=1)), axis=0)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    
    ax.semilogy(mean_convergence[:500], label='Mean')
    ax.semilogy(precision_convergence[:500], label='Precision')
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Absolute Change')
    
    fig.savefig('cw2/figures/B/convergence.pdf')

    
    
    

if __name__ == '__main__':
    main(*load_all())
    plt.show()
