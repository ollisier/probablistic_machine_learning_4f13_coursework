import numpy as np
import matplotlib.pyplot as plt
from gibbsrank import gibbs_sample
from utils import load_data
import os

def main(use_cache=True):
    # set seed for reproducibility
    np.random.seed(0)
    # load data
    W, G, M, N = load_data()

    cache_file = 'cw2/cache/gibbs_samples.npy'
    if use_cache and os.path.exists(cache_file):
        skill_samples = np.load(cache_file)
        num_iters = skill_samples.shape[1]
    else:
        # number of iterations
        num_iters = 10000
        # perform gibbs sampling, skill samples is an num_players x num_samples array
        skill_samples = gibbs_sample(G, M, num_iters)
        np.save(cache_file, skill_samples)

    # Sample Plot
    N_players = 3
    players = np.arange(N_players)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    for i, player in enumerate(players):
        ax.plot(skill_samples[player, :200], label=f'Player {W[player]}')
    ax.set_title(f'Skill Samples for Player {W[player]}')
    ax.set_ylabel('Skill Level')
    ax.grid(True)
    ax.set_xlabel('Iteration')
    ax.legend()

    fig.tight_layout()
    fig.savefig('cw2/figures/A/skill_samples.eps')

    # Autocovariance Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    autocov = np.zeros((M, num_iters))
    for p in range(M):
        autocov[p, :] = (np.correlate(skill_samples[p, :]-np.mean(skill_samples[p, :]), skill_samples[p, :]-np.mean(skill_samples[p, :]), mode='full')[num_iters-1:])/np.arange(num_iters, 0, -1)/np.var(skill_samples[p, :])

    max_lag = 20
    for p in range(M):
        ax.plot(autocov[p, :max_lag], label=f'Player {W[p]}')

    ax.set_title('Autocovariance of Skill Samples')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocovariance')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig('cw2/figures/A/auto_covariance.eps')

    # Convergence/Burn In Plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 4))
    ax[0].plot(np.mean(skill_samples[:, :50], axis=0), label='Mean of skills')
    ax[0].set_title('Mean of Skills Over Iterations')
    ax[0].set_ylabel('Mean Skill Level')
    ax[0].grid(True)

    ax[1].plot(np.std(skill_samples[:, :50], axis=0), label='Standard deviation of skills')
    ax[1].set_title('Standard Deviation of Skills Over Iterations')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Standard Deviation')
    ax[1].grid(True)

    fig.tight_layout()
    fig.savefig('cw2/figures/A/convergence.eps')


    s = np.median(np.var(skill_samples, axis=1))
    print(200000*s)

    plt.show()
    
if __name__ == '__main__':
    main()