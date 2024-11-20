from eprank import eprank
from utils import load_data
import matplotlib.pyplot as plt
import numpy as np
import os

def main(use_cache=True):
    # load data
    W, G, M, N = load_data()


    cache_file = 'cw2/cache/eprank_samples.npy'
    if use_cache and os.path.exists(cache_file):
        mean_player_skills, precision_player_skills = np.load(cache_file)
        num_iters = mean_player_skills.shape[1]
    else:
        num_iters = 20
        # run message passing algorithm, returns mean and precision for each player
        mean_player_skills, precision_player_skills = eprank(G, M, num_iters)
        np.save(cache_file, (mean_player_skills, precision_player_skills))

    fig, ax = plt.subplots(2, 1, figsize=(10, 4))

    players = np.arange(5)
    for i, player in enumerate(players):
        ax[0].plot(mean_player_skills[player, :], label=f'Player {W[player]}')
        ax[1].plot(precision_player_skills[player, :], label=f'Player {W[player]}')
        

    ax[0].legend()
    fig.tight_layout()

    fig.savefig('cw2/figures/B/convergence.eps')

    plt.show()

if __name__ == '__main__':
    main()