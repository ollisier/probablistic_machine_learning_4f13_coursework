import numpy as np
import matplotlib.pyplot as plt

from utils import load_all

def main(W, G, M, N, gibbs_samples, mean_player_skills, precision_player_skills):
    
    num_iters = gibbs_samples.shape[1]
    
    # Sample Plot
    players = np.arange(0,3)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    for i, player in enumerate(players):
        ax.plot(gibbs_samples[player, :200], label=W[player])
        
    ax.set_ylabel('Skill')
    ax.set_xlabel('Iteration')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    
    fig.savefig('cw2/figures/A/skill_samples.pdf')

    # Auto covariance plot
    autocov = np.zeros((M, num_iters))
    for p in range(M):
        autocov[p, :] = (np.correlate(gibbs_samples[p, :]-np.mean(gibbs_samples[p, :]), gibbs_samples[p, :]-np.mean(gibbs_samples[p, :]), mode='full')[num_iters-1:])/np.arange(num_iters, 0, -1)/np.var(gibbs_samples[p, :])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))

    max_lag = 20
    for p in range(M):
        ax.plot(autocov[p, :max_lag], label=f'Player {W[p]}')

    ax.set_xlabel('Lag')
    ax.set_ylabel('Auto Covariance')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig('cw2/figures/A/auto_covariance.pdf')
    
    print(f'Variance increase due to correlation = {1 + 2*np.mean(np.sum(autocov[:,:1000], axis=(1)))}')

    # Convergence/Burn In Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    
    ax.plot(np.mean(gibbs_samples[:, :50], axis=0), label='Skill Population Mean', color='tab:blue')
    ax.grid(True)
    ax.set_ylabel('Mean')

    ax_right = ax.twinx()
    ax_right.plot(np.var(gibbs_samples[:, :50], axis=0), label='Skill Population Variance', color='tab:orange')
    ax_right.set_ylabel('Variance')
    ax.set_xlabel('Iteration')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_right.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    
    fig.savefig('cw2/figures/A/convergence.pdf')

    # Convergence Estimate
    s = np.median(np.var(gibbs_samples, axis=1))
    
    print('Median marginal variance = ', s)
    
if __name__ == '__main__':
    main(*load_all())
    plt.show()
