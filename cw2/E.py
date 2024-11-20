import numpy as np
import matplotlib.pyplot as plt

from utils import load_all


def main(W, G, M, N, gibbs_samples, mean_player_skills, precision_player_skills):
    
    
    # Gibbs
    gibbs_mu = np.mean(gibbs_samples[:,100:], axis=1)
    gibbs_sigma = np.std(gibbs_samples[:,100:], axis=1)
    
    # EP
    ep_mu = mean_player_skills[:,-1]
    ep_sigma = np.sqrt(1/precision_player_skills[:,-1])
    
    # Winrate
    k = np.zeros(M)
    n = np.zeros(M)
    for i in range(N):
        k[G[i,0]] += 1
        n[G[i,0]] += 1
        n[G[i,1]] += 1
        
    winrate_mu = np.sqrt(12)*(k/n - 0.5)
    winrate_sigma = np.sqrt(12*k*(n-k)/n**3)

    
    # Plotting
    sorted_idx = np.argsort(gibbs_mu)
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    xx = np.arange(M)
    
    ax.bar(xx+1/3, gibbs_mu[sorted_idx], yerr=gibbs_sigma[sorted_idx], alpha=0, ecolor='r')
    ax.scatter(xx+1/3, gibbs_mu[sorted_idx], marker='x', color='r', label='Gibbs')
    
    ax.bar(xx, ep_mu[sorted_idx], yerr=ep_sigma[sorted_idx], alpha=0, ecolor='b')
    ax.scatter(xx, ep_mu[sorted_idx], marker='x', color='b', label='EP')
    
    ax.bar(xx-1/3, winrate_mu[sorted_idx], yerr=winrate_sigma[sorted_idx], alpha=0, ecolor='g')
    ax.scatter(xx-1/3, winrate_mu[sorted_idx], marker='x', color='g', label='Winrate')
    
    ax.set_xticks(xx, labels=W[sorted_idx], rotation=90)
    ax.set_ylabel('Skill')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlim(-1, M)
    fig.tight_layout()
    
    fig.savefig('cw2/figures/E/ranking.pdf')
    
if __name__ == '__main__':
    main(*load_all())
    plt.show()
