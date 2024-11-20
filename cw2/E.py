from utils import load_data
import numpy as np
import matplotlib.pyplot as plt


def main():
    W, G, M, N = load_data()
    
    cache_file = 'cw2/cache/gibbs_samples.npy'
    gibbs_samples = np.load(cache_file)
    gibbs_samples = gibbs_samples[:,10:]
    
    gibbs_mu = np.mean(gibbs_samples, axis=1)
    gibbs_sigma = np.std(gibbs_samples, axis=1)
    
    cache_file = 'cw2/cache/eprank_samples.npy'
    mean_player_skills, precision_player_skills = np.load(cache_file)
    mean_player_skills = mean_player_skills[:,-1]
    precision_player_skills = precision_player_skills[:,-1]
    
    eprank_mu = mean_player_skills
    eprank_sigma = np.sqrt(1/precision_player_skills)
    
    sorted_idx = np.argsort(gibbs_mu)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 20))
    xx = np.arange(M)
    
    ax.barh(xx+1/3, gibbs_mu[sorted_idx], xerr=gibbs_sigma[sorted_idx], alpha=0, ecolor='r')
    ax.scatter(gibbs_mu[sorted_idx], xx+1/3, marker='x', color='r')
    
    ax.barh(xx, eprank_mu[sorted_idx], xerr=eprank_sigma[sorted_idx], alpha=0, ecolor='b')
    ax.scatter(eprank_mu[sorted_idx], xx, marker='x', color='b')
    ax.set_yticks(xx, labels=W[sorted_idx])
    
    
    k = np.zeros(M)
    n = np.zeros(M)
    for i in range(N):
        k[G[i,0]] += 1
        n[G[i,0]] += 1
        n[G[i,1]] += 1
        
    winrate_mu = k/n
    winrate_sigma = np.sqrt(k*(n-k)/n**3)
    
    ax.barh(xx-1/3, winrate_mu[sorted_idx], xerr=winrate_sigma[sorted_idx], alpha=0, ecolor='g')
    ax.scatter(winrate_mu[sorted_idx], xx-1/3, marker='x', color='g')
        

    
    plt.show()

    
    
if __name__ == '__main__':
    main()