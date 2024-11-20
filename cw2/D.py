import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

from utils import load_all


def main(W, G, M, N, gibbs_samples, mean_player_skills, precision_player_skills):
    gibbs_samples = gibbs_samples[:, 100:]
    
    p1_name = 'Rafael Nadal'
    p2_name = 'Roger Federer'
    
    p1 = np.where(W == p1_name)[0][0]
    p2 = np.where(W == p2_name)[0][0]
    
    
    # Gaussian Approximation
    mu = np.mean(gibbs_samples[(p1, p2),:], axis=1)
    cov = np.cov(gibbs_samples[(p1, p2),:])
    
    print(f'Gaussian Approximation: \n mu = {mu} \n cov = {cov}')
    
    ## Marginals
    P_s = norm.cdf((mu[0] - mu[1])/np.sqrt(cov[0,0] + cov[1,1]))
    print(f'Gaussian Marginals P(Nadal > Federer) = {P_s}')
    
    ## Joint
    A = np.array([[1, -1]])
    P_s = norm.cdf((A@mu)/np.sqrt((A@cov@A.T)))
    print(f'Gaussian Joint P(Nadal > Federer) = {P_s}')
    
    ## Plotting
    fig = plt.figure(figsize=(5, 5))
    
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0, hspace=0)
    
    x1 = np.linspace(np.min(gibbs_samples[p1,:]), np.max(gibbs_samples[p1,:]), 1000)
    x2 = np.linspace(np.min(gibbs_samples[p2,:]), np.max(gibbs_samples[p2,:]), 1000)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.dstack((X1, X2))
    
    ax = fig.add_subplot(gs[1, 0])
    ax.hist2d(gibbs_samples[p1,:], gibbs_samples[p2,:], bins=20, density=True, alpha=0.5)
    ax.contour(X1, X2, multivariate_normal.pdf(X, mu, cov))
    ax.plot(x1, x1, linestyle='--', color='k', label='$w_{Nadal}=w_{Federer}$')
    ax.legend()
    
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histx.hist(gibbs_samples[p1,:], bins=20, alpha=0.5, density=True, color='r')
    ax_histx.plot(x1, norm.pdf(x1, mu[0], np.sqrt(cov[0,0])), color='r')
    ax_histx.axis('off')
    
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_histy.hist(gibbs_samples[p2,:], bins=20, alpha=0.5, density=True, color='b', orientation="horizontal")
    ax_histy.plot(norm.pdf(x2, mu[1], np.sqrt(cov[1,1])), x2, color='b')
    ax_histy.axis('off')

    ax.set_xlabel(f'{p1_name} Skill')
    ax.set_ylabel(f'{p2_name} Skill')
    ax.set_aspect('equal', 'box')
    
    fig.savefig('cw2/figures/D/joint.pdf')
    
    # Samples
    s_samples = gibbs_samples[p1,:] - gibbs_samples[p2,:]
    P_s = np.mean(s_samples > 0)
    
    print(f'Samples P(Nadal > Federer) = {P_s}')
    
    # Joint Ranking
    player_names = ['Novak Djokovic', 'Rafael Nadal', 'Roger Federer', 'Andy Murray']
    players = np.array([np.where(W == name) for name in player_names])[:,0,0]
    P_s = np.zeros((len(players), len(players)))
    
    for i, p1 in enumerate(players):
        for j, p2 in enumerate(players):
            P_s[i,j] = np.mean(gibbs_samples[p1,:] > gibbs_samples[p2,:])
        
    print(player_names)
    print('P(s>0):')
    print(P_s.round(2))

if __name__ == '__main__':
    main(*load_all())
    plt.show()
