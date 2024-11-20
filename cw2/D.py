from utils import load_data
import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt

def main():
    W, G, M, N = load_data()
    
    cache_file = 'cw2/cache/gibbs_samples.npy'
    gibbs_samples = np.load(cache_file)
    gibbs_samples = gibbs_samples[:,10:]
    
    p1_name = 'Rafael-Nadal'
    p2_name = 'Roger-Federer'
    
    p1 = np.where(W == p1_name)[0][0]
    p2 = np.where(W == p2_name)[0][0]
    
    # Marginals
    p1_mu = np.mean(gibbs_samples[p1,:])
    p1_sigma2 = np.var(gibbs_samples[p1,:])
    p2_mu = np.mean(gibbs_samples[p2,:])
    p2_sigma2 = np.var(gibbs_samples[p2,:])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    x = np.linspace(np.min(gibbs_samples[(p1,p2),:]), np.max(gibbs_samples[(p1,p2),:]), 100)
    ax.hist(gibbs_samples[p1,:], bins=20, alpha=0.5, density=True, color='r')
    ax.hist(gibbs_samples[p2,:], bins=20, alpha=0.5, density=True, color='b')
    ax.plot(x, norm.pdf(x, p1_mu, np.sqrt(p1_sigma2)), color='r')
    ax.plot(x, norm.pdf(x, p2_mu, np.sqrt(p2_sigma2)), color='b')
    ax.legend([p1_name, p2_name])
    ax.grid(True)
    
    fig.savefig('cw2/figures/D/marginals.pdf')
    
    P_s = norm.cdf((p1_mu - p2_mu)/np.sqrt(p1_sigma2 + p2_sigma2))
    print(P_s)
    
    # Joint
    joint_mu = np.array([p1_mu, p2_mu])
    joint_sigma2 = np.cov(gibbs_samples[np.array([p1, p2]),:])
    
    print(joint_sigma2)
    
    mu = joint_mu[0] - joint_mu[1]
    sigma2 = joint_sigma2[0,0] + joint_sigma2[1,1] - 2*joint_sigma2[0,1]
    
    P_s = norm.cdf(mu/np.sqrt(sigma2))
    print(P_s)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    x1, x2 = np.meshgrid(np.linspace(np.min(gibbs_samples[p1,:]), np.max(gibbs_samples[p1,:]), 100), np.linspace(np.min(gibbs_samples[p2,:]), np.max(gibbs_samples[p2,:]), 100))
    x = np.dstack((x1, x2))
    ax.hist2d(gibbs_samples[p1,:], gibbs_samples[p2,:], bins=20, density=True, alpha=0.5, label='Sample Histogram')
    ax.contour(x1, x2, multivariate_normal.pdf(x, joint_mu, joint_sigma2), label='Gaussian Fit')
    ax.plot(x1[(0,-1),(0,-1)], x1[(0,-1),(0,-1)], linestyle='--', color='k', label='w_{Nadal}=w_{Federer}')
    ax.legend()
    
    ax.set_xlabel(p1_name)
    ax.set_ylabel(p2_name)
    
    ax.set_aspect('equal', 'box')
    
    fig.savefig('cw2/figures/D/joint.pdf')
    
    # Samples

    s_samples = gibbs_samples[p1,:] - gibbs_samples[p2,:]
    P_s = np.mean(s_samples > 0)
    
    print(P_s)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    
    x = np.linspace(np.min(s_samples), np.max(s_samples), 100)
    ax.hist(s_samples, bins=20, alpha=0.5, density=True)
    ax.plot(x, norm.pdf(x, mu, np.sqrt(sigma2)))
    print(p1_mu, p2_mu)
    
    plt.show()
    

    
if __name__ == '__main__':
    main()