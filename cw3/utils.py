import os

import numpy as np
from scipy.io import loadmat


def load_data():
    # load data
    data = loadmat('cw3/data/kos_doc_data.mat')
    
    A = data['A']
    B = data['B']
    V = np.array([p[0].replace('-', ' ') for p in data['V'][:,0]])
    
    return A, B, V

# def generate_gibbs_samples(W, G, M, N, cache_file, num_iters):    
#     # perform gibbs sampling, skill samples is an num_players x num_samples array
#     skill_samples = gibbs_sample(G, M, num_iters)
#     np.save(cache_file, skill_samples)
    
# def generate_ep_samples(W, G, M, N, cache_file, num_iters):
#     # perform EP sampling, skill samples is an num_players x num_samples array
#     skill_samples = eprank(G, M, num_iters)
#     np.save(cache_file, skill_samples)
    
def load_all(use_cache=True):
    np.random.seed(0)
    
    A, B, V = load_data()
    
    # gibbs_cache = 'cw2/cache/gibbs_samples.npy'
    # ep_cache = 'cw2/cache/ep_samples.npy'
    
    # gibbs_iterations = 10000
    # ep_iterations = 1000
    
    # if use_cache:
    #     if not os.path.exists(gibbs_cache):
    #         generate_gibbs_samples(W, G, M, N, gibbs_cache, gibbs_iterations)
        
    #     if not os.path.exists(ep_cache):
    #         generate_ep_samples(W, G, M, N, ep_cache, ep_iterations)
    # else:
    #     generate_gibbs_samples(W, G, M, N, gibbs_cache, gibbs_iterations)
    #     generate_ep_samples(W, G, M, N, ep_cache, ep_iterations)
    
    # gibbs_samples = np.load(gibbs_cache)
    # mean_player_skills, precision_player_skills = np.load(ep_cache)
    
    return A, B, V