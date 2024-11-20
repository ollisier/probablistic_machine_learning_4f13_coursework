import numpy as np
from scipy.stats import norm

from utils import load_all


def main(W, G, M, N, gibbs_samples, mean_player_skills, precision_player_skills):
    mean_player_skills = mean_player_skills[:,-1]
    precision_player_skills = precision_player_skills[:,-1]
    
    player_names = ['Novak Djokovic', 'Rafael Nadal', 'Roger Federer', 'Andy Murray']
    players = np.array([np.where(W == name) for name in player_names])[:,0,0]
    
    skill_mu = mean_player_skills[players]
    skill_sigma2 = 1/precision_player_skills[players]
    
    s_mu = skill_mu[:,np.newaxis] - skill_mu[np.newaxis,:]
    s_sigma2 = skill_sigma2[:,np.newaxis] + skill_sigma2[np.newaxis,:]
    
    t_mu = s_mu
    t_sigma2 = s_sigma2 + 1
    
    P_s = norm.cdf(s_mu/np.sqrt(s_sigma2))
    P_t = norm.cdf(t_mu/np.sqrt(t_sigma2))
    
    print(player_names)
    print('P(s>0):')
    print(P_s.round(2))
    print('P(t>0):')
    print(P_t.round(2))
    
    
if __name__ == '__main__':
    main(*load_all())    