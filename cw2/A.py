import numpy as np
import matplotlib.pyplot as plt
from gibbsrank import gibbs_sample
from utils import load_data

# set seed for reproducibility
np.random.seed(0)
# load data
W, G, M, N = load_data()

# number of iterations
num_iters = 1100
# perform gibbs sampling, skill samples is an num_players x num_samples array
skill_samples = gibbs_sample(G, M, num_iters)

# Sample Plot
N_players = 3
players = np.random.choice(M, N_players)
fig, ax = plt.subplots(N_players, 1, figsize=(10, 8))

for i, player in enumerate(players):
    ax[i].plot(skill_samples[player, :200])
    ax[i].set_title(f'Skill Samples for Player {W[player]}')
    ax[i].set_xlabel('Iteration')
    ax[i].set_ylabel('Skill Level')
    ax[i].grid(True)

fig.tight_layout()
fig.savefig('cw2/figures/A/skill_samples.eps')

# Autocovariance Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
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
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(np.mean(skill_samples[:, :50], axis=0), label='Mean of skills')
ax[0].set_title('Mean of Skills Over Iterations')
ax[0].set_ylabel('Mean Skill Level')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(np.std(skill_samples[:, :50], axis=0), label='Standard deviation of skills')
ax[1].set_title('Standard Deviation of Skills Over Iterations')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Standard Deviation')
ax[1].legend()
ax[1].grid(True)

fig.tight_layout()
fig.savefig('cw2/figures/A/convergence.eps')

plt.show()
