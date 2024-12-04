import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

from utils import load_all

def main(A, B, V):
    M = V.shape[0]
    
    counts = np.zeros(M)
    
    for i in range(A.shape[0]):
        counts[A[i,1]-1] += A[i,2]
        
    N = np.sum(counts)
    
    ML_multinomial = counts / N
    
    top_indices = np.argsort(ML_multinomial)
    
    top_words = V[top_indices]
    top_probs = ML_multinomial[top_indices]
    
    print(f'{top_words[-1]} - {np.log(top_probs[-1])}')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_words[-20:], top_probs[-20:], color='blue')
    ax.set_xlabel('Words')
    ax.set_ylabel('Probability')
    ax.set_title('Top 10 Most Likely Words')
    fig.tight_layout()
    fig.savefig('cw3/figures/A/most_likely.png')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_words[:20], top_probs[:20], color='blue')
    ax.set_xlabel('Words')
    ax.set_ylabel('Probability')
    ax.set_title('Top 10 Most Likely Words')
    fig.tight_layout()
    fig.savefig('cw3/figures/A/least_likely.png')
    
    
if __name__ == '__main__':
    main(*load_all())
    plt.show()
