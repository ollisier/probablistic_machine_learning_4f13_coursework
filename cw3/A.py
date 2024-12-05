import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

from utils import load_all, compute_word_counts

def main(A, B, V):
    k, M = compute_word_counts(A, V)
    N = np.sum(k)

    ML_multinomial = k / N
    
    top_indices = np.argsort(ML_multinomial)
    
    top_words = V[top_indices]
    top_probs = ML_multinomial[top_indices]
    
    print(f'{top_words[-1]} - {np.log(top_probs[-1])}')
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top_words[-20:], top_probs[-20:], color='blue')
    ax.set_ylabel('Words')
    ax.set_xlabel('Probability')
    fig.tight_layout()
    fig.savefig('cw3/figures/A/most_likely.pdf')
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top_words[:20], top_probs[:20], color='blue')
    ax.set_ylabel('Words')
    ax.set_xlabel('Probability')
    fig.tight_layout()
    fig.savefig('cw3/figures/A/least_likely.pdf')
    
    
if __name__ == '__main__':
    main(*load_all())
    plt.show()
