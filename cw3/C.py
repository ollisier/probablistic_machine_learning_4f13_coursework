import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, gammaln

from utils import load_all, compute_word_counts

def main(A, B, V):
    k, M = compute_word_counts(A, V)
    N = np.sum(k)
    
    D = 100
    a = np.linspace(0.1, 2, D)
    log_marginal_likelihood = np.zeros(D)
    for i in range(D):
        alpha = np.ones(M) * a[i]
        
        alpha_prime = alpha + k
        
        log_marginal_likelihood[i] = betaln(alpha_prime) - betaln(alpha)
       
    idx_max = np.argmax(log_marginal_likelihood)
    a_max = a[idx_max]
    print(f'Max Likelihood a = {a_max}')
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(a, log_marginal_likelihood)
    ax.scatter(a_max, log_marginal_likelihood[idx_max], color='tab:red')
    ax.set_xlabel('$a$')
    ax.set_ylabel('$\log(p(\mathcal{A}))$')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig('cw3/figures/C/hyper_parameter_opt.pdf')
    
    alpha_prime = a_max + k
    
    pi_mean = alpha_prime / np.sum(alpha_prime)
    
    document_id = 2001
        
    A_doc = B[B[:,0] == document_id,:]
    doc_log_likelihood = document_log_likelihood(A_doc, V, pi_mean)
    
    print(f'Document {document_id} Log Likelihood = {doc_log_likelihood}')
    print(f'Document {document_id} perplexity = {perplexity(doc_log_likelihood, np.sum(A_doc[:,2]))}')
    
    test_log_likelihood = document_log_likelihood(B, V, pi_mean)
    print(f'Test Set {document_id} perplexity = {perplexity(test_log_likelihood, np.sum(B[:,2]))}')
    
    print(f'M = {M}')
    
def perplexity(likelihood, N):
    return np.exp(-likelihood / N)
    
def document_log_likelihood(A, V, pi):
    k_doc, _ = compute_word_counts(A, V)
    doc_log_likelihood = np.dot(k_doc, np.log(pi))
    return doc_log_likelihood

def betaln(x):
    return np.sum(gammaln(x)) - gammaln(np.sum(x))

def factorialln(x):
    return gammaln(x+1)
    
    
if __name__ == '__main__':
    main(*load_all())
    plt.show()
