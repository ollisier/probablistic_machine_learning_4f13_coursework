from scipy.io import loadmat

def load_data():
    # load data
    data = loadmat('cw2/data/tennis_data.mat')
    # Array containing the names of each player
    W = data['W'][:,0]
    # loop over array to format more nicely
    for i, player in enumerate(W):
        W[i] = player[0]
    # Array of size num_games x 2. The first entry in each row is the winner of game i, the second is the loser
    G = data['G'] - 1
    # Number of players
    M = W.shape[0]
    # Number of Games
    N = G.shape[0]
    return W, G, M, N