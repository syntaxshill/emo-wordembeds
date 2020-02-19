import torch
from langdetect import detect
import numpy as np
from gensim.models import KeyedVectors

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOPK = 50000  # use only the top k most frequent words
N_NEIGHBORS = 50  # consider the nearest n neighbors

print("Running...")

# load word embeddings
embeds = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

print("Loaded embeddings")

# load word classes
with open("data/stress_cues.txt", "r+", encoding="utf-8") as f:
    stress_words = [l.strip() for l in f.readlines()]

def class_of_word(word):
    return 1 if word in stress_words else 0

# TODO: needed? good?? this cuts down the vocabulary to, like, useful words
good_vocab = []
for key in embeds.vocab.keys():
    try:
        # python evaluates if conditions lazily so hopefully langdetect will not make this take FOREVER
        if key == key.lower() and key.isalpha() and detect(key) == 'en':
            good_vocab.append(key)
    except:
        continue
# good_vocab = [key for key in embeds.vocab.keys() if key == key.lower() and key.isalpha() and detect(key) == 'en']

print("Cut down vocab to " + str(len(good_vocab)) + " words")

# put the good vectors together in a matrix
X = torch.zeros(len(good_vocab), embeds.vectors.shape[1]).to(DEVICE)

for i, k in enumerate(good_vocab):
    X[i] = torch.tensor(embeds[k])

# get our basic embedding matrix
# X = torch.tensor(embeds.vectors)[:TOPK].to(DEVICE)

# calculate XX^T, which becomes a symmetric matrix of "closeness" scores
M = torch.matmul(X, X.t())

print("Calculated XX^T")

# calculate score(X)
scores = np.zeros(M.shape[0])

for xi in range(M.shape[0]):
    # get n_neighbors nearest neighbors of xi
    neighbors = np.argsort(M[xi].cpu().numpy())[-N_NEIGHBORS:]
    # calculate score
    # now neighbors is a list of indices of xi's nearest neighbors
    scores[xi] = sum([1 for neighbor in neighbors if class_of_word(good_vocab[xi]) == class_of_word(good_vocab[neighbor])])

print("Got scores!")

print("Average score of stress words")
print(mean([score for i, score in enumerate(scores) if class_of_word(i) == 1]))
print("Average score of all other words")
print(mean([score for i, score in enumerate(scores) if class_of_word(i) == 0]))
