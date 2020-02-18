from sklearn.manifold import TSNE
import numpy as np
import umap
import kmapper as km
import random
import torch as T

import seaborn as sns
import matplotlib.pyplot as plt

import pickle

# ------ For seaborn.
def init():
    sns.set()

def palette(p):
    sns.color_palette(p)

# ------ Miscellaneous helper functions.
def load(f, dim=300):
    lines = open(f, 'r').readlines()
    vectors = {}
    for l in lines:
        splits = l.split()
        if len(splits) - 1 != dim:
            print(l)
            print(len(splits))
            print()
            continue
        vectors[splits[0]] = np.array([float(x) for x in splits[1:]])

    return vectors

def save(x, f):
    pickle.dump(x, open(f, 'wb'))

def load_list(f):
    return [l.strip() for l in open(f, 'r').readlines()]

def stress_cues(vocab, f):
    words = []
    for line in open(f, 'r').readlines():
        word = line.strip()
        if word.endswith('*'):
            prefix = word.strip('*')
            for v in vocab:
                if v.startswith(prefix):
                    words.append(v)
        else:
            if word in vocab:
                words.append(word)

    return words

def compute_distances(vectors, cues=None, gpu=True):
    if cues is None:
        cues = vectors.keys()

    M = np.array([vectors[cue] for cue in cues])
    distances = []
    
    if gpu:
        M = T.from_numpy(M).cuda()
        for cue in cues:
            dists = M - T.tensor(vectors[cue]).cuda()
            dists = dists * dists
            dists = T.sum(dists, dim=1)
            dists = dists.cpu()
            dists = dists.numpy()
            distances.append(dists.flatten())
    else:
        for cue in cues:
            dists = M - vectors[cue]
            dists = dists * dists
            dists = np.sum(dists, axis=1)
            distances.append(dists.flatten())

    distances = np.array(distances)

    return distances, cues

def densities(X, Y, buckets):
    '''
        Collects scatterplot data into buckets and returns densities of each.
    '''
    locations = {}

    #if len(buckets) < 2: buckets = (buckets, buckets) # never mind, this doesn't work

    for x, y in zip(X, Y):
        x = int(x / buckets[0]) * buckets[0]
        y = int(y / buckets[1]) * buckets[1]
        if (x, y) not in locations: locations[(x, y)] = 0
        locations[(x, y)] += 1

    X_loc = []
    Y_loc = []
    densities = []

    for key in locations.keys():
        X_loc.append(key[0])
        Y_loc.append(key[1])
        densities.append(locations[key])

    return X_loc, Y_loc, densities

# ------ Wrappers for plots.
def histogram(x, labels,
              rotate=False,
              show=False, savefile=None):
    '''
        Wrapper for seaborn histogram.
    '''
    ax = sns.distplot(x)
    ax.set(xlabel=labels[0], ylabel=labels[1])
    if rotate: ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    if savefile: plt.savefig(savefile)
    if show: plt.show()
    plt.clf()

def histograms(xs, labels,
               rotate=False,
               show=False, savefile=None):
    '''
        Wrapper for seaborn histogram with multiple distributions.
    '''
    for x in xs:
        ax = sns.distplot(x)
    ax.set(xlabel=labels[0], ylabel=labels[1])
    if rotate: ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    if savefile: plt.savefig(savefile)
    if show: plt.show()
    plt.clf()

def violin(x, y, labels, legend=None,
           order=None, colors=None, xlim=None, ylim=None, rotate=False,
           show=False, savefile=None):
    '''
        Wrapper for seaborn violin plot.
    '''
    ax = sns.violinplot(x, y, order=order, hue=colors)
    ax.set(xlabel=labels[0], ylabel=labels[1])
    if xlim: ax.set(xlim=xlim)
    if ylim: ax.set(ylim=ylim)
    if rotate: ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    if savefile: plt.savefig(savefile)
    if show: plt.show()
    plt.clf()

def scatter(x, y, labels, legend=None,
            square=False, xlim=None, ylim=None, rotate=False,
            buckets=False, colors=None, size=1, alpha=.7,
            show=False, savefile=None):
    '''
        Wrapper for seaborn scatterplot. Optionally buckets data.
    '''
    if buckets:
        x, y, density = densities(x, y, buckets)
        density = np.log2(np.array(density))
        if colors is None:
            colors = density
        else:
            size = density
            
    ax = sns.scatterplot(x, y, hue=colors, size=size, marker='.', linewidth=0, alpha=alpha)
    ax.set(xlabel=labels[0], ylabel=labels[1])
    #if legend: ax._legend.set_title(legend)
    if xlim: ax.set(xlim=xlim)
    if ylim: ax.set(ylim=ylim)
    if square:
        plt.gca().set_aspect('equal', adjustable='box')
        if xlim is None: xlim = ax.get_xlim()
        if ylim is None: ylim = ax.get_ylim()
        lim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        ax.set(xlim=lim, ylim=lim)
    if rotate: ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    if savefile: plt.savefig(savefile)
    if show: plt.show()
    plt.clf()

# ------ Dimension reduction and TDA
def tsne(vectors, estimates, cues=None,
         lr=500, perplex=500,
         precompute=True, save_distances=True,
         show=False, save_prefix=None, save_points=None, verbose=True):
    '''
         Runs scikit-learn's t-SNE and plots with seaborn.
    '''
    if cues is None: cues = [cue for cue in estimates.keys()]

    if precompute:
        if verbose: print('[tsne] Computing distances...')
        distances, _ = compute_distances(vectors, cues)
        #if save_distances: save(distances, 'distances.pkl')
        if verbose: print('[tsne] Fitting t-SNE...')
        embedded = TSNE(n_components=2, metric='precomputed', learning_rate=lr, perplexity=perplex).fit_transform(distances)
    else:
        if verbose: print('[tsne] Fitting t-SNE...')
        M = np.array([vectors[cue] for cue in cues])
        embedded = TSNE(n_components=2).fit_transform(M)

    if verbose: print('[tsne] Preparing data for plotting...')
    x = []
    y = []
    for v in embedded:
        x.append(v[0])
        y.append(v[1])
    x = np.array(x)
    y = np.array(y)
    
    colors = [estimates[cue] for cue in cues]
    #sns.scatterplot(x, y, hue=colors, marker='.', size=1, linewidth=0, alpha=0.7)
    for opacity in [.7, .1, .01]:
        sns.scatterplot(x, y, hue=colors, marker='.', size=1, linewidth=0, alpha=opacity)
        if save_prefix: plt.savefig('plots/' + save_prefix + 'perplex-' + str(perplex) + '-lr-' + str(lr)+ '.png')
        plt.clf()
    #fig, ax = plt.subplots()
    #plt.scatter(x, y, c=colors, marker='.')
    ##cbar = plt.colorbar(ax.pcolor(colors))
    #if save_prefix: plt.savefig('plots/' + save_prefix + 'perplex-' + str(perplex) + '-lr-' + str(lr)+ '.png')
    if save_points: save((x, y, cues, colors), save_points)
    #plt.savefig('plots/test.png')
    if show: plt.show()

def umap_plot(vectors, estimates, cues=None,
              neighbors=15, min_dist=0.1, dim=2, metric='euclidean',
              show=False, save_prefix=None, save_points=None, verbose=True):
    '''
        Runs UMAP and plots with seaborn.
    '''
    reducer = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=min_dist,
        n_components=dim,
        metric=metric
        )
    if cues is None: cues = [cue for cue in estimates.keys()]
    M = np.array([vectors[cue] for cue in cues])

    if verbose: print('[umap] Fitting UMAP...')
    embedded = reducer.fit_transform(M)

    if verbose: print('[umap] Preparing data for plotting...')
    x = []
    y = []
    for v in embedded:
        x.append(v[0])
        y.append(v[1])
    x = np.array(x)
    y = np.array(y)

    colors = [estimates[cue] for cue in cues]
    for opacity in [.7, .1, .01]:
        sns.scatterplot(x, y, hue=colors, marker='.', size=1, linewidth=0, alpha=opacity)
        if save_prefix: plt.savefig('plots/' + save_prefix + 'umap-' + str(neighbors) + '-' + str(opacity) + '-' + metric + '.png')
        plt.clf()
    if save_points: save((x, y, cues, colors), save_points)
    if show: plt.show()

def kmap(vectors, cues, save_prefix=None):
    '''
        Wrapper for KeplerMapper (https://kepler-mapper.scikit-tda.org).
    '''
    X = np.array([vectors[cue] for cue in cues])
    mapper = km.KeplerMapper(verbose=1)
    new_X = mapper.fit_transform(X, projection=[0, 1])
    graph = mapper.map(new_X, X, nr_cubes=10)
    mapper.visualize(graph, path_html="kmap_output.html" if not save_prefix else save_prefix + 'kmap.html',
                     title="kmap(n_samples=5000, noise=0.03, factor=0.3)")
    
# ------ Plot analysis tools
def explore_scatter(X, Y, words, colors=None,
                    sample=None,
                    xr=None, yr=None):
    '''
        Given a scatterplot and corresponding words, finds words within the
        given ranges on the x- and y-axes. If None, returns all values.
    '''
    matches = []
    if colors is None: colors = [None for x in X]
    
    for x, y, word, color in zip(X, Y, words, colors):
        if (xr is None or (x >= xr[0] and x <= xr[1])) and (yr is None or (y >= yr[0] and y <= yr[1])):
            if sample and random.random() > sample: continue
            if color: matches.append(word)
            else: matches.append((word, color))

    return matches

# ------ Exploratory visualizations
def plot_cosines(cues, vectors, sample=None):
    '''
        Plots a histogram of cosine distances between words. If a sample rate
        is specified, randomly samples from pairs of words with that rate.
    '''
    cosines = []
    
    if sample and sample < 1:
        for cue in cues:
            for other in cues:
                if random.random() < sample:
                    pass # TODO
    
# initialize once
init()

if __name__ == "__main__":
    vectors = load('data/domain_w2v.txt')
    vocab = [k for k in vectors.keys()]
    cues = stress_cues(vocab, 'data/stress_union.txt')
    with open('data/stress_cues.txt', 'w') as out:
        for cue in cues:
            out.write(cue + '\n')
    estimates = {x:1 if x in cues else 0 for x in vocab}
    cues = list(set(cues + [vocab[random.randint(0, len(vocab) - 1)] for i in range(1000)]))

    #umap_plot(vectors, estimates, cues=cues, save_prefix='test')
    tsne(vectors, estimates, cues=cues, perplex=500, lr=500, save_prefix='test-tsne-', save_points='test')
    
    #umap_plot(vectors, estimates, neighbors=15, metric='euclidean', cues=cues, save_prefix='umap/fasttext-en-', save_points='scatter-umap.pkl')
    #kmap(vectors, cues, save_prefix='fasttext-en-')

    #X, Y, cues, colors = load('scatter-umap.pkl')
    #window = explore_scatter(X, Y, cues, colors, xr=[-5, -2], yr=[-4, -1])
    #for w in window:
    #    print(w)
    #central = []
    #for x, y, cue in zip(X, Y, cues):
    #    if x > -2.5 and x < 6 and y > 0 and y < 5:
    #        central.append(cue)
    #    elif x >= 0 and x < 5 and y > -4 and y < 5:
    #        central.append(cue)
    #    elif x > -2 and x < 5 and y > -2.5 and y < 0:
    #        central.append(cue)
    #umap_plot(vectors, estimates, cues=central, save_prefix='umap/central-en-', save_points='scatter-central.pkl')
    #exit(0)
