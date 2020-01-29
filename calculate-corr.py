import sys
import numpy as np
import pickle
import nltk
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from bert_embedding import BertEmbedding
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # TO TURN GLOVE TO KEYED VECTORS
# from gensim.scripts.glove2word2vec import glove2word2vec
# glove2word2vec(glove_input_file="glove.840B.300d.txt", word2vec_output_file="gensim_glove_vectors_large.txt") #only need to run this once

LABELED_DATA_FN = "train_mk2.pkl"
TEST_DATA_FN = "test_mk2.pkl"
# model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", limit=500000, binary=True)
# model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", limit=500000, binary=False)
# model = KeyedVectors.load_word2vec_format("gensim_glove_vectors_large.txt", limit=500000, binary=False)
# model = KeyedVectors.load_word2vec_format("gensim_twitter_200D.txt", limit=500000, binary=False)
model = KeyedVectors.load_word2vec_format("deps.words/w2v_deps.txt", limit=500000, binary=False)
# model = gensim.models.KeyedVectors.load("word2vec_model_300d.model")

with open(LABELED_DATA_FN, "rb") as f:
    train_data = pickle.load(f)

with open(TEST_DATA_FN, "rb") as f:
    test_data = pickle.load(f)

vectorizer = TfidfVectorizer()
vectorizer.fit(" ".join(d['text']) for d in train_data)

tfidf_weighing = False
# distances = True
domain_means = False
variance = False
plot = True
useBert = True


sub2domain = {"almosthomeless": "financial",
              "anxiety": "anxiety",
              "assistance": "financial",
              "domesticviolence": "abuse",
              "food_pantry": "financial",
              "homeless": "financial",
              "ptsd": "ptsd",
              "relationships": "social",
              "stress": "anxiety",
              "survivorsofabuse": "abuse"}


def get_model_vector(domain):
    docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
            if sub2domain.get(d['subreddit']) == domain and d['label'] == 1]
    return get_wordembeds(docs, True)


# returns model vector formed from word embeddings of docs if mean=True
# else returns word embeddings formed from docs
def get_wordembeds(docs, mean):
    # form document embeddings by averaging embeddings for each word
    doc_vectors = []
    count = 0
    if not useBert:
        tfidf_dict = vectorizer.vocabulary_
        tfidf = vectorizer.transform(docs)
        for doc in docs:
            word_embeds = []
            for word in doc.split(' '):
                if word in model:
                    if tfidf_weighing:
                        if word in tfidf_dict:
                            row = tfidf[count].todense()
                            tfidf_weight = row.item(0, tfidf_dict.get(word))
                            word_embeds.append(model[word]*tfidf_weight)
                    else:
                        word_embeds.append(model[word])
            if len(word_embeds) != 0:
                doc_vectors.append(np.asarray(word_embeds).mean(0))
            count += 1
    else:
        bert_embedding = BertEmbedding()
        for doc in docs:
            embeds = bert_embedding(doc)
            vecs = [x[1][0] for x in embeds if len(x[1]) != 0]
            arr = np.vstack(vecs)
            doc_vectors.append(np.asmatrix(arr).mean(0))


    if mean == True:
        return np.asmatrix(doc_vectors).mean(0)
    else:
        return np.asmatrix(doc_vectors)


def compute_cosine_similarity(test_data, E_s):
    docs = [" ".join(d['text']) for d in test_data]
    y = [d['label'] for d in test_data]
    s_similarities = []
    wordembeds = get_wordembeds(docs, False)
    for row in wordembeds:
        s_similarities.append(cosine_similarity(E_s, row)[0][0])

    return np.corrcoef(y, s_similarities)[1, 0]



def vsm_corr_wordembeds():

    print("Computing model emotion class vectors on train data")

    if domain_means == True:
        social_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                       if sub2domain.get(d['subreddit']) == "social"]
        financial_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                          if sub2domain.get(d['subreddit']) == "financial"]
        anxiety_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                        if sub2domain.get(d['subreddit']) == "anxiety"]
        abuse_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                      if sub2domain.get(d['subreddit']) == "abuse"]
        ptsd_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                     if sub2domain.get(d['subreddit']) == "ptsd"]

        social_s_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                        if sub2domain.get(d['subreddit']) == "social" and d['label'] == 1]
        financial_s_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                        if sub2domain.get(d['subreddit']) == "financial" and d['label'] == 1]
        anxiety_s_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                        if sub2domain.get(d['subreddit']) == "anxiety" and d['label'] == 1]
        abuse_s_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                        if sub2domain.get(d['subreddit']) == "abuse" and d['label'] == 1]
        ptsd_s_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                        if sub2domain.get(d['subreddit']) == "ptsd" and d['label'] == 1]

        social_ns_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                        if sub2domain.get(d['subreddit']) == "social" and d['label'] == 0]
        financial_ns_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                        if sub2domain.get(d['subreddit']) == "financial" and d['label'] == 0]
        anxiety_ns_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                        if sub2domain.get(d['subreddit']) == "anxiety" and d['label'] == 0]
        abuse_ns_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                        if sub2domain.get(d['subreddit']) == "abuse" and d['label'] == 0]
        ptsd_ns_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data
                        if sub2domain.get(d['subreddit']) == "ptsd" and d['label'] == 0]

        if variance == True:
            social = get_wordembeds(social_docs, False)
            financial = get_wordembeds(financial_docs, False)
            anxiety = get_wordembeds(anxiety_docs, False)
            abuse = get_wordembeds(abuse_docs, False)
            ptsd = get_wordembeds(ptsd_docs, False)

            print("Variance of all test vectors (social): ", np.var(social))
            print("Variance of all test vectors (financial): ", np.var(financial))
            print("Variance of all test vectors (anxiety): ", np.var(anxiety))
            print("Variance of all test vectors (abuse): ", np.var(abuse))
            print("Variance of all test vectors (ptsd): ", np.var(ptsd))

            social_s = get_wordembeds(social_s_docs, False)
            financial_s = get_wordembeds(financial_s_docs, False)
            anxiety_s = get_wordembeds(anxiety_s_docs, False)
            abuse_s = get_wordembeds(abuse_s_docs, False)
            ptsd_s = get_wordembeds(ptsd_s_docs, False)

            print("Variance of stressed test vectors (social): ", np.var(social_s))
            print("Variance of stressed test vectors (financial): ", np.var(financial_s))
            print("Variance of stressed test vectors (anxiety): ", np.var(anxiety_s))
            print("Variance of stressed test vectors (abuse): ", np.var(abuse_s))
            print("Variance of stressed test vectors (ptsd): ", np.var(ptsd_s))

            social_ns = get_wordembeds(social_ns_docs, False)
            financial_ns = get_wordembeds(financial_ns_docs, False)
            anxiety_ns = get_wordembeds(anxiety_ns_docs, False)
            abuse_ns = get_wordembeds(abuse_ns_docs, False)
            ptsd_ns = get_wordembeds(ptsd_ns_docs, False)

            print("Variance of unstressed test vectors (social): ", np.var(social_ns))
            print("Variance of unstressed test vectors (financial): ", np.var(financial_ns))
            print("Variance of unstressed test vectors (anxiety): ", np.var(anxiety_ns))
            print("Variance of unstressed test vectors (abuse): ", np.var(abuse_ns))
            print("Variance of unstressed test vectors (ptsd): ", np.var(ptsd_ns))

            if plot == True:
                docs_we = get_wordembeds([" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data], False)
                pca = PCA(n_components=2)
                pca.fit(docs_we)
                X_soc = pca.transform(social)
                X_fin = pca.transform(financial)
                X_anx = pca.transform(anxiety)
                X_abu = pca.transform(abuse)
                X_pts = pca.transform(ptsd)

                plt.figure()
                ax = plt.axes(projection='3d')
                lw = 2
                plt.scatter(X_soc[:, 0], X_soc[:, 1], color='navy', alpha=.8, lw=lw, label='social')
                plt.scatter(X_fin[:, 0], X_fin[:, 1], color='turquoise', alpha=.8, lw=lw, label='financial')
                plt.scatter(X_anx[:, 0], X_anx[:, 1], color='darkorange', alpha=.8, lw=lw, label='anxiety')
                plt.scatter(X_abu[:, 0], X_abu[:, 1], color='green', alpha=.8, lw=lw, label='abuse')
                plt.scatter(X_pts[:, 0], X_pts[:, 1], color='purple', alpha=.8, lw=lw, label='ptsd')
                plt.legend(loc='best', shadow=False, scatterpoints=1)
                plt.title('PDA of word embeddings')

                plt.show()



        else:
            E_social_s = get_wordembeds(social_s_docs, True)
            E_financial_s = get_wordembeds(financial_s_docs, True)
            E_anxiety_s = get_wordembeds(anxiety_s_docs, True)
            E_abuse_s = get_wordembeds(abuse_s_docs, True)
            E_ptsd_s = get_wordembeds(ptsd_s_docs, True)

            test_social = [d for d in test_data if sub2domain.get(d['subreddit']) == "social"]
            test_financial = [d for d in test_data if sub2domain.get(d['subreddit']) == "financial"]
            test_anxiety = [d for d in test_data if sub2domain.get(d['subreddit']) == "anxiety"]
            test_abuse = [d for d in test_data if sub2domain.get(d['subreddit']) == "abuse"]
            test_ptsd = [d for d in test_data if sub2domain.get(d['subreddit']) == "ptsd"]

            print("Computing cosine similarity on test data")
            print("social: ", compute_cosine_similarity(test_social, E_social_s))
            print("financial: ", compute_cosine_similarity(test_financial, E_financial_s))
            print("anxiety: ", compute_cosine_similarity(test_anxiety, E_anxiety_s))
            print("abuse: ", compute_cosine_similarity(test_abuse, E_abuse_s))
            print("ptsd: ", compute_cosine_similarity(test_ptsd, E_ptsd_s))



    else:
        docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data]
        notstress_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data if d['label'] == 0]
        stress_docs = [" ".join(nltk.word_tokenize(" ".join(d['text']))) for d in train_data if d['label'] == 1]

        if variance == True:
            docs_we = get_wordembeds(docs, False)
            print("Variance of all test vectors: ", np.var(docs_we))

            we_s = get_wordembeds(stress_docs, False)
            print("Variance of stressed test vectors: ", np.var(we_s))

            we_ns = get_wordembeds(notstress_docs, False)
            print("Variance of unstressed test vectors: ", np.var(we_ns))

            # https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html
            if plot == True:
                pca = PCA(n_components=3)
                pca.fit(docs_we)
                X_s = pca.transform(we_s)
                X_ns = pca.transform(we_ns)
                print("PCA explained variance: ", str(pca.explained_variance_))
                print("PCA explained variance ratio: ", str(pca.explained_variance_ratio_))

                plt.figure()
                ax = plt.axes(projection='3d')
                lw = 3
                ax.scatter3D(X_s[:, 0], X_s[:, 1], X_s[:, 2], color='navy', alpha=.8, lw=lw, label='stressed')
                ax.scatter3D(X_ns[:, 0], X_ns[:, 1], X_ns[:, 2], color='turquoise', alpha=.8, lw=lw, label='not stressed')
                ax.legend(loc='best', shadow=False, scatterpoints=1)
                # ax.title('PCA of word embeddings')

                # lda = LinearDiscriminantAnalysis(n_components=3)
                # y = [d['label'] for d in train_data]
                # X = lda.fit_transform(docs_we, y)
                # plt.figure()
                #
                # i = 0
                # X_s_lda, X_ns_lda = [], []
                # y_s_lda, y_ns_lda = [], []
                # for x in X:
                #     if y[i] == 0:
                #         X_ns_lda.append(x)
                #         y_ns_lda.append(y[i])
                #     else:
                #         X_s_lda.append(x)
                #         y_s_lda.append(y[i])
                #     i += 1
                #
                # plt.scatter(X_s_lda, y_s_lda, color='navy', alpha=.8, lw=lw, label='stressed')
                # plt.scatter(X_ns_lda, y_ns_lda, color='turquoise', alpha=.8, lw=lw, label='not stressed')
                # plt.legend(loc='best', shadow=False, scatterpoints=1)
                # plt.title('LDA of word embeddings')

                plt.show()



        else:
            E_s = get_wordembeds(stress_docs, True)
            print("Cosine similarity correlation (E_stress): ", compute_cosine_similarity(test_data, E_s))


# def vsm_tf_idf(data):
#     # get tf-idf
#     # vectorizer takes list, where each doc is a string
#     stress_docs = [" ".join(d['text']) for d in data if d['label'] == 1]
#     not_stress_docs = [" ".join(d['text']) for d in data if d['label'] == 0]
#     vectorizer = TfidfVectorizer()
#     vectorizer.fit(" ".join(d['text']) for d in data)
#     tf_idf_vecs_s = vectorizer.transform(stress_docs)
#     tf_idf_vecs_ns = vectorizer.transform(not_stress_docs)
#
#     # calculate model vectors E_stress and E_notstress
#     E_s = tf_idf_vecs_s.mean(0)    # row-wise mean
#     E_ns = tf_idf_vecs_ns.mean(0)
#
#     return E_s, E_ns, vectorizer
#
#
# def vsm_corr_tf_idf(variance):
#     with open(LABELED_DATA_FN, "rb") as f:
#         train_data = pickle.load(f)
#
#     with open(TEST_DATA_FN, "rb") as f:
#         test_data = pickle.load(f)
#
#     print("Computing model emotion class vectors on train data")
#     E_s, E_ns, vectorizer = vsm_tf_idf(train_data,variance)
#     y = [d['label'] for d in test_data]
#
#     print("Computing cosine similarity on test data")
#     s_similarities = []
#     ns_similarities = []
#     all_queries = []
#     for d in test_data:
#         doc = " ".join(d['text'])
#         query_s, query_ns = vectorizer.transform([doc]).todense(), vectorizer.transform([doc]).todense()
#         all_queries.append(query_s)
#         all_queries.append(query_ns)
#         s_similarities.append(cosine_similarity(E_s, query_s)[0][0])
#         ns_similarities.append(cosine_similarity(E_ns, query_ns)[0][0])
#
#     print("Cosine similarity correlation (E_stress):", np.corrcoef(y, s_similarities)[1, 0])
#     print("Cosine similarity correlation (E_notstress):", np.corrcoef(y, ns_similarities)[1, 0])


if __name__ == "__main__":
    vsm_corr_wordembeds()
