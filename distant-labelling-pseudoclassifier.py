from classify_lsa import get_wordembeds
from classify_lsa import get_model_vector
import nltk
import os
from scipy.spatial.distance import cosine
import numpy as np

def data_preprocessing(directory):
    docs = []
    for filename in os.listdir(directory):
        with open(directory+'/'+filename, "rb") as f:
            text = f.read()
        sentences = text.decode().split('.')
        trim = " ".join(sentences[: 5 if (len(sentences) > 5) else len(sentences)])
        docs.append(" ".join(nltk.word_tokenize(trim)))

    return docs


def compute_distances(X, E_s):
    dists = []
    for x in X:
        dists.append(cosine(x, E_s))
    return dists

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

if __name__ == "__main__":
    docs = data_preprocessing("./reddit/almosthomeless")
    # docs = data_preprocessing(["almosthomeless", "food_pantry", "homeless"])
    X = get_wordembeds(docs, False)
    E_s = get_model_vector("financial")
    dists = compute_distances(X, E_s)

    print("Average distance: ", np.mean(dists))

    print("*******************************************************************************************" +
          "*******************************************************************************************")
    print("DISTANT POINTS")
    print("*******************************************************************************************" +
          "*******************************************************************************************")
    count = 0
    for dist in dists:
        if dist > .2:
            print(str(dist) + ": " + docs[count])
            print("==================================================================================================")
        count += 1

    print("*******************************************************************************************" +
          "*******************************************************************************************")
    print("CLOSE POINTS")
    print("*******************************************************************************************" +
          "*******************************************************************************************")
    count = 0
    for dist in dists:
        if dist < .025:
            print(str(dist) + ": " + docs[count])
            print("==================================================================================================")
        count += 1


    docs = data_preprocessing("./reddit/anxiety")
    X = get_wordembeds(docs, False)
    E_s = get_model_vector("anxiety")
    dists = compute_distances(X, E_s)

    print("Average distance: ", np.mean(dists))

    print("*******************************************************************************************" +
          "*******************************************************************************************")
    print("DISTANT POINTS")
    print("*******************************************************************************************" +
          "*******************************************************************************************")
    count = 0
    for dist in dists:
        if dist > .2:
            print(str(dist) + ": " + str(docs[count].encode("utf-8")))
            print("==================================================================================================")
        count += 1

    print("*******************************************************************************************" +
          "*******************************************************************************************")
    print("CLOSE POINTS")
    print("*******************************************************************************************" +
          "*******************************************************************************************")
    count = 0
    for dist in dists:
        if dist < .025:
            print(str(dist) + ": " + str(docs[count].encode("utf-8")))
            print("==================================================================================================")
        count += 1