"""
    Second exercice de notre cinquième cours de machine learning.
    Le but était d'utiliser de créer un algorithme de recommandation
    pour indiquer quel article ressemble le plus à un certain article donnée.
"""

import os
import math
from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_similarity(vector1, vector2):
    """
        calculate the similarity between two vector
    """
    dot_product = vector2.multiply(vector1).sum()
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(vector2.power(2).sum())
    if not magnitude:
        return 0
    return dot_product/magnitude

def main():
    """
        ...
    """

    learn_path = "./data/SimpleText/SimpleText_auto/"
    dirs = os.listdir(learn_path)
    learn_data = [] 
    # This would print all the files and directories
    for file in dirs:
        with open(learn_path + file, encoding="utf8") as myfile:
            learn_data.append(myfile.read().replace('\n', ''))


    test_path = "./data/SimpleText/SimpleText_test/S0022314X13001777.txt"
    test_data = []
    with open(test_path, encoding="utf8") as myfile:
            test_data.append(myfile.read().replace('\n', ''))


    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(learn_data)

    #list of words
    #print(vectorizer.get_feature_names())

    #(number of documents, nb words in dico)
    #print(X.shape)

    skl_tfidf_comparisons = []

    for count_0, doc_0 in enumerate(X.toarray()):
        skl_tfidf_comparisons.append(
            (cosine_similarity(doc_0, vectorizer.transform(test_data)), count_0))

    for x in sorted(skl_tfidf_comparisons, reverse = True):
        print(x)

if __name__ == "__main__":
    main()
