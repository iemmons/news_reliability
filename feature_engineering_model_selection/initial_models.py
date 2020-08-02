"""
Just a convenience function to evaluate the model score for a wide range of classification models.
Used for model selection.
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack


def initial_model_evaluation(x: list, y: np.ndarray, vector_method, svd: int = None):
    model_map = {
        'log': LogisticRegression(penalty='none', solver='saga'),
        'logl1': LogisticRegression(penalty='l1', solver='liblinear'),
        'logl2': LogisticRegression(),
        'neural_network': MLPClassifier(),
        'random_forest': RandomForestClassifier(),
        'knn': KNeighborsClassifier(n_neighbors=20, p=1, algorithm='brute')
    }
    vector_method_map = {
        'count': CountVectorizer(stop_words='english'),
        'tfidf': TfidfVectorizer(stop_words='english')
    }

    vectors = []
    for d in x:
        vectorizer = vector_method_map[vector_method]
        vectors.append(vectorizer.fit_transform(d))
    vector = hstack(vectors)

    if svd is not None:
        svd_model = TruncatedSVD(svd)
        vector = svd_model.fit_transform(vector)

        scl = StandardScaler()
        vector = scl.fit_transform(vector)

    for k, v in model_map.items():
        print(f'Cross-validation accuracy for {k} = {np.mean(cross_val_score(v, vector, y))}')


def bayes_evaluation(x: list, y: np.ndarray):
    vectors = []
    for d in x:
        vectorizer = CountVectorizer(stop_words='english')
        vectors.append(vectorizer.fit_transform(d))
    x = hstack(vectors)
    print(f'Cross-validation accuracy for naive_bayes = {np.mean(cross_val_score(MultinomialNB(), x, y))}')
