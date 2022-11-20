from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score


def tf_idf_transform(
    ngram_range: Tuple[int],
    max_features: int,
    train_texts: np.ndarray,
    test_texts: np.ndarray,
) -> Tuple[csr_matrix]:
    vectorizer_params = {
        "ngram_range": ngram_range,
        "max_features": max_features,
        "tokenizer": lambda s: s.split(),
    }
    vectorizer = TfidfVectorizer(**vectorizer_params)
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    return X_train_tfidf, X_test_tfidf


def fit_and_score(
    C: float,
    X_train: csr_matrix,
    y_train: np.ndarray,
    X_val: csr_matrix,
    y_val: np.ndarray,
) -> Tuple[float]:
    clf = LogisticRegression(
        C=C,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        multi_class="multinomial",  # to solve the task like Softmax Classifier
        solver="sag",  # for faster training
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    return precision_score(y_val, y_pred, average="macro").round(3), recall_score(
        y_val, y_pred, average="macro"
    ).round(3)
