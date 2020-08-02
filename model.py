import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack


class ReliabilityClassifier:
    def __init__(self):
        self.text_tf = TfidfVectorizer(stop_words="english")
        self.title_tf = TfidfVectorizer(stop_words="english")
        self.sgd = TruncatedSVD(1000)
        self.scl = StandardScaler()
        self.model = LogisticRegression(penalty="elasticnet", solver="saga", C=0.05, l1_ratio=0.5, max_iter=1000)
        self.class_labels = {0: "Reliable", 1: "Unreliable"}

    def fit(self, train: pd.DataFrame):
        """
        :param train: dataframe with two predictor columns: text and title and a label column
                        0 - reliable
                        1 - unreliable
        """
        assert all(col in train.columns for col in ["text", "title", "label"]), \
            "Must include text, title, and label columns in training data."
        train = train[["text", "title", "label"]].dropna()  # remove extraneous columns
        text_vec = self.text_tf.fit_transform(train["text"])
        title_vec = self.title_tf.fit_transform(train["text"])
        vec = hstack([text_vec, title_vec])
        vec = self.sgd.fit_transform(vec)
        vec = self.scl.fit_transform(vec)
        self.model.fit(vec, train["label"])

    def predict(self, text: str, title: str):
        text_vec = self.text_tf.transform([text])
        title_vec = self.title_tf.transform([title])
        vec = hstack([text_vec, title_vec])
        vec = self.sgd.transform(vec)
        vec = self.scl.transform(vec)

        def _translate_output(prob):
            high_index = np.argmax(prob)
            return {"label": self.class_labels[self.model.classes_[high_index]],
                    "confidence": prob[high_index]}

        return _translate_output(self.model.predict_proba(vec)[0])
