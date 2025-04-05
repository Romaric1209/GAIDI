import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import isspmatrix, csr_matrix
from catboost import CatBoostClassifier
from notebooks.functions import (
    word_count, basic_cleaning, cons_density, get_sentence_stress, redundance,
    sentiment_polarity, word_choice, coherence, reading_ease, gunning_fog
)


class InputHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, str):
            X = [X]
        if isinstance(X, list):
            X = pd.DataFrame({"text": X})
        elif isinstance(X, pd.DataFrame):
            if "text" not in X.columns:
                raise ValueError("Input DataFrame must have a 'text' column")
        else:
            X = pd.DataFrame({"text": list(X)})
        return X

class HowManyWords(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def get_feature_names_out(self, input_features=None):
        return ["word_count"]

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X["text"]
        word_c = X.apply(word_count)
        return word_c.values.reshape(-1, 1)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X["text"]
        else:
            texts = X

        cleaned = texts.apply(basic_cleaning)
        return pd.DataFrame({"preprocessed": cleaned})

class ConsDensity(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X["preprocessed"]
        else:
            raise ValueError("Input must be a DataFrame with 'preprocessed' column")
        return texts.apply(cons_density).values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return ["cons_density"]

class Stress(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        stress_scores = X["preprocessed"].apply(get_sentence_stress)
        return stress_scores.values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return ["stress_value"]

class Sentiment(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X["preprocessed"]
        else:
            raise ValueError("Input must be a DataFrame with 'preprocessed' column")
        return texts.apply(sentiment_polarity).values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return ["sentiment_score"]


class Redundance(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X["preprocessed"]
        else:
            raise ValueError("Input must be a DataFrame with 'preprocessed' column")
        return texts.apply(redundance).values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return ["redundance"]


class UnusualWord(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X["preprocessed"]
        else:
            raise ValueError("Input must be a DataFrame with 'preprocessed' column")
        return texts.apply(word_choice).values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return ["unusual_words"]


class Coherence(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X["preprocessed"]
        else:
            raise ValueError("Input must be a DataFrame with 'preprocessed' column")
        return texts.apply(coherence).values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return ["coherence"]

class ReadingEase(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def get_feature_names_out(self, input_features=None):
        return ["reading_ease"]

    def transform(self, X):
        return X["text"].apply(reading_ease).values.reshape(-1, 1)

class GunningFog(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def get_feature_names_out(self, input_features=None):
        return ["gunning_fog"]

    def transform(self, X):
        return X["text"].apply(gunning_fog).values.reshape(-1, 1)

class LogTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return np.log1p(X)

class Tfidf_Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_range=(2, 3), max_features=10000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.max_features)
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.tfidf.fit(X["preprocessed"])
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return self.tfidf.transform(X["preprocessed"])

    def get_feature_names_out(self, input_features=None):
        return self.tfidf.get_feature_names_out()


class CatBoostSparseHandler(CatBoostClassifier):
    def fit(self, X, y=None, **kwargs):
        if isspmatrix(X):
            X = csr_matrix(X)
            X.data = np.copy(X.data)          
            X.indices = np.copy(X.indices)    
            X.indptr = np.copy(X.indptr)      
            X._has_canonical_format = True    
        return super().fit(X, y, **kwargs)
