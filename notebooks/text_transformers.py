import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import isspmatrix, csr_matrix
from catboost import CatBoostClassifier
from functions import nlp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from notebooks.functions import (
    basic_cleaning, spacy_preprocessor, redundance,
    reading_ease, gunning_fog
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


class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Expecting X to be a DataFrame with a "text" column or a Series of texts
        if isinstance(X, pd.DataFrame):
            texts = X["text"]
        else:
            texts = X
        return pd.DataFrame({"preprocessed": texts.apply(spacy_preprocessor)})


class SpacyTfidfVectorizer(BaseEstimator, TransformerMixin):
    """Custom TF-IDF with spaCy tokenization"""
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.spacy_tokenizer,
            **kwargs
        )
    
    def spacy_tokenizer(self, text):
        """Tokenize using spaCy for TF-IDF"""
        doc = nlp(text)
        return [token.lemma_.lower().strip() 
                for token in doc 
                if not token.is_stop 
                and not token.is_punct 
                and len(token.lemma_) > 2]
    
    def fit(self, X, y=None):
        # Extract the "preprocessed" column if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X["preprocessed"]
        self.vectorizer.fit(X)
        return self
    
    def transform(self, X):
        # Extract the "preprocessed" column if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X["preprocessed"]
        return self.vectorizer.transform(X)


class Redundance(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if "preprocessed" not in X.columns:
                raise ValueError("DataFrame must have 'preprocessed' column")
            texts = X["preprocessed"]
        elif isinstance(X, pd.Series):
            texts = X
        else:
            raise ValueError("Input must be a Series or DataFrame with 'preprocessed' column")

        scores = texts.apply(redundance)
        return scores.values.reshape(-1, 1)

  
class SentimentAnalyzer(BaseEstimator, TransformerMixin):
    """Modern sentiment analysis using VADER"""
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # If X is a DataFrame, use its first column
        if isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0]
        else:
            texts = X
        return [[self.analyzer.polarity_scores(text)['compound']] for text in texts]



class CatBoostSparseHandler(CatBoostClassifier):
    def fit(self, X, y=None, **kwargs):
        if isspmatrix(X):
            X = csr_matrix(X)
            X.data = np.copy(X.data)          
            X.indices = np.copy(X.indices)    
            X.indptr = np.copy(X.indptr)      
            X._has_canonical_format = True    
        return super().fit(X, y, **kwargs)


class ReadingEase(BaseEstimator, TransformerMixin):
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
        return texts.apply(reading_ease).values.reshape(-1, 1)


class GunningFog(BaseEstimator, TransformerMixin):
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
        return texts.apply(gunning_fog).values.reshape(-1, 1)