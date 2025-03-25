from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd



class InputHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
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
    def fit(self,X,y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return ["word_count"]

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X["text"]
        word_c = X.apply(word_count)
        return pd.DataFrame({"word_count": word_c})

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return ["preprocessed"]

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X["text"]
        cleaned = X.apply(basic_cleaning)
        return pd.DataFrame({"preprocessed": cleaned})

class ConsDensity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return ["cons_density"]

    def transform(self, X):
        return X["preprocessed"].apply(cons_density).values.reshape(-1, 1)

class Stress(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return ["stress_value"]

    def transform(self, X):
        return X["preprocessed"].apply(get_sentence_stress).values.reshape(-1, 1)

class Sentiment(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return ["sentiment_score"]

    def transform(self, X):
        return X["preprocessed"].apply(sentiment_polarity).values.reshape(-1, 1)

class Redundance(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return ["redundance"]

    def transform(self, X):
        return X["preprocessed"].apply(redundance).values.reshape(-1, 1)

class UnusualWord(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return ["unusual_words"]

    def transform(self, X):
        return X["preprocessed"].apply(word_choice).values.reshape(-1, 1)

class Coherence(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return ["coherence"]

    def transform(self, X):
        return X["preprocessed"].apply(coherence).values.reshape(-1, 1)

class ReadingEase(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return ["reading_ease"]

    def transform(self, X):
        return X["text"].apply(reading_ease).values.reshape(-1, 1)

class GunningFog(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return ["gunning_fog"]

    def transform(self, X):
        return X["text"].apply(gunning_fog).values.reshape(-1, 1)

class LogTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)
