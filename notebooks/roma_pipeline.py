import pandas as pd
import numpy as np
import string
import joblib
import re
import nltk
nltk.data.path.append("/home/romaric/code/nghia95/fake-data-detector/notebooks/roma_NTLK_Data_Cache")
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from textblob import TextBlob
from gensim.models import LsiModel
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import textstat
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

cmu_dict = cmudict.dict()


### Define custom functions


def word_count(text):
    if not isinstance(text, str):
       text = str(text)
    return len(text.split())


def basic_cleaning(text):
    if not isinstance(text, str):  # Convert to string if it's not
       text = str(text)
    # Remove whitespace
    prepoc_text = text.strip()
    # Lowercasing
    prepoc_text = prepoc_text.lower()
    # remove digits
    prepoc_text = "".join(char for char in prepoc_text if not char.isdigit())
    # remove punctuation
    for punctuation in string.punctuation:
        prepoc_text = prepoc_text.replace(punctuation," ")
    # remove regex
    prepoc_text = re.sub('<[^<]+?',"",prepoc_text)

    return prepoc_text


def cons_density(text):

    consonnant = sum(1 for char in text if char.isalpha() and char not in "aeiou")
    vowel = sum(1 for char in text if char.isalpha() and char in "aeiou")
    total_letters = vowel + consonnant
    return round((consonnant/(vowel + consonnant)),3) if total_letters > 0 else 0


def get_word_stress(word):
    if word in cmu_dict:
        return sum(int(char) for syllable in cmu_dict[word][0] for char in syllable if char.isdigit())
    return 0

def get_sentence_stress(sentence):
    words = sentence.split()
    stress_values = [get_word_stress(word) for word in words]
    return sum(stress_values)


def redundance(text):
    # give a redundance score, considering the lenght of each text, if a lemmatized words appears more than three times the mean, it is considered redundant.

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean_tokens = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in clean_tokens]

    word_counts = Counter(lemmatized_tokens)
    mean_freq = sum(word_counts.values()) / len(word_counts) if len(word_counts)!= 0 else 0

    if mean_freq != 0:
        score = sum(1 for word, count in word_counts.items() if count > 2.5 * mean_freq)
    else:
        score = 0

    return score


def sentiment_polarity(text):
    sent_pol = TextBlob(text).sentiment.polarity
    return abs(round(sent_pol,3))


def word_choice(text):
    common_ai_words =["commendable",'transhumanist', 'meticulous', 'elevate','hello', 'tapestry','leverage',
                  'journey', 'headache','resonate','testament','explore', 'binary','delve',
                  'enrich', 'seamless','multifaceted', 'sorry','foster', 'convey', 'beacon',
                  'interplay', 'oh', 'navigate','form','adhere','cannot', 'landscape','remember',
                  'paramount', 'comprehensive', 'placeholder','grammar','real','summary','symphony',
                  'furthermore','relationship','ultimately','profound','art','supercharge','evolve',
                  'beyoud','reimagine','vibrant', 'robust','pivotal','certainly','quinoa','orchestrate','align',
                  'diverse','recommend','annals','note','employ','bustling','indeed','digital','enigma', 'outfit',
                  'indelible','refrain','culture','treat','emerge','meticulous','esteemed','weight','whimsical','bespoke',
                  'highlight','antagonist','unlock','key','breakdown','tailor','misinformation','treasure','paradigm','captivate',
                  'song','underscore','calculate','especially','climate','hedging','inclusive','exercise','ai','embrace',
                  'level','nuance','career','dynamic','accent','ethos','cheap','firstly','online','goodbye'
                  ]
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    word_count = 0
    for word in text.split():
        if word in common_ai_words:
            word_count += 1

    return word_count


def coherence(text):
    # uses gensim to measure coherence, use the lsi model(latent semantic indexing, coherence c_v because we provide the text)
    tokens = word_tokenize(text)
    if not tokens:
        coherence_score = 0
    else:
        dictionary = corpora.Dictionary([tokens])
        corpus_gensim = [dictionary.doc2bow(tokens)]
        lsa_model = LsiModel(corpus_gensim, id2word=dictionary)

        coherence_model = CoherenceModel(
            model=lsa_model,
            texts=[tokens],
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
    return coherence_score



def reading_ease(text):
    reading_ease= textstat.flesch_reading_ease(text)
    return reading_ease


def gunning_fog(text):
    gunning_fog = textstat.gunning_fog(text)
    return gunning_fog


### Custom Transformers


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

def log_transform(x):
    return np.log1p(x)


### Pipeline


log_scaler = FunctionTransformer(log_transform, validate=True)

pipeline = Pipeline([
    ("input_handler", InputHandler()),
    ("union", FeatureUnion([
        ("preprocessed_features", Pipeline([
            ("preprocessor", TextPreprocessor()),
            ("features", FeatureUnion([
                ("cons_density", ConsDensity()),
                ("stress_value", Pipeline([
                    ("extract", Stress()),
                    ("scaler", MinMaxScaler())
                ])),
                ("sentiment_score", Sentiment()),
                ("redundance", Pipeline([
                    ("extract", Redundance()),
                    ("log_scaling", log_scaler)
                ])),
                ("unusualword", Pipeline([
                    ("extract", UnusualWord()),
                    ("log_scaling", log_scaler)
                ])),
                ("coherence", Coherence())
            ]))
        ])),
        ("original_text_features", Pipeline([
            ("features", FeatureUnion([
                ("wordcount", Pipeline([
                    ("extract", HowManyWords()),
                    ("scaler", MinMaxScaler())
                ])),
                ("readingease", Pipeline([
                    ("extract", ReadingEase()),
                    ("scaler", MinMaxScaler())
                ])),
                ("gunningfog", Pipeline([
                    ("extract", GunningFog()),
                    ("scaler", MinMaxScaler())
                ]))
            ]))
        ]))
    ]))
])


feature_names = [
    "cons_density", "stress_value", "sentiment_score",
    "redundance", "unusual_words", "coherence",
    "word_count", "reading_ease", "gunning_fog"
]

# joblib.dump(pipeline, "roma_pipeline.joblib")
