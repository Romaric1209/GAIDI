import pandas as pd
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.utils.validation import check_is_fitted
from notebooks.transformers import (
    InputHandler, TextPreprocessor, ConsDensity, Stress, Sentiment, Redundance,
    UnusualWord,HowManyWords, Coherence, ReadingEase, GunningFog, LogTransform,
    Tfidf_Vectorizer
)

### Pipeline


log_scaler = LogTransform()

pipeline = Pipeline([
    ("input_handler", InputHandler()),
    ("union", FeatureUnion([
        ("preprocessed_features", Pipeline([
            ("preprocessor", TextPreprocessor()),
            ("features", FeatureUnion([
                ("cons_density", Pipeline([
                    ("extract", ConsDensity()),
                    ("log_scaling", log_scaler)
                ])),
                ("stress_value", Pipeline([
                    ("extract", Stress()),
                    ("log_scaling", log_scaler)
                ])),
                ("sentiment_score", Pipeline([
                    ("extract", Sentiment()),
                    ("log_scaling", log_scaler)
                ])),
                ("redundance", Pipeline([
                    ("extract", Redundance()),
                    ("log_scaling", log_scaler)
                ])),
                ("unusualword", Pipeline([
                    ("extract", UnusualWord()),
                    ("log_scaling", log_scaler)
                ])),
                ("coherence", Pipeline([
                    ("extract", Coherence()),
                    ("log_scaling", log_scaler)
                ])),
                ("tfidf",  Pipeline([
                    ("vectorizer",Tfidf_Vectorizer(ngram_range=(2, 3), max_features=10000)),
                    ("scaler",MaxAbsScaler())
                ]))
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

X = pd.read_csv(r"data/texts_data/5k_sampled_dataset.csv")
pipeline.fit(X)

# Check TF-IDF
preprocessed_pipeline = pipeline.named_steps['union'].transformer_list[0][1]  # preprocessed_features
preprocessed_features = preprocessed_pipeline.named_steps['features']
tfidf_transformer = preprocessed_features.named_transformers['tfidf'].named_steps['vectorizer']

try:
    check_is_fitted(tfidf_transformer.tfidf)
    print("✅ TF-IDF is fitted! Vocabulary size:", len(tfidf_transformer.tfidf.vocabulary_))
except Exception as e:
    print(f"❌ TF-IDF fitting error: {e}")

# Check WordCount
original_pipeline = pipeline.named_steps['union'].transformer_list[1][1]  # original_text_features
original_features = original_pipeline.named_steps['features']
wordcount_transformer = original_features.named_transformers['wordcount'].named_steps['extract']

try:
    check_is_fitted(wordcount_transformer)
    print("✅ WordCount is fitted!")
except Exception as e:
    print(f"❌ WordCount fitting error: {e}")


joblib.dump(pipeline, "notebooks/pipeline.joblib", compress=0)
# pipeline=joblib.load("notebooks/pipeline.joblib")
# sample_text = pd.DataFrame({"text": ["This is a test text"]})
# preprocessed_input = pipeline.transform(sample_text)
# print(preprocessed_input.shape)
