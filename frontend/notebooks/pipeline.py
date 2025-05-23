from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from notebooks.text_transformers import *

### Pipeline
TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "max_features": 27000,
    "sublinear_tf": True,
    "max_df": 0.9,
    "min_df": 0.00025
}

def build_pipeline():
    pipeline = Pipeline([
        ("input_handler", InputHandler()),
        ("feature_union", FeatureUnion([
            ("spacy_features", Pipeline([
                ("spacy_preprocess", SpacyPreprocessor()),
                ("features", FeatureUnion([
                    ("tfidf", Pipeline([
                        ("vectorizer", SpacyTfidfVectorizer(**TFIDF_CONFIG)),
                        ("scaler", MaxAbsScaler())
                    ])),
                    ("redundance", Pipeline([
                        ("extract", Redundance()),
                        ("scaler", MinMaxScaler())
                    ]))
                ]))
            ])),
            ("sentiment_features", Pipeline([
                ("light_clean", TextPreprocessor()),
                ("sentiment", SentimentAnalyzer()),
                ("scaler", MinMaxScaler())
            ])),
            ("readability_features", FeatureUnion([
                ("reading_ease", Pipeline([
                    ("extract", ReadingEase()),
                    ("scaler", MinMaxScaler())
                ])),
                ("gunning_fog", Pipeline([
                    ("extract", GunningFog()),
                    ("scaler", MinMaxScaler())
                ]))
            ]))
        ]))
    ])
