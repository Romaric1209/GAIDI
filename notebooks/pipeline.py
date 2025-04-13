import pandas as pd
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.utils.validation import check_is_fitted
from text_transformers import (
    InputHandler, TextPreprocessor, SpacyPreprocessor, SpacyTfidfVectorizer, 
    Redundance, SentimentAnalyzer, ReadingEase, GunningFog
)

### Pipeline
TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "max_features": 15000,
    "sublinear_tf": True,
    # "max_df": 0.9,
    # "min_df": 0.001
}


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

X = pd.read_csv(r"combined_file.csv")
pipeline.fit(X)

check_is_fitted(pipeline.named_steps['feature_union'])

joblib.dump(pipeline, "pipeline.joblib", compress=3)