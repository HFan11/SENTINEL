import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import torch

class HybridFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, roberta_model, roberta_tokenizer, tfidf_vectorizer):
        self.roberta_model = roberta_model
        self.roberta_tokenizer = roberta_tokenizer
        self.tfidf_vectorizer = tfidf_vectorizer
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.tfidf_vectorizer.fit(X)
        return self

    def transform(self, X):
        # BERT embeddings
        encoded_input = self.roberta_tokenizer(X, return_tensors='pt', padding=True, truncation=True)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            features_bert = self.roberta_model(**encoded_input).pooler_output.cpu().numpy()

        # Normalize BERT embeddings
        features_bert = self.scaler.fit_transform(features_bert)

        # TF-IDF features
        features_tfidf = self.tfidf_vectorizer.transform(X)

        # Concatenate features
        features_combined = np.hstack((features_bert, features_tfidf.toarray()))
        return features_combined
