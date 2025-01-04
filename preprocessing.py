import re
import string
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='light'):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocess_text(text) for text in X]

    def preprocess_text(self, text):
        if self.strategy == "none":
            return text
        elif self.strategy == "light":
            text = text.encode("ascii", "ignore").decode('ascii')
            text = text.strip()
            text = text.strip("\"")
            for c in string.punctuation:
                text = text.replace(c, "")
            if text[-1] != ".":
                text = text.split(".")
                text = ".".join(text[:-1])
                text += "."
        else:
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s.,;?!:()\'\"%-]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
        return text

class TfidfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_range=(1, 3), max_features=None):
        self.ngram_range = ngram_range
        self.max_features = max_features

    def fit(self, X, y=None):
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.max_features,
                                          lowercase=False, sublinear_tf=True, strip_accents='unicode')
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)

def preprocess_data(train_file, test_file, strategy='light'):
    import pandas as pd
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    preprocessor = TextPreprocessor(strategy=strategy)
    train_df['text'] = preprocessor.transform(train_df['text'])
    train_df = train_df.drop_duplicates(subset=['text'])
    train_df.reset_index(drop=True, inplace=True)

    test_df['text'] = preprocessor.transform(test_df['text'])

    return train_df, test_df
