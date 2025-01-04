from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomRobertaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, X, y):
        return self

    def predict(self, X):
        self.model.eval()
        encoded_inputs = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt')
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            return torch.argmax(probs, dim=1).cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        encoded_inputs = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt')
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            return probs.cpu().numpy()

def train_models():
    mnb = LogisticRegression(random_state=42)
    sgd = SGDClassifier(max_iter=9000, tol=1e-4, random_state=6743)
    lgbm = LGBMClassifier(n_estimators=3000, objective='binary', metric='auc', random_state=6743)
    catboost = CatBoostClassifier(iterations=3000, random_seed=6543, verbose=False)

    models = {
        'LogisticRegression': mnb,
        'SGDClassifier': sgd,
        'LGBMClassifier': lgbm,
        'CatBoostClassifier': catboost
    }

    return models
