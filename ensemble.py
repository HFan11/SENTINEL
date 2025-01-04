from sklearn.ensemble import VotingClassifier
from models import CustomRobertaClassifier, mnb, sgd, lgbm, catboost
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np

def create_ensemble(estimators, weights):
    ensemble = VotingClassifier(estimators=estimators, voting='soft', weights=weights, n_jobs=-1)
    return ensemble

def evaluate_ensemble(ensemble, X_train_features, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_ensemble = cross_val_predict(ensemble, X_train_features, y_train, cv=cv, method='predict_proba')

    auc_score = roc_auc_score(y_train, y_pred_ensemble[:, 1])
    accuracy = accuracy_score(y_train, np.argmax(y_pred_ensemble, axis=1))
    f1 = f1_score(y_train, np.argmax(y_pred_ensemble, axis=1))

    print(f"Ensemble AUC: {auc_score:.4f}")
    print(f"Ensemble Accuracy: {accuracy:.4f}")
    print(f"Ensemble F1 Score: {f1:.4f}")

def run_ensemble(X_train_features, y_train, model_checkpoint):
    custom_classifier = CustomRobertaClassifier(model_checkpoint)

    estimators = [
        ('mnb', mnb),
        ('sgd', sgd),
        ('lgbm', lgbm),
        ('catboost', catboost),
        ('roberta', custom_classifier)
    ]

    # Load the weights from a file or define them here
    weights = load_weights_from_file('weights.json')  # Implement this function to load weights from a file
    # weights = [0.1, 0.31, 0.28, 0.67, 0.8]  # Alternatively, define the weights directly here

    ensemble = create_ensemble(estimators, weights)
    evaluate_ensemble(ensemble, X_train_features, y_train)
    return ensemble
