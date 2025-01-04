from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import shap
from lime import lime_text
import numpy as np

def evaluate_model(model, X, y):
    cv_methods = [
        ("Stratified K-Fold", StratifiedKFold(n_splits=5, shuffle=True, random_state=42)),
        ("Leave-One-Out", LeaveOneOut())
    ]

    for cv_name, cv in cv_methods:
        print(f"Evaluation using {cv_name} Cross-Validation:")
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"Accuracy: {np.mean(scores):.4f}")
        
        y_pred = cross_val_predict(model, X, y, cv=cv)
        
        print(f"AUC: {roc_auc_score(y, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y, y_pred):.4f}")
        print(f"Precision: {precision_score(y, y_pred):.4f}")
        print(f"Recall: {recall_score(y, y_pred):.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}\n")

def evaluate_ensemble(y_true, y_pred_ensemble):
    auc_score = roc_auc_score(y_true, y_pred_ensemble[:, 1])
    accuracy = accuracy_score(y_true, np.argmax(y_pred_ensemble, axis=1))
    f1 = f1_score(y_true, np.argmax(y_pred_ensemble, axis=1))
    precision = precision_score(y_true, np.argmax(y_pred_ensemble, axis=1))
    recall = recall_score(y_true, np.argmax(y_pred_ensemble, axis=1))
    
    print(f"Ensemble AUC: {auc_score:.4f}")
    print(f"Ensemble Accuracy: {accuracy:.4f}")
    print(f"Ensemble F1 Score: {f1:.4f}")
    print(f"Ensemble Precision: {precision:.4f}")
    print(f"Ensemble Recall: {recall:.4f}")

def interpret_model(model, X_test, X_test_features, feature_names):
    # Model interpretation using SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test_features)
    shap.summary_plot(shap_values, X_test_features, feature_names=feature_names)
    
    # Model interpretation using LIME
    explainer = lime_text.LimeTextExplainer(class_names=['Human', 'AI'])
    exp = explainer.explain_instance(X_test[0], model.predict_proba, num_features=6)
    exp.show_in_notebook(text=True)
