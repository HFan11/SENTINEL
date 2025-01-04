from preprocessing import preprocess_data
from feature_extraction import HybridFeatureExtractor
from models import train_models, objective
from ensemble import create_ensemble, evaluate_ensemble, run_ensemble
from evaluation import evaluate_model
from transformers import RobertaTokenizer, RobertaModel
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import optuna

def main():
    # Preprocess the data
    train_file = "train_data.csv"
    test_file = "test_data.csv"
    train_df, test_df = preprocess_data(train_file, test_file)

    # Extract features
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roberta_model = roberta_model.to(device)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    feature_extractor = HybridFeatureExtractor(roberta_model, roberta_tokenizer, tfidf_vectorizer)

    X_train = train_df['text'].tolist()
    y_train = train_df['generated']
    X_test = test_df['text'].tolist()

    feature_extractor.fit(X_train)
    X_train_features = feature_extractor.transform(X_train)
    X_test_features = feature_extractor.transform(X_test)

    # Train individual models
    models = train_models()

    # Evaluate individual models
    for model_name, model in models.items():
        print(f"Evaluating {model_name}:")
        evaluate_model(model, X_train_features, y_train)

    # Hyperparameter optimization using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_features, y_train), n_trials=100)
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # Train the ensemble model using the best hyperparameters
    ensemble = create_ensemble(models.values(), best_params['weights'])
    y_pred_ensemble = run_ensemble(ensemble, X_train_features, y_train)

    # Evaluate the ensemble model
    evaluate_ensemble(y_train, y_pred_ensemble)

    # Make predictions on the test set
    test_preds = ensemble.predict_proba(X_test_features)[:, 1]

    # Save the submission file
    submission_df = pd.DataFrame({'id': test_df['id'], 'generated': test_preds})
    submission_df.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    main()
