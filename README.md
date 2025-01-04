
# SENTINEL: A Heterogeneous Ensemble Framework for Detecting AI-Generated Text in the Era of Advanced Language Models

We develop SENTINEL model, a comprehensive solution that employs a novel heterogeneous ensembling framework to effectively distinguish human-written text from AI-generated content in academic essays. By dynamically combining Deep Learning (DL) models like BERT and RoBERTa with Classical Machine Learning (CML) algorithms such as MNB, SGD, LightGBM, and CatBoost, SENTINEL leverages the strengths of both deep learning and traditional algorithms.

Our contributions include the development of the custom `TextPreprocessor`, `HybridFeatureExtractor`, and the heterogeneous ensemble architecture with optimized soft voting. We also curate a dynamic dataset comprising over 10,000 entries generated using cutting-edge LLMs to ensure SENTINEL's adaptability. The hybrid feature extraction technique integrates contextual embeddings from RoBERTa with TF-IDF n-grams, enhancing the model's ability to capture both deep semantic structures and surface-level textual patterns. Advanced hyperparameter optimization using Optuna and robust validation strategies ensure superior performance.

## Key Features
- **Enhanced Feature Representation**: Combines contextual embeddings with TF-IDF n-grams.
- **Advanced Hyperparameter Optimization**: Utilizes Bayesian Optimization and Optuna for fine-tuning.
- **Robust Validation Strategies**: Includes stratified and adaptive cross-validation techniques.
- **Interpretative Insights**: Leverages SHAP and LIME for detailed feature contribution analysis.
- **Comprehensive Testing**: Validated on a diverse Kaggle dataset comprising 10,000 academic essays.

## Installation
1. Clone the repository:
1. Clone the repository:
   ```
   git clone https://github.com/NatalieCao323/AITextDetection.git
   ```
2.  Navigate to the project directory and install the required dependencies:
   ```
   cd AITextDetection
   pip install -r requirements.txt
   ```

## Usage
1. Prepare your training and test datasets in CSV format.
2. Update the file paths in `main.py` to point to your dataset files.
3. Run the main script:
   ```
   python main.py
   ```
4. The script will train the ensemble model, evaluate its performance, and generate predictions on the test set.

## Project Structure
- `main.py`: Main entry point of the project
- `preprocessing.py`: Preprocessing utilities and custom transformers
- `feature_extraction.py`: Feature extraction utilities and custom transformers
- `models.py`: Individual model definitions and hyperparameter tuning utilities
- `ensemble.py`: Ensemble model definition and training utilities
- `evaluation.py`: Evaluation metrics and utilities
- `utils.py`: Utility functions used across the project
- `requirements.txt`: List of required dependencies

## Dataset
Our project utilizes two distinct datasets: the original dataset from the [Kaggle competition](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/overview) (around 10,000 essays). The Kaggle dataset comprises about 10,000 essays, sourced from a competition and used primarily for training and testing. We apply preprocessing techniques such as duplicate removal, text correction, and Byte Pair Encoding (BPE) tokenization to improve data quality.

In addition, we conducted experiments on a diverse and comprehensive dataset developed from various sources, consisting of 11,580 text entries. The dataset sources include a variety of repositories and collections: HPPT, IDMGSP, Med-MMHL, OpenGPTText, HC-Var, fakenews machine data, Deepfake bot submissions, Human ChatGPT Comparison Corpus (HC3), MGTBench, and Alpaca. This results in a balanced dataset of 5790 human-written and 5790 AI-generated texts. The task associated with this dataset is binary classification, where the input is a text excerpt, and the output is a label indicating whether the text is human-written (0) or AI-generated (1). The dataset provides a diverse range of text types and styles, enabling the evaluation of AI-generated text detection models across various domains.

Both datasets are employed for binary classification tasks, where models discern if a text is human-written (0) or AI-generated (1). The extensive variety in our developed datasets allows our models to be tested across different text types and styles, ensuring robustness and generalizability in detecting AI-generated content. This combination of datasets, with precise preprocessing and a clear task definition, forms a solid foundation for our AI-generated text detection framework.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.
