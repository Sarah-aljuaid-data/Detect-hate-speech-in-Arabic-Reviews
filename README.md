# Detect-hate-speech-in-Arabic-Reviews
# Arabic Hate Speech Detection using AraBERT, MLP, and Traditional Machine Learning Models

## Overview
This project aims to classify Arabic text reviews into three categories: **hate speech**, **neutral**, and **non-hate speech** using deep learning and traditional machine learning models. The project leverages the power of **pre-trained transformers**, **neural networks**, **TF-IDF-based classifiers**, along with **oversampling techniques** to address class imbalance and improve performance.

The models used in this project include:

- **AraBERTv2** (Fine-tuned Transformer for sequence classification)
- **Multi-Layer Perceptron (MLP)** Neural Network
- Traditional Machine Learning Models:
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Support Vector Machine (SVM)

## Dataset
The dataset consists of Arabic company review texts labeled into three categories:
- `-1` → Hate Speech → `0`
- `0` → Neutral → `1`
- `1` → Non-Hate Speech → `2`

### Preprocessing Steps:
- Removing diacritics and unwanted characters
- Custom Arabic stopword filtering (excluding "لا")
- Label encoding and text normalization

## Project Workflow

### 1. Data Preprocessing:
- Applied `ArabertPreprocessor` for AraBERT input
- Manual cleaning for TF-IDF models (punctuation removal, token filtering)
- SMOTE applied to balance class distributions

### 2. Feature Representation:
- **AraBERTv2:** Tokenization and embedding via HuggingFace Transformers
- **MLP & Traditional Models:** TF-IDF vectorization with up to 5000 features

### 3. Model Training:
- AraBERT: Trained using `Trainer` API with early stopping and weight decay
- MLP: 4 dense layers + dropout, trained using Keras with validation split
- ML Models: Trained with `GridSearchCV` using an imbalanced pipeline with SMOTE

### 4. Evaluation:
- Metrics: `accuracy`, `weighted F1-score`, `confusion matrix`
- 5-fold cross-validation used for classical models
- Performance visualized with heatmaps

## Visualizations
-  Class distribution before and after SMOTE
-  Confusion matrices for all models
-  Accuracy and F1-scores in printed reports

## Installation & Requirements

```bash
pip install transformers==4.51.3 arabert datasets scikit-learn imbalanced-learn keras matplotlib seaborn nltk
```

### Note:
Download NLTK Arabic stopwords before running:
```python
import nltk
nltk.download('stopwords')
```

## Running the Project
Ensure your dataset file is named `CompanyReviews.csv` and placed under `/content/sample_data/`.

- For AraBERT:
```python
trainer.train()
```

- For MLP:
```python
model.fit(X_train_bal, y_train_bal, ...)
```

- For traditional models:
```python
grid.fit(X_train, y_train)
```

## Conclusion
The results indicate that **AraBERT** provides the strongest contextual understanding of Arabic text, especially in distinguishing subtle expressions of hate speech. However, MLP and optimized traditional models also deliver competitive performance when trained on well-balanced TF-IDF features. O

## Future Improvements
- Add model explainability (LIME or SHAP) for interpretability
- Explore fine-grained hate speech detection (by category: religion, gender, etc.)
- Deploy as a web service for real-time Arabic content moderation

---

### Author: Sarah Ajuaid 
For any queries, feel free to reach out!
