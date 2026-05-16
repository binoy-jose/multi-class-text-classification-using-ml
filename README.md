# Multi-Class Text Classification using Machine Learning

## Project Overview

This project focuses on solving a **multi-class text classification** problem using Natural Language Processing (NLP) and Machine Learning techniques. The objective is to classify text comments into multiple categories using TF-IDF vectorization and metadata-based feature engineering.

The project includes:

* Exploratory Data Analysis (EDA)
* Text preprocessing and cleaning
* TF-IDF feature extraction
* Feature engineering
* Model training and evaluation
* Hyperparameter tuning
* Performance comparison of multiple models

---

# Dataset Features

The dataset contains both text and metadata features:

* `created_date`
* `post_id`
* `emoticon_1`
* `emoticon_2`
* `emoticon_3`
* `upvote`
* `downvote`
* `if_1`
* `if_2`
* `race`
* `religion`
* `gender`
* `disability`
* `comment`
* `label`

---

#  Technologies & Libraries Used

##  Programming Language

* Python

##  Main Libraries

### Data Handling

* pandas
* numpy

### Visualization

* matplotlib
* seaborn
* wordcloud

### NLP & Feature Engineering

* scikit-learn
* TfidfVectorizer

### Machine Learning Models

* Logistic Regression
* LinearSVC
* LightGBM

### Model Evaluation

* accuracy_score
* f1_score
* classification_report
* confusion_matrix

---

#  Exploratory Data Analysis (EDA)

The following EDA techniques were performed:

* Class imbalance visualization
* Comment length analysis
* Word count analysis
* Duplicate value checking
* Word cloud visualization
* Box plots for text statistics

---

#  Data Preprocessing

The preprocessing pipeline includes:

* Handling missing values
* Removing duplicate comments
* Text cleaning
* Converting text to lowercase
* TF-IDF vectorization
* Combining text and numerical features

---

#  Feature Engineering

## TF-IDF Vectorization

Text comments were converted into numerical vectors using:

```python
TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)
```

### Techniques Used

* Unigrams and Bigrams
* Rare word filtering
* Frequent word filtering
* Sublinear TF scaling

---

#  Models Used

## 1. Logistic Regression

* Linear classification model
* Uses sigmoid/softmax probabilities
* Effective baseline model for text classification

## 2. LinearSVC

* Support Vector Machine based classifier
* Uses hinge loss and maximum margin separation
* Performs efficiently on sparse high-dimensional TF-IDF data

## 3. LightGBM

* Gradient boosting framework
* Uses leaf-wise tree growth and histogram-based learning
* Fast and efficient for large feature spaces

---

#  Hyperparameter Tuning

The project used:

* GridSearchCV
* RandomizedSearchCV

## Parameters Tuned

Examples:

* `C`
* `learning_rate`
* `n_estimators`
* `num_leaves`
* `max_depth`

---

#  Evaluation Metrics

The models were evaluated using:

* Accuracy
* F1 Score
* Precision
* Recall
* Confusion Matrix

---

#  Project Workflow

```text
Raw Text Data
↓
Data Cleaning & Preprocessing
↓
Exploratory Data Analysis
↓
TF-IDF Feature Extraction
↓
Feature Engineering
↓
Model Training
↓
Hyperparameter Tuning
↓
Prediction & Evaluation
```

---

#  Key Learnings

Through this project, I gained practical experience in:

* Natural Language Processing (NLP)
* TF-IDF vectorization
* Text classification techniques
* Handling sparse high-dimensional data
* Machine Learning model comparison
* Hyperparameter tuning
* Model evaluation and interpretation

---

#  Future Improvements

Potential future improvements include:

* Deep Learning models (LSTM, Transformers)
* BERT embeddings
* Advanced text preprocessing
* Ensemble techniques
* Better handling of class imbalance

---

#  Conclusion

This project demonstrates the implementation of multiple machine learning approaches for multi-class text classification using NLP techniques. Logistic Regression, LinearSVC, and LightGBM models were trained and evaluated to understand their effectiveness on sparse text data.

---


