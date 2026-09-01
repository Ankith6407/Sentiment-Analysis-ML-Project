# 💬 SentimentIQ: Machine Learning-Based Sentiment Analysis

## 📌 Project Overview

SentimentIQ is a Machine Learning and Natural Language Processing (NLP) project designed to classify textual data into different sentiment categories.

The application analyzes user-provided text and predicts whether the sentiment expressed is:

- 😊 Positive
- 😐 Neutral
- 😞 Negative

The project uses **TF-IDF Vectorization** for text feature extraction and **Linear Support Vector Machine (Linear SVM)** as the final classification model.

An interactive web application was developed using **Streamlit**, allowing users to enter text and receive real-time sentiment predictions.

---

## 🎯 Project Objective

The main objective of this project is to build and evaluate multiple Machine Learning models for sentiment classification and select the best-performing model based on evaluation metrics.

The workflow includes:

- Data loading and understanding
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Text cleaning
- Feature extraction using TF-IDF
- Training multiple Machine Learning models
- Model evaluation and comparison
- Final model selection
- Model serialization
- Interactive deployment using Streamlit

---

## 📊 Exploratory Data Analysis

The dataset was analyzed to understand its overall structure and quality.

The EDA process included:

- Dataset shape analysis
- Data type inspection
- Missing value analysis
- Duplicate value checking
- Sentiment class distribution analysis
- Text length analysis
- Word count analysis
- Visualization of sentiment distribution

These steps helped understand the characteristics of the dataset before model development.

---

## 🧹 Data Preprocessing

The textual data was prepared for Machine Learning using appropriate preprocessing techniques.

The preprocessing workflow included:

- Text cleaning
- Converting text into a suitable format
- Removing unnecessary characters
- Preparing the target sentiment variable
- Splitting the dataset into training and testing sets

---

## 🔤 Feature Engineering

The text data was converted into numerical features using:

### TF-IDF Vectorization

**TF-IDF (Term Frequency-Inverse Document Frequency)** was used to transform text into numerical feature vectors.

TF-IDF helps identify important words based on:

- Their frequency within a document
- Their importance across the complete dataset

The resulting feature vectors were used as input for the Machine Learning models.

---

## 🤖 Machine Learning Models Evaluated

A total of **6 Machine Learning classification models** were evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Bernoulli Naive Bayes
4. Linear Support Vector Machine (Linear SVM)
5. Random Forest Classifier
6. K-Nearest Neighbors Classifier

The models were compared using multiple evaluation metrics.

---

## 📈 Model Performance

The following metrics were used for model evaluation:

- Accuracy
- Precision
- Recall
- F1 Score

### Model Comparison

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Linear SVM | **78.82%** | **75.99%** |
| Logistic Regression | 78.47% | 73.17% |
| Bernoulli Naive Bayes | 76.39% | 72.28% |
| K-Nearest Neighbors | 74.31% | 71.52% |
| Random Forest | 75.69% | 69.98% |
| Multinomial Naive Bayes | 74.65% | 68.82% |

---

## 🏆 Final Model

After comparing all six models, **Linear Support Vector Machine (Linear SVM)** was selected as the final model.

### Final Model Performance

| Metric | Score |
|---|---:|
| Test Accuracy | **78.82%** |
| Weighted F1 Score | **75.99%** |

Linear SVM achieved the best overall performance among the evaluated models.

---

## ⚠️ Model Generalization Analysis

The final model was also evaluated by comparing training and testing performance.

| Model | Training Accuracy | Testing Accuracy | F1 Score | Generalization Gap |
|---|---:|---:|---:|---:|
| Linear SVM | 100.00% | 78.82% | 75.99% | 21.18% |

The difference between training and testing accuracy indicates that the model may have some level of overfitting. However, Linear SVM was retained as the final model because it achieved the best test performance among the evaluated models.

---

## 🌐 Streamlit Application

An interactive web application was developed using **Streamlit**.

The application allows users to:

- Enter custom text
- Use pre-defined sentiment examples
- Analyze text sentiment
- View the predicted sentiment
- View normalized model decision scores
- Track session prediction history
- Clear input text and session history

---

## 📂 Project Structure

```text
Sentiment-Analysis-ML-Project/
│
├── app.py
├── linear_svm_sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
├── Sentiment_Analysis.ipynb
└── .gitignore
