# 💬 SentimentIQ – Machine Learning Sentiment Analysis

## 📌 Project Overview

SentimentIQ is an end-to-end Natural Language Processing and Machine Learning project designed to classify textual data into sentiment categories.

The project analyzes user-provided text such as product reviews, comments, and feedback and predicts the overall sentiment.

The application classifies text into the following three categories:

- 😊 Positive
- 😐 Neutral
- 😞 Negative

The final model was selected after evaluating and comparing multiple machine learning algorithms. The trained model is integrated into an interactive Streamlit web application for real-time sentiment prediction.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Perform Exploratory Data Analysis (EDA) on the sentiment dataset.
- Clean and preprocess textual data.
- Convert text into numerical features using TF-IDF Vectorization.
- Train and compare multiple Machine Learning classification models.
- Evaluate models using appropriate classification metrics.
- Select the best-performing model.
- Save the trained model and TF-IDF vectorizer.
- Build an interactive Streamlit application for real-time sentiment prediction.
- Deploy the application for practical use.

---

## 📊 Machine Learning Models Evaluated

The following six machine learning models were trained and evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Bernoulli Naive Bayes
4. Linear Support Vector Machine (Linear SVM)
5. Random Forest Classifier
6. K-Nearest Neighbors (KNN)

After comparing the model performance, **Linear Support Vector Machine (Linear SVM)** was selected as the final model.

---

## 🏆 Final Model Performance

| Model | Accuracy | Weighted F1 Score |
|---|---:|---:|
| Linear SVM | **78.82%** | **75.99%** |

The Linear SVM model achieved the highest accuracy among the evaluated models and was selected as the final model for deployment.

---

## 🧠 Text Processing

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency) Vectorization** to convert textual data into numerical features.

TF-IDF helps represent the importance of words in individual documents relative to the complete dataset.

### Workflow

```text
Input Text
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Linear SVM Model
    ↓
Sentiment Prediction
```

---

## 🔍 Model Evaluation Metrics

The machine learning models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

### Why these metrics?

- **Accuracy** measures the overall percentage of correct predictions.
- **Precision** measures how accurate the positive predictions are.
- **Recall** measures how well the model identifies relevant sentiment classes.
- **F1 Score** provides a balance between Precision and Recall.

---

## 🚀 Streamlit Application

The trained Linear SVM model and TF-IDF vectorizer are integrated into a Streamlit web application.

The application allows users to:

- Enter custom text for sentiment analysis.
- Use quick example reviews.
- Predict Positive, Neutral, or Negative sentiment.
- View model decision scores.
- View session prediction history.
- Clear the input and prediction history.
- Access technical information about the model.

---

## 📁 Project Structure

```text
Sentiment-Analysis-ML-Project/
│
├── app.py
├── requirements.txt
├── linear_svm_sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── README.md
├── .gitignore
└── Sentiment_Analysis.ipynb
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
```

### 2. Navigate to the Project Directory

```bash
cd YOUR-REPOSITORY-NAME
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🖥️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Linear Support Vector Machine
- Streamlit
- GitHub

---

## 📈 Model Comparison

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) |
|---|---:|---:|---:|---:|
| Linear SVM | **78.82** | 75.39 | 78.82 | **75.99** |
| Logistic Regression | 78.47 | 74.78 | 78.47 | 73.17 |
| Bernoulli Naive Bayes | 76.39 | 73.32 | 76.39 | 72.28 |
| K-Nearest Neighbors | 74.31 | 69.95 | 74.31 | 71.52 |
| Random Forest | 75.69 | 65.15 | 75.69 | 69.98 |
| Multinomial Naive Bayes | 74.65 | 65.71 | 74.65 | 68.82 |

---

## 💡 Example Prediction

### Input

```text
This product is excellent and I really enjoyed using it.
```

### Output

```text
Predicted Sentiment: Positive
```

---

## 💾 Saved Model Files

The following trained resources are saved and used by the Streamlit application:

- `linear_svm_sentiment_model.pkl`
- `tfidf_vectorizer.pkl`

These files allow the application to make predictions without retraining the model every time the application runs.

---

## 🔮 Future Improvements

Potential improvements for the project include:

- Hyperparameter optimization using GridSearchCV or RandomizedSearchCV.
- Advanced text preprocessing and feature engineering.
- Experimenting with n-grams and different TF-IDF configurations.
- Using deep learning models such as LSTM.
- Experimenting with transformer-based models such as BERT.
- Adding prediction confidence visualization.
- Deploying the application on a cloud platform.

---

## 👨‍💻 Author

**Ankith**

Aspiring Data Scientist | Machine Learning | Natural Language Processing

---

## ⭐ Conclusion

This project demonstrates an end-to-end Machine Learning workflow, including:

- Exploratory Data Analysis
- Text preprocessing
- Feature extraction using TF-IDF
- Training multiple Machine Learning models
- Model comparison and evaluation
- Final model selection
- Model serialization
- Streamlit application development
- Deployment preparation

The final **Linear Support Vector Machine model achieved 78.82% test accuracy** and was selected as the final model after comparing six different machine learning algorithms.

---

⭐ If you found this project useful, consider giving the repository a star!
