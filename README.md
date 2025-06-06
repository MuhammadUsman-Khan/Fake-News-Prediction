# 📰 Fake News Prediction using Machine Learning

This project builds a machine learning model to classify news articles as real or fake using **Natural Language Processing (NLP)** and **Logistic Regression**.

---

## 🔍 Problem Statement

Fake news poses a significant challenge in the digital world. This project aims to classify news articles based on their text content into two categories:

* Real News (label = 0)

* Fake News (label = 1)

*⚠️ Note: The model's predictions are limited to the provided dataset and may not generalize well to real-world news articles.*

---

## 📁 Dataset

* Source: Kaggle Fake News Dataset

* File used: *train.csv*

* Features: *id, title, author, text, label*

---

## 🛠️ Technologies Used

* **Python** *(Google Colab)*

* **Numpy, Pandas**

* **nltk** for stopwords removal and stemming

* **scikit-learn** for TF-IDF, model training, and evaluation

---

## ⚙️ Data Preprocessing

1. **Missing Values:** Replaced with empty strings

2. **Content Creation:** Combined author and title into a new content feature

3. **Text Cleaning:**

  * Remove non-alphabetic characters

  * Convert to lowercase

  * Tokenize

  * Remove stopwords

  * Apply stemming

4. **Feature Extraction:** Used *TfidfVectorizer* to convert text into numerical form

---

## 🤖 Model Details

* Algorithm: **Logistic Regression**

* Data Split: 80% training, 20% testing

* Input: *TF-IDF* features from preprocessed text

* Evaluation: Accuracy score

---

## 📊 Results

* Successfully trained and evaluated the model.

* Achieved good accuracy on the test set from the dataset.

---

## 🧪 Usage

* Upload ```train.csv``` to your Colab session

* Run the notebook cells sequentially

* The notebook handles all preprocessing, training, and prediction

---

## 🚧 Limitations

* The model is trained and tested only on the provided dataset.

* It may not accurately classify news from outside sources due to:

   * Dataset bias

   * Lack of contextual understanding

   * No real-world generalization capabilities
 
---

## 🔮 Future Improvements

* Experiment with advanced models (SVM, XGBoost)

* Use deep learning models (LSTM, BERT)

* Apply cross-validation and hyperparameter tuning

* Train on more diverse and recent data sources

---
