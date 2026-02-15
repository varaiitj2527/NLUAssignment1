# Sports vs. Politics Text Classifier

## Overview
This project implements a machine learning system to classify news articles into two categories: **Politics (0)** and **Sports (1)**. It compares three models (Logistic Regression, SVM, Random Forest) using Bag of Words and TF-IDF features.

## 📂 Files
* `df_file.csv`: The dataset containing labeled news articles.
* `eda.py`: Script for Exploratory Data Analysis (Word Clouds, frequency plots).
* `models.py`: Script for training models and generating performance metrics/plots.

## ⚙️ Setup & Usage
1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn wordcloud
    ```
2.  **Run EDA:**
    ```bash
    python eda.py
    ```
3.  **Run Model Analysis:**
    ```bash
    python models.py
    ```

## 🏆 Key Results
* **Best Model:** Support Vector Machine (SVM)
* **Best Feature:** TF-IDF (1-gram)
* **Accuracy:** ~99.4%
* **Insight:** "Election" and "Government" are top predictors for Politics; "Match" and "Team" are top predictors for Sports.
