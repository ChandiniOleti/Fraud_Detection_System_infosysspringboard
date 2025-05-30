# Fraud_Detection_System_infosysspringboard
This repository contains a machine learning project focused on detecting fraudulent transactions in credit card datasets. The goal is to build a reliable and efficient model that minimizes false positives and accurately identifies fraudulent activities.


## Overview
This project aims to classify credit card transactions as fraudulent or non-fraudulent using machine learning techniques. The dataset used for this project is the "Credit Card Fraud Detection" dataset from Kaggle, which contains anonymized features and transaction information.

## Problem Statement
The objective is to build a robust model capable of identifying fraudulent transactions to mitigate risks and improve financial security.

## Dataset
- **Source**: Kaggle - [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Description**: The dataset includes 284,807 transactions with 31 features (28 anonymized features, time, amount, and class).
- **Class Distribution**:
  - 0: Non-fraudulent transactions (majority class)
  - 1: Fraudulent transactions (minority class)

## Dataset Overview
- **Total Data**: 8,83,097
- **Total Rows in Dataset**: 28487
- **Columns**: 31
- **Class Distribution in Raw Data**:
  - Non-Fraudulent Transactions: 284315
  - Fraudulent Transactions: 495

## Dependencies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn
- XGBoost
- CatBoost
  
## Steps Followed

### 1. Data Preprocessing
- Checked for null values and data consistency.
- Scaled numerical features using `RobustScaler`.
- Balanced the class distribution using SMOTE (Synthetic Minority Oversampling Technique).

### 2. Exploratory Data Analysis (EDA)
- Analyzed the distribution of fraudulent and non-fraudulent transactions.
- Visualized transaction amount trends and time-based patterns.
- Applied standardization to numerical features.
- Addressed the class imbalance using oversampling techniques (e.g., SMOTE).
  
- **Correlation Heatmap**:
  - Analyzed correlations between features and the target variable.
  - Identified features most related to fraudulent transactions.

### 3. Model Development
- Implemented various machine learning models:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - XGBoost
  - LightGBM
  - CatBoost
  - SVM
    
- Evaluated performance using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.

### 4. Model Evaluation
- Used cross-validation to assess model performance.
- Compared metrics to identify the best-performing model.
- **Best Performing Model**: CatBoost with an accuracy of 99.95%



🚀 ##Features
Exploratory Data Analysis (EDA): Gain insights into the dataset with detailed visualizations.
Preprocessing: Handles imbalanced datasets using techniques like oversampling (SMOTE).
Feature Engineering: Selects key features to optimize model performance.
Machine Learning Models: Implements algorithms like Logistic Regression, Random Forest, and Gradient Boosting for fraud detection.
Evaluation Metrics: Utilizes precision, recall, F1-score, and AUC-ROC curve for performance evaluation.


## Results
- Achieved high precision and recall on the minority (fraudulent) class.
- Selected the best-performing model based on AUC-ROC and F1-score.
- Fraudulent transactions are sparse, but the models effectively classified them.

## Conclusion
This project successfully demonstrates the application of machine learning techniques to detect fraudulent transactions in credit card data. By addressing challenges such as class imbalance and feature anonymization, the developed models achieved high accuracy and reliability. The insights gained can be extended to real-world systems to enhance financial security and mitigate fraud risks.




📂 Dataset
The project uses the Credit Card Fraud Detection Dataset, which contains anonymized features and includes both fraudulent and non-fraudulent transactions.

🛠️ Installation

Navigate to the project directory:
cd credit_card_fraud_detection
Install dependencies:
pip install -r requirements.txt

⚙️ How It Works
1. Load and preprocess the dataset.
2. Perform EDA to understand data distribution and detect class imbalance.
3. Train and evaluate models using different machine learning algorithms.
4. Select the best-performing model based on evaluation metrics.

📊 Results
The model achieves high accuracy and AUC-ROC scores while maintaining a balance between precision and recall to minimize fraud detection errors.

📧 Contact
For questions or collaborations, feel free to reach out at oletichandini78@gmail.com
