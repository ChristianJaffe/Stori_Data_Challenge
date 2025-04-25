# Stori Data Science Challenge

## Introduction

This repository contains the analysis and code developed in response to the Stori Data Science Challenge, the primary goals were to perform exploratory data analysis (EDA) on the provided credit card dataset (`data_stori.csv`), address specific questions regarding customer balance and activity, and build a predictive model for fraud detection.

**Note:** The challenge materials requested discretion regarding public sharing of submissions. This repository is primarily intended for documenting the work performed.

## Challenge Description

The detailed challenge description, including the specific questions addressed (Q1, Q2, Q3), can be found in the included PDF file:
* `Stori_Data_Challenge.pdf`

## Repository Structure

* **`Stori_Challenge_Analysis.ipynb`**: The main Jupyter Notebook containing the step-by-step analysis, visualizations, model training/evaluation code, interpretations, and conclusions. This is the primary file to review the work.
* **`model_utils.py`**: A Python script containing helper functions and classes used within the notebook to promote code organization and reusability. This includes:
    * `read_dataset()`: Function to load data from various formats.
    * `create_preprocessor()`: Function to create the standard data preprocessing pipeline (imputation + scaling).
    * `FraudModelTrainer`: A class encapsulating model training, evaluation, and feature importance extraction.
* **`data_stori.csv`**: The dataset provided for the challenge.
* **`Stori_Data_Challenge.pdf`**: The original challenge description document.

## Methodology Overview

The analysis follows a standard data science workflow:

1.  **Exploratory Data Analysis (EDA):** Investigated data distributions (e.g., `balance`, `fraud`), correlations between features, and potential outliers using histograms and box plots. Log transformations were used to better visualize skewed distributions.
2.  **Data Preparation:** Handled missing values (using median imputation via `SimpleImputer`), identified relevant features, performed feature scaling (using `StandardScaler`), and split the data into training and testing sets.
3.  **Modeling (Fraud Prediction - Q3):**
    * Addressed the significant **class imbalance** in the `fraud` target variable.
    * Implemented and compared two classification models:
        * **Logistic Regression:** Used as a baseline, incorporating `class_weight='balanced'`.
        * **XGBoost Classifier:** A gradient boosting model known for high performance, using `scale_pos_weight` for imbalance.
    * Evaluated models using appropriate metrics for imbalanced classification (Precision, Recall, F1-Score for the fraud class, AUC-ROC, Average Precision Score).
    * Analyzed feature importance for both models (coefficients for LR, feature importance scores for XGBoost).
4.  **Interpretation & Conclusions:** Summarized findings from EDA, model performance, identified key predictors, and discussed the trade-offs between the models. Code was structured using functions and classes for clarity.

## Technologies Used

* Python 3.12.4
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Jupyter Notebook / Lab

## Usage

1.  Ensure you have the required Python libraries installed (e.g., `pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyterlab`).
2.  Place `model_utils.py` and `data_stori.csv` in the same directory as the `Stori_Challenge_Analysis.ipynb` notebook.
3.  Open and run the cells in the `Stori_Challenge_Analysis.ipynb` notebook sequentially.
