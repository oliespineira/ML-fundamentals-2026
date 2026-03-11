# Individual Assignment I  
## AI: Machine Learning Foundations  

### Bank Marketing – Term Deposit Subscription Prediction  

---

## Overview

This project implements a complete supervised machine learning pipeline to predict whether a client will subscribe to a term deposit. The emphasis of this assignment is not solely on model performance, but on correct pipeline discipline, prevention of data leakage and principled preprocessing.

The modelling approach uses Logistic Regression as a linear classifier. Class imbalance is addressed using SMOTE, and evaluation is performed using appropriate metrics beyond simple accuracy.

---

## Problem Formulation

The prediction target is the variable `y`, which encodes whether the client subscribed to a term deposit (`yes` or `no`). This defines a binary classification task.

The objective is to predict, using client and campaign information available at the time of contact, whether a client will subscribe. Any variable unavailable at prediction time must be excluded to prevent temporal leakage.

---

## Pipeline Structure

The pipeline follows a strict and deliberate order:

1. Identify prediction target  
2. Data loading and exploratory analysis  
3. Train/validation split (stratified)  
4. Missing value handling (fitted on training only)  
5. Encoding of categorical variables (fitted on training only)  
6. Feature scaling (fitted on training only)  
7. SMOTE resampling (training set only)  
8. Logistic Regression training  
9. Threshold adjustment and evaluation  

All transformations that learn from the data are fitted exclusively on the training set and applied via `transform` to validation data to prevent data leakage.

---

## Exploratory Data Analysis

Key observations from the exploratory phase:

- The dataset contains 4119 observations and 21 variables.
- The target variable is heavily imbalanced (~89% no, ~11% yes).
- Several numerical variables show right skew.
- The variable `pdays` contains a sentinel value (999) indicating "not previously contacted".
- The variable `duration` constitutes temporal leakage and is removed before modelling.

These findings directly informed preprocessing decisions.

---

## Handling Class Imbalance

Given the 89:11 imbalance, SMOTE was applied to the training set only. This helps the model learn minority class structure without altering validation distribution.

Evaluation is performed on the original class proportions to preserve real-world realism.

---

## Model

Logistic Regression was selected because:

- The task is binary classification.
- The decision boundary is linear and interpretable.
- It allows probability outputs and threshold tuning.

The model was first evaluated at the default threshold (0.5) and later at an adjusted threshold (0.65) to better reflect business priorities.

---

## Results

- Accuracy: ~88% at threshold 0.65  
- AUC: 0.79  
- Average Precision: 0.46  

While overall accuracy approaches the Zero Rule baseline, the model meaningfully improves Recall for the minority class, demonstrating genuine predictive value beyond majority-class prediction.

---

## Limitations and Potential Improvements

Future improvements could include:

- Partial SMOTE balancing to reduce distribution shift  
- Hyperparameter tuning of regularisation strength  
- Systematic threshold optimisation using Precision–Recall curves  
- Interaction feature engineering  
- Using class weights as an alternative to resampling  

These were not implemented, as the primary objective of the assignment is pipeline correctness and methodological rigour.

---

## Reproducibility

To run this project:

1. Install required libraries:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn
   - imbalanced-learn

2. Ensure `bank-additional.csv` is located in the working directory.

3. Execute the notebook sequentially to preserve pipeline order.

---

## Final Remarks

This implementation prioritises:

- Prevention of data leakage  
- Proper separation of training and validation data  
- Correct ordering of preprocessing steps  
- Transparent modelling assumptions  

The resulting model is coherent, methodologically sound and aligned with the business objective of identifying potential subscribers.