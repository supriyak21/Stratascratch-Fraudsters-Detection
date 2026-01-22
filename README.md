# Fraudster Detection (Users + Transactions)

This repository contains an end-to-end solution to identify fraudulent users using two provided datasets:
- `users.csv` (user attributes + target label `IS_FRAUDSTER`)
- `transactions.csv` (transaction history for users)

The project follows a complete ML workflow: exploratory analysis → feature engineering → model training → evaluation.



# Problem Statement

Build a predictive model that identifies fraudsters (`IS_FRAUDSTER`) using only the provided data (no external sources).  
The solution should clearly show the thought process, code structure, and reasoning behind features and evaluation.


# Data

# Files
- `users.csv`: user-level data (target column: `IS_FRAUDSTER`)
- `transactions.csv`: transaction-level data
- `countries.csv`: dictionary for user `COUNTRY` codes (optional)
- `currency_details.csv`: dictionary for transaction `CURRENCY` codes (optional)

# Target
- `IS_FRAUDSTER` (user-level label)

# Approach

1) Exploratory Data Analysis (EDA)
   Key checks performed:
  - Fraud class imbalance (fraud users are a small minority)
  - Distribution differences between fraud and non-fraud transactions:
  - transaction amount behavior
  - transaction types
  - entry methods
  - timing patterns (hour / weekend / night)


2) Feature Engineering (Simple + Interpretable)
   Because the target is **user-level**, transaction data is aggregated to **one row per user**.
   
  **User-level features (from `users.csv`)**
  - `HAS_EMAIL`
  - `FAILED_SIGN_IN_ATTEMPTS`
  - `KYC` (one-hot encoded)
  - `COUNTRY` (one-hot encoded)

  **Transaction aggregate features (per `USER_ID`)**
  - `tx_count` (number of transactions)
  - `active_days` (unique transaction days)
  - `avg_amount_usd`, `max_amount_usd`, `total_amount_usd`
  - `weekend_avg_txn` (share of weekend transactions)
  - `night_avg_txn` (share of night transactions)

  **Transaction behavior mix**
  - `% of transaction TYPE` per user (via crosstab normalize)
  - `% of ENTRY_METHOD` per user (via crosstab normalize)


3) Model
   Baseline model:
    - **Logistic Regression** (`class_weight="balanced"`) to handle class imbalance
    - Simple preprocessing:
      - `pd.get_dummies()` for `KYC` and `COUNTRY`
      - Drop `USER_ID` before training

Data split:
- Train / Validation / Test split using stratification


4) Evaluation
   Because fraud detection is imbalanced, evaluation focuses on:
    - **ROC-AUC**
    - **PR-AUC (Average Precision)** (more informative for rare classes)
    - Threshold tuning using validation probabilities
    - Confusion matrix + Precision / Recall / F1 on test set
    
    A tuned threshold is used to balance:
    - **Precision** (reduce false positives)
    - **Recall** (catch more fraud)



## Repository Structure

