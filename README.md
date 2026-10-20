# Credit Risk & Loan Default Analysis

## Project Overview

This project analyzes loan and borrower data to identify patterns associated with loan defaults and explores how machine learning can help estimate default risk.

The project uses **MySQL, Python, Pandas, scikit-learn, and Power BI** to perform data cleaning, exploratory data analysis, risk analysis, model training, evaluation, and dashboard development.

The objective is to understand how borrower characteristics and loan-related factors are associated with default risk, and to compare Logistic Regression and Random Forest models using classification metrics.

## Dataset

The project uses the [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data).

- Original records: 32,581
- Original columns: 12
- Final cleaned records: 32,411
- Target variable: `loan_status`

Target variable:
- `0` — No default
- `1` — Default

### Main Variables

| Variable | Description |
|---|---|
| `person_age` | Borrower's age |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan risk grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_status` | Default status |
| `loan_percent_income` | Loan amount as a percentage of income |
| `cb_person_default_on_file` | Previous default history |
| `cb_person_cred_hist_length` | Length of credit history |

## Tools and Technologies

- MySQL
- SQL
- Python
- Pandas and NumPy
- Matplotlib
- scikit-learn
- SQLAlchemy
- Power BI
- DAX
- Git and GitHub

## Project Workflow

Raw Dataset → MySQL Import → SQL Data Cleaning and Validation → SQL Risk Analysis → Python EDA → Machine Learning → Power BI Dashboard → Insights and Findings

## Data Cleaning Using SQL

The raw dataset was imported into MySQL before cleaning so the original data could be inspected separately.

The following steps were performed:

- Checked the total number of records and missing values.
- Identified duplicate records.
- Investigated unrealistic age values.
- Removed duplicate records and records with age values above 100.
- Added a unique `loan_id`.
- Handled missing employment length and interest rate values using median imputation.
- Converted columns to appropriate numeric data types.
- Performed final validation checks.

The cleaned dataset was stored in the MySQL table `loans_cleaned`.

### Cleaning Results

| Metric | Result |
|---|---:|
| Original records | 32,581 |
| Final cleaned records | 32,411 |
| Original columns | 12 |
| Final columns | 13 |
| Missing employment length values after cleaning | 0 |
| Missing interest rate values after cleaning | 0 |
| Age range after cleaning | 20–94 |

The cleaning process removed duplicate records and five records with ages above 100. Missing employment length and interest rate values were filled using median values calculated from the dataset.

## SQL Analysis

SQL was used to analyze default patterns across borrower and loan characteristics.

Analysis included:

- Overall default rate
- Default rate by loan grade
- Default rate by loan intent
- Default rate by home ownership
- Default rate by income group
- Default rate by employment length
- Default rate by age group
- Default rate by previous default history
- Default rate by credit history length
- Average interest rate by default status
- Average loan amount by default status
- Average loan-to-income ratio by default status
- Total defaulted loan exposure

Three reusable SQL summary views were also created:

- Overall Risk Summary
- Grade Risk Summary
- Income Risk Summary

## Python Exploratory Data Analysis

Python was used to load the cleaned dataset from MySQL into Pandas and perform exploratory analysis.

### Analysis Performed

- Dataset structure and data types
- Missing-value verification
- Default distribution and overall default rate
- Loan grade and loan intent analysis
- Income and employment group analysis
- Previous default history analysis
- Credit history and age group analysis
- Loan amount and interest rate analysis
- Loan-to-income ratio analysis

### Visualizations Created

1. Default Rate by Loan Grade
2. Default Rate by Income Group
3. Default Rate by Previous Default History
4. Default Rate by Home Ownership
5. Default Rate by Loan Intent
6. Average Loan Amount by Loan Status
7. Average Loan-to-Income Ratio by Loan Status
8. Average Interest Rate by Loan Status

## Machine Learning

The project was extended to compare two classification models for estimating whether a loan will default:

- Logistic Regression
- Random Forest Classifier

The models were built using scikit-learn and trained on the cleaned dataset loaded from MySQL.

### Feature Selection

The models use nine features intended to be available before a loan approval decision.

**Numeric features**
- `person_age`
- `person_income`
- `person_emp_length`
- `loan_amnt`
- `loan_percent_income`
- `cb_person_cred_hist_length`

**Categorical features**
- `person_home_ownership`
- `loan_intent`
- `cb_person_default_on_file`

The target is `loan_status`.

The unique identifier `loan_id` was excluded because it does not represent a meaningful borrower characteristic. `loan_grade` and `loan_int_rate` were also excluded from this version because they may be determined during the underwriting process. This is a modelling assumption that should be verified against the intended real-world prediction scenario.

### Preprocessing

The modelling pipeline includes:

- Median imputation for missing numeric values
- Standard scaling for numeric features
- Most-frequent-value imputation for categorical features
- One-hot encoding for categorical features
- Stratified train, validation, and test splits

The data was split into:

| Dataset | Records |
|---|---:|
| Training | 19,446 |
| Validation | 6,482 |
| Testing | 6,483 |
| **Total** | **32,411** |

The validation set was used to compare decision thresholds. Final reported metrics were calculated on the test split.

### Model Evaluation

Because the dataset contains more non-default loans than defaulted loans, accuracy alone is not sufficient to evaluate the models. Precision, recall, F1-score, ROC-AUC, and Average Precision were also considered.

The decision threshold affects the balance between identifying defaulted loans and incorrectly flagging non-defaulted loans.

### Test Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Selected threshold | 0.40 | 0.50 |
| Accuracy | 83.14% | 86.41% |
| Precision | 65.34% | 71.67% |
| Recall | 48.80% | 62.62% |
| F1-score | 0.5587 | 0.6684 |
| ROC-AUC | 0.8019 | 0.8589 |
| Average Precision | 0.6299 | 0.7640 |

**Initial comparison:** Random Forest performed better than Logistic Regression on the reported test metrics, including recall, F1-score, ROC-AUC, and Average Precision.

At the selected threshold, Random Forest identified more defaulted loans while maintaining higher precision than Logistic Regression. The appropriate threshold would ultimately depend on the relative costs of missed defaults and false alerts.

These results are experimental and should not be interpreted as evidence that the model is ready for real lending decisions.

### Random Forest Feature Importance

Permutation importance was calculated on the validation dataset using Average Precision as the scoring metric.

| Feature | Importance |
|---|---:|
| `loan_percent_income` | 0.2810 |
| `person_home_ownership` | 0.2293 |
| `person_income` | 0.1063 |
| `cb_person_default_on_file` | 0.0658 |
| `loan_intent` | 0.0601 |
| `loan_amnt` | 0.0127 |
| `person_emp_length` | 0.0111 |
| `person_age` | 0.0040 |
| `cb_person_cred_hist_length` | 0.0013 |

In this evaluation, `loan_percent_income` and `person_home_ownership` had the largest permutation importance values.

Permutation importance measures how much model performance changes when a feature's values are shuffled. It describes the feature's contribution to this model's predictive performance; it does not establish causation.

### Generated Machine Learning Outputs

The script saves evaluation results under `ML/outputs/`:

- `model_comparison.csv`
- `validation_threshold_results.csv`
- `random_forest_feature_importance.csv`
- `logistic_regression_confusion_matrix.csv`
- `random_forest_confusion_matrix.csv`

These files make it easier to inspect the model comparison, threshold results, feature importance, and classification errors.

## Power BI Dashboard

The interactive dashboard was developed using Power BI.

### Page 1 — Credit Risk Overview

The overview page contains:

- Total Loans
- Total Amount at Risk
- Default Rate
- Average Interest Rate
- Portfolio by Home Ownership
- Default Rate by Loan Grade
- Grade Summary

### Page 2 — Risk Analysis

The risk analysis page contains:

- Loan Grade slicer
- Loan Intent slicer
- Home Ownership slicer
- Income Group slicer
- Loan Intent Breakdown
- Loan-to-Income Ratio vs Default scatter plot
- Average Loan Amount by Income Group

The dashboard allows users to filter the portfolio and examine risk patterns across different segments.

## Key Findings

### Overall Risk

| Metric | Result |
|---|---:|
| Total loans | 32,411 |
| Defaulted loans | 7,089 |
| Overall default rate | 21.87% |
| Average loan amount | 9,593.59 |
| Average interest rate | 11.01% |
| Total defaulted loan exposure | 76,968,675 |

### Default Rate by Loan Grade

| Loan Grade | Default Rate |
|---|---:|
| A | 9.96% |
| B | 16.32% |
| C | 20.76% |
| D | 59.06% |
| E | 64.49% |
| F | 70.54% |
| G | 98.44% |

Grade G has relatively few observations, so its high default rate should be interpreted cautiously.

### Default Rate by Income Group

| Income Group | Default Rate |
|---|---:|
| Low | 33.44% |
| Medium | 15.43% |
| High | 8.95% |

Lower-income groups had higher observed default rates in this dataset. This is an association, not proof that income alone causes default.

### Previous Default History

| Previous Default on File | Default Rate |
|---|---:|
| Yes | 37.87% |
| No | 18.44% |

Borrowers with a previous default recorded had a higher observed default rate.

### Loan-to-Income Ratio

| Loan Status | Average Loan-to-Income Ratio |
|---|---:|
| Non-default | 14.88% |
| Default | 24.69% |

Defaulted loans had a higher average loan-to-income ratio in the cleaned dataset.

### Interest Rate

| Loan Status | Average Interest Rate |
|---|---:|
| Non-default | 10.49% |
| Default | 12.88% |

### Average Loan Amount

| Loan Status | Average Loan Amount |
|---|---:|
| Non-default | 9,239.76 |
| Default | 10,857.48 |

## Limitations and Interpretation

- The findings describe associations in the dataset and do not prove causation.
- Some categories contain relatively few loans, making their default rates less stable.
- The dataset represents historical loan records. It may not represent applicants who were rejected before a loan was issued, which can introduce selection bias.
- The model's usefulness depends on whether all selected features are genuinely available at the time of the intended prediction.
- The reported test results follow iterative experimentation. The test split has been inspected during earlier model and threshold experiments, so it should not be described as a completely untouched final test set.
- Model performance may differ on new data, other lenders, or future economic conditions.
- The models are educational experiments and are not validated for production lending decisions.

## Project Structure

```text
credit-risk-analysis/
├── README.md
├── SQL/
│   └── data_cleaning_and_analysis.sql
├── Python/
│   └── credit_risk_analysis.py
├── ML/
│   ├── credit_risk_model.py
│   └── outputs/
│       ├── model_comparison.csv
│       ├── validation_threshold_results.csv
│       ├── random_forest_feature_importance.csv
│       ├── logistic_regression_confusion_matrix.csv
│       └── random_forest_confusion_matrix.csv
├── Power BI/
│   └── Credit_Risk_Analysis.pbix
└── Screenshots/
    ├── 1st Dashboard.png
    └── 2nd Dashboard.png
```

## Conclusion

This project demonstrates an end-to-end data workflow, from SQL-based data cleaning and validation to exploratory analysis, interactive Power BI reporting, and introductory machine learning.

The SQL analysis and dashboard identify patterns in historical loan defaults. The machine learning extension compares Logistic Regression and Random Forest and evaluates their ability to distinguish defaulted from non-defaulted loans.

Together, these components demonstrate practical skills in data preparation, analysis, visualization, classification, model evaluation, and communicating findings with appropriate limitations.

## Skills Demonstrated

- SQL data cleaning and aggregation
- MySQL database querying
- Data validation and duplicate detection
- Missing-value handling and type conversion
- Exploratory Data Analysis
- Python and Pandas
- NumPy and Matplotlib
- Feature preprocessing and one-hot encoding
- Logistic Regression and Random Forest
- Classification metrics and threshold selection
- Permutation feature importance
- Power BI dashboard development
- DAX measures and interactive filtering
- Business-oriented interpretation of data
- Git and GitHub project documentation
