# CreditLens — Credit Risk & Loan Default Analysis

An end-to-end data analytics and machine learning project that explores loan default patterns, evaluates classification models, and estimates loan default probability through an interactive Streamlit application.

## Project Overview

CreditLens analyzes borrower information to understand patterns associated with loan defaults. The project combines SQL-based data cleaning and analysis, Python-based exploratory data analysis, machine learning, and an interactive web application.

The objective is to demonstrate how data can be processed, analyzed, visualized, and used to build a predictive model.

**Project repository:** [Credit Risk Analysis on GitHub](https://github.com/aryayende09/credit-risk-analysis)

**Live demo:** Coming soon — the application will be linked here after successful deployment.

## Project Objectives

- Clean and prepare loan applicant data using SQL.
- Explore borrower characteristics and loan default patterns.
- Compare Logistic Regression and Random Forest classification models.
- Evaluate model performance using classification metrics.
- Identify important features associated with model predictions.
- Build an interactive Streamlit application for applicant-level risk assessment.
- Maintain assessment history and export results to CSV.

## Tech Stack

| Technology | Purpose |
|---|---|
| MySQL | Data cleaning, transformation, and SQL analysis |
| Python | Data processing and machine learning |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Scikit-learn | Model training and evaluation |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web application |
| Power BI | Interactive business intelligence dashboard |
| Git and GitHub | Version control and project hosting |

## Dataset

The project uses the Credit Risk Dataset available on Kaggle.

**Dataset source:** [Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

The original dataset contains 32,581 records and 12 columns describing borrower characteristics, loan details, and default status.

### Important Features

| Feature | Description |
|---|---|
| `person_age` | Borrower's age |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership category |
| `person_emp_length` | Employment length in years |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade assigned in the source dataset |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_status` | Target variable indicating default status |
| `loan_percent_income` | Loan amount relative to annual income |
| `cb_person_default_on_file` | Historical default indicator |
| `cb_person_cred_hist_length` | Length of credit history |

The target variable is `loan_status`:

- `0` — No default
- `1` — Default

## Project Workflow

### 1. SQL Data Cleaning and Analysis

MySQL is used to inspect and prepare the dataset before analysis.

The workflow includes:

- Inspecting the raw dataset and column structure.
- Identifying missing values and duplicate records.
- Investigating suspicious values and potential data-quality issues.
- Preparing a cleaned dataset for downstream analysis.
- Performing SQL queries to explore loan and borrower patterns.

The original dataset contains 32,581 records. After cleaning, the analysis dataset contains 32,411 records.

### 2. Exploratory Data Analysis

Python, Pandas, and Matplotlib are used to explore the cleaned data and understand its characteristics.

The analysis examines:

- The distribution of default and non-default cases.
- Borrower income, age, and employment characteristics.
- Loan amounts and loan-to-income ratios.
- Relationships between applicant characteristics and default status.
- Patterns that may help explain the model's predictions.

### 3. Machine Learning

Two classification models were trained and compared:

- Logistic Regression
- Random Forest Classifier

The dataset was divided into training, validation, and test subsets using a 60/20/20 split and `random_state=42`.

The validation data was used for model-selection and threshold-selection decisions. The reported test metrics are from the project experiments; the test data has already been inspected during development.

### 4. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Confusion Matrix

The project uses a classification threshold to convert the model's estimated probability into a predicted class.

### 5. CreditLens Streamlit Application

The Streamlit app loads the saved Random Forest model and provides an interactive interface for assessing applicant profiles.

App features include:

- **Risk Assessment:** Enter applicant details and obtain an estimated default probability.
- **Loan-to-Income Calculation:** Automatically calculate the loan amount as a proportion of annual income.
- **Default Classification:** Display the model's predicted class using the selected classification threshold.
- **Previous Applicant Assessments:** View assessments recorded during the active application session.
- **CSV Export:** Download assessment records for further analysis.
- **Model Insights:** Explore global permutation feature importance.
- **Input Validation:** Check for inconsistent employment and credit-history lengths.

The app is intended as a portfolio demonstration of a machine-learning workflow, not as a production lending system.

### 6. Power BI Dashboard

A Power BI dashboard is included to present the project's credit-risk analysis visually.

The dashboard file is located at:

`Power BI/Credit_Risk_Analysis.pbix`

Open the file using Power BI Desktop to explore the dashboard.

## Model Results

The following are the reported Random Forest results from the project's test evaluation.

| Metric | Random Forest |
|---|---:|
| Accuracy | 86.41% |
| Precision | 71.67% |
| Recall | 62.62% |
| F1-score | 66.84% |
| ROC-AUC | 85.89% |
| Average Precision | 76.40% |

The Random Forest model achieved a higher F1-score and ROC-AUC than Logistic Regression in the recorded experiments.

These results indicate that the model distinguishes between default and non-default cases, but its predictions are not perfect.

**Important:** These metrics are specific to the dataset and evaluation procedure used in this project. They do not guarantee performance on new applicants or real-world lending data.

## Feature Importance

Permutation feature importance was calculated using the validation data to understand which features most affected Random Forest performance.

The leading features in the recorded results were:

| Feature | Permutation Importance |
|---|---:|
| `loan_percent_income` | 0.28103 |
| `person_home_ownership` | 0.22929 |
| `person_income` | 0.10634 |
| `cb_person_default_on_file` | 0.06576 |
| `loan_intent` | 0.06008 |

The results suggest that loan-to-income ratio, home ownership, and income were particularly influential in the model's performance.

Permutation importance describes global model behavior. It does not explain every individual prediction, establish causation, or prove that a feature independently causes default.

## Repository Structure

```text
credit-risk-analysis/
├── ML/
│   ├── credit_risk_model.py
│   ├── models/
│   │   ├── logistic_regression.joblib
│   │   └── random_forest.joblib
│   └── outputs/
│       ├── logistic_regression_confusion_matrix.csv
│       ├── model_comparison.csv
│       ├── random_forest_confusion_matrix.csv
│       ├── random_forest_feature_importance.csv
│       └── validation_threshold_results.csv
├── Power BI/
│   └── Credit_Risk_Analysis.pbix
├── Python/
│   └── credit_risk_analysis.py
├── Screenshots/
│   ├── 1st Dashboard.png
│   └── 2nd Dashboard.png
├── SQL/
│   └── data_cleaning_and_analysis.sql
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

The structure above describes the intended project layout. Some files or folders may be absent from a fresh GitHub checkout if they have not been committed.

## Installation and Setup

### Prerequisites

Install the following:

- Python 3.10 or a compatible version for the project's dependencies.
- MySQL Server and MySQL Workbench for the SQL workflow.
- Git, if you want to clone the repository.
- Power BI Desktop to open the dashboard.

### 1. Clone the Repository

```bash
git clone https://github.com/aryayende09/credit-risk-analysis.git
cd credit-risk-analysis
```

### 2. Install Python Dependencies

Create a virtual environment if desired, then install the packages listed in `requirements.txt`.

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file must be included in the repository before using this installation command.

### 3. Run the Streamlit Application

From the project root, run:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

The application loads the saved Random Forest model from:

`ML/models/random_forest.joblib`

The Model Insights tab reads the saved feature-importance output from:

`ML/outputs/random_forest_feature_importance.csv`

The saved model and its required files must be present for the application to run successfully.

### 4. Run the Machine Learning Script

To retrain and evaluate the models, run:

```bash
python ML/credit_risk_model.py
```

Review the script before running it to confirm that the database configuration, data source, and output paths match your local setup.

The script may require access to the cleaned dataset or MySQL database used during development. Do not include database passwords or other credentials in the repository.

## Limitations

- The dataset is historical and may not represent current lending populations.
- The model may make incorrect predictions.
- The default threshold is a project setting and does not represent a real lender's policy.
- Estimated probabilities are model outputs, not guarantees of default or repayment.
- Feature importance describes model behavior, not causation.
- Applicant assessment history is stored in the active Streamlit session and is not a permanent database.
- The application has not been established as a production-grade or regulatory-compliant lending system.
- Real lending decisions require appropriate validation, fairness testing, governance, privacy safeguards, and human oversight.

## Future Improvements

Potential future improvements include:

- Deploying CreditLens as a public web application.
- Adding a dedicated database for persistent assessment history.
- Improving model monitoring and evaluation.
- Adding individual prediction explanations after learning appropriate interpretability methods.
- Improving chart readability and adding further analytical insights.
- Testing model performance and fairness on additional datasets.

## Author

**Arya Yende**

- GitHub: [aryayende09](https://github.com/aryayende09)
- Project repository: [Credit Risk Analysis](https://github.com/aryayende09/credit-risk-analysis)

---

*CreditLens is an educational portfolio project demonstrating data analytics, SQL, machine learning, model evaluation, and application development.*
