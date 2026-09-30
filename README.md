# Credit Risk & Loan Default Analysis

## Project Overview

This project analyzes loan and borrower data to identify patterns associated with loan defaults.

The project uses SQL, Python, and Power BI to perform data cleaning, exploratory analysis, risk analysis, and dashboard development.

The main objective is to understand how factors such as loan grade, income, loan purpose, home ownership, previous default history, interest rate, and loan-to-income ratio are associated with loan default risk.

---

## Dataset

The dataset used for this project is the Credit Risk Dataset from Kaggle.

Source: Kaggle  
Original Records: 32,581  
Original Columns: 12  
Target Variable: loan_status

In the dataset:

- 0 represents a non-default
- 1 represents a default

### Main Variables

- person_age — Borrower's age
- person_income — Annual income
- person_home_ownership — Home ownership status
- person_emp_length — Employment length
- loan_intent — Purpose of the loan
- loan_grade — Loan risk grade
- loan_amnt — Loan amount
- loan_int_rate — Interest rate
- loan_status — Default status
- loan_percent_income — Loan amount as a percentage of income
- cb_person_default_on_file — Previous default history
- cb_person_cred_hist_length — Length of credit history

---

## Tools and Technologies

- MySQL
- Python
- Pandas
- NumPy
- Matplotlib
- Power BI
- DAX
- GitHub

---

## Project Workflow

Raw Dataset  
↓  
MySQL Data Import  
↓  
Data Cleaning and Validation  
↓  
SQL Risk Analysis  
↓  
Python Exploratory Data Analysis  
↓  
Power BI Dashboard  
↓  
Insights and Findings

---

## Data Cleaning Using SQL

The raw dataset was first imported into MySQL without modifying the original data.

The following data cleaning and validation steps were performed:

- Checked the total number of records
- Checked missing values
- Checked duplicate records
- Identified unrealistic age values
- Removed duplicate records
- Removed records with age values above 100
- Added a unique loan ID
- Handled missing employment length
- Handled missing interest rate values
- Converted columns from text format to appropriate numeric data types
- Performed final validation checks

The cleaned dataset was stored in the MySQL table named loans_cleaned.

---

## Cleaning Result

After cleaning and validation:

- Original records: 32,581
- Final cleaned records: 32,411
- Original columns: 12
- Final columns: 13
- Missing employment length values after cleaning: 0
- Missing interest rate values after cleaning: 0
- Age range after cleaning: 20–94

The cleaning process removed duplicate records and five records with unrealistic age values above 100.

Missing employment length and interest rate values were handled using median values calculated from the dataset.

---

## SQL Analysis

SQL was used to analyze loan default patterns across different borrower and loan characteristics.

The analysis included:

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

Three SQL summary views were also created for reusable analysis:

- Overall Risk Summary
- Grade Risk Summary
- Income Risk Summary

---

## Python Analysis

Python was used for exploratory data analysis after the SQL cleaning process.

The cleaned dataset was loaded from MySQL into Pandas and analyzed using grouping, aggregation, and visualization techniques.

### Analysis Performed

- Dataset structure and data types
- Missing value verification
- Default distribution
- Overall default rate
- Loan grade analysis
- Loan intent analysis
- Income group analysis
- Employment group analysis
- Previous default analysis
- Credit history analysis
- Age group analysis
- Loan amount analysis
- Interest rate analysis
- Loan-to-income analysis

### Visualizations Created

The Python analysis included:

1. Default Rate by Loan Grade
2. Default Rate by Income Group
3. Default Rate by Previous Default History
4. Default Rate by Home Ownership
5. Default Rate by Loan Intent
6. Average Loan Amount by Loan Status
7. Average Loan-to-Income Ratio by Loan Status
8. Average Interest Rate by Loan Status

---

## Power BI Dashboard

The final interactive dashboard was created using Power BI.

### Page 1 — Credit Risk Overview

The overview dashboard contains:

- Total Loans
- Total Amount at Risk
- Default Rate
- Average Interest Rate
- Portfolio by Home Ownership
- Default Rate by Loan Grade
- Grade Summary

### Page 2 — Risk Analysis

The risk analysis dashboard contains:

- Loan Grade slicer
- Loan Intent slicer
- Home Ownership slicer
- Income Group slicer
- Loan Intent Breakdown
- Loan-to-Income Ratio vs Default scatter plot
- Average Loan Amount by Income Group

The dashboard allows users to interact with the data using slicers and examine different risk segments.

---

## Key Findings

### Overall Risk

- Total Loans: 32,411
- Defaulted Loans: 7,089
- Overall Default Rate: 21.87%
- Average Loan Amount: 9,593.59
- Average Interest Rate: 11.01%
- Total Defaulted Loan Exposure: 76,968,675

### Default Rate by Loan Grade

| Loan Grade | Default Rate |
|------------|--------------|
| A | 9.96% |
| B | 16.32% |
| C | 20.76% |
| D | 59.06% |
| E | 64.49% |
| F | 70.54% |
| G | 98.44% |

Grade G contains a relatively small number of loans, so its default rate should be interpreted with caution.

### Default Rate by Income Group

| Income Group | Default Rate |
|--------------|--------------|
| Low | 33.44% |
| Medium | 15.43% |
| High | 8.95% |

The analysis shows an association between income group and observed default rate. This does not mean that income alone causes loan default.

### Previous Default History

| Previous Default on File | Default Rate |
|--------------------------|--------------|
| Yes | 37.87% |
| No | 18.44% |

Borrowers with a previous default recorded had a higher observed default rate in this dataset.

### Loan-to-Income Ratio

| Loan Status | Average Loan-to-Income Ratio |
|-------------|------------------------------|
| Non-default | 14.88% |
| Default | 24.69% |

Defaulted loans had a higher average loan-to-income ratio in this dataset.

### Interest Rate

| Loan Status | Average Interest Rate |
|-------------|-----------------------|
| Non-default | 10.49% |
| Default | 12.88% |

### Average Loan Amount

| Loan Status | Average Loan Amount |
|-------------|---------------------|
| Non-default | 9,239.76 |
| Default | 10,857.48 |

---

## Important Interpretation Note

The findings in this project describe associations within the dataset.

They should not be interpreted as proof that a particular borrower characteristic directly causes loan default.

Some categories also contain relatively small numbers of observations, so their default rates should be interpreted carefully.

---

## Project Structure

Credit-Risk-Analysis

├── README.md  
├── SQL  
│   └── data_cleaning_and_analysis.sql  
├── Python  
│   └── credit_risk_analysis.py  
├── PowerBI  
│   └── Credit_Risk_Analysis.pbix  
└── screenshots

---

## Conclusion

This project demonstrates an end-to-end data analysis workflow using SQL, Python, and Power BI.

The analysis begins with raw loan data, cleans and validates the dataset using SQL, performs exploratory analysis using Python, and presents the final findings through an interactive Power BI dashboard.

The project helped identify patterns in loan default behavior across loan grade, income, previous default history, loan intent, home ownership, interest rate, and loan-to-income ratio.

---

## Skills Demonstrated

- SQL data cleaning
- SQL aggregation and analysis
- Data validation
- Missing value handling
- Duplicate detection
- Data type conversion
- Exploratory Data Analysis
- Pandas
- NumPy
- Matplotlib
- Data visualization
- Power BI dashboard development
- DAX measures
- Interactive filtering
- Business-oriented data interpretation