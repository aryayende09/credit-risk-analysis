import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt


# ============================================================
# 1. CONNECT TO MYSQL
# ============================================================

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="YOUR_MYSQL_PASSWORD",
    database="credit_risk"
)

print("Connected to MySQL successfully!")


# ============================================================
# 2. LOAD CLEANED DATA
# ============================================================

query = "SELECT * FROM loans_cleaned"

df = pd.read_sql(query, connection)

print("Data loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 3. BASIC DATA INSPECTION
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nLoan grade distribution:")
print(df["loan_grade"].value_counts())

print("\nLoan status distribution:")
print(df["loan_status"].value_counts())


# ============================================================
# 4. OVERALL PROJECT KPIs
# ============================================================

total_loans = len(df)
defaulted_loans = df["loan_status"].sum()
default_rate = df["loan_status"].mean() * 100
average_loan = df["loan_amnt"].mean()
average_interest = df["loan_int_rate"].mean()

print("\nOverall Project KPIs:")
print("Total Loans:", total_loans)
print("Defaulted Loans:", defaulted_loans)
print("Default Rate:", round(default_rate, 2), "%")
print("Average Loan Amount:", round(average_loan, 2))
print("Average Interest Rate:", round(average_interest, 2), "%")


# ============================================================
# 5. RISK ANALYSIS
# ============================================================

# ----- Average loan amount by loan status -----

average_loan_by_status = df.groupby("loan_status")["loan_amnt"].mean()

print("\nAverage Loan Amount by Loan Status:")
print(average_loan_by_status)


# ----- Average loan-to-income ratio by loan status -----

average_ratio_by_status = (
    df.groupby("loan_status")["loan_percent_income"].mean() * 100
)

print("\nAverage Loan-to-Income Ratio by Loan Status:")
print(average_ratio_by_status)


# ----- Average interest rate by loan status -----

average_interest_by_status = df.groupby("loan_status")["loan_int_rate"].mean()

print("\nAverage Interest Rate by Loan Status:")
print(average_interest_by_status)


# ----- Default rate by loan grade -----

default_by_grade = (
    df.groupby("loan_grade")["loan_status"].mean() * 100
)

default_by_grade = default_by_grade.sort_values(ascending=False)

print("\nDefault Rate by Loan Grade:")
print(default_by_grade)


# ----- Default rate by loan intent -----

default_by_intent = (
    df.groupby("loan_intent")["loan_status"].mean() * 100
)

default_by_intent = default_by_intent.sort_values(ascending=False)

print("\nDefault Rate by Loan Intent:")
print(default_by_intent)


# ----- Default rate by home ownership -----

default_by_home = (
    df.groupby("person_home_ownership")["loan_status"].mean() * 100
)

default_by_home = default_by_home.sort_values(ascending=False)

print("\nDefault Rate by Home Ownership:")
print(default_by_home)


# ============================================================
# 6. CREATE ANALYTICAL GROUPS
# ============================================================

# ----- Income groups -----

df["income_group"] = pd.cut(
    df["person_income"],
    bins=[0, 50000, 100000, float("inf")],
    labels=["Low", "Medium", "High"]
)

default_by_income = (
    df.groupby("income_group", observed=True)["loan_status"].mean() * 100
)

default_by_income = default_by_income.sort_values(ascending=False)

print("\nDefault Rate by Income Group:")
print(default_by_income)


# ----- Employment groups -----

df["employment_group"] = pd.cut(
    df["person_emp_length"],
    bins=[-1, 3, 7, float("inf")],
    labels=["Short", "Medium", "Long"]
)

default_by_employment = (
    df.groupby("employment_group", observed=True)["loan_status"].mean() * 100
)

default_by_employment = default_by_employment.sort_values(ascending=False)

print("\nDefault Rate by Employment Length:")
print(default_by_employment)


# ----- Previous default history -----

default_by_history = (
    df.groupby("cb_person_default_on_file")["loan_status"].mean() * 100
)

default_by_history = default_by_history.sort_values(ascending=False)

print("\nDefault Rate by Previous Default History:")
print(default_by_history)


# ----- Credit history groups -----

df["credit_history_group"] = pd.cut(
    df["cb_person_cred_hist_length"],
    bins=[0, 5, 10, float("inf")],
    labels=["Short", "Medium", "Long"]
)

default_by_credit_history = (
    df.groupby("credit_history_group", observed=True)["loan_status"].mean() * 100
)

default_by_credit_history = default_by_credit_history.sort_values(
    ascending=False
)

print("\nDefault Rate by Credit History Length:")
print(default_by_credit_history)


# ----- Age groups -----

df["age_group"] = pd.cut(
    df["person_age"],
    bins=[0, 25, 40, float("inf")],
    labels=["Young", "Adult", "Older"]
)

default_by_age = (
    df.groupby("age_group", observed=True)["loan_status"].mean() * 100
)

default_by_age = default_by_age.sort_values(ascending=False)

print("\nDefault Rate by Age Group:")
print(default_by_age)


# ============================================================
# 7. MATPLOTLIB VISUALIZATIONS
# ============================================================


# ----- 1. Default Rate by Loan Grade -----

plt.bar(default_by_grade.index, default_by_grade.values)

plt.xlabel("Loan Grade")
plt.ylabel("Default Rate (%)")
plt.title("Default Rate by Loan Grade")

for i, value in enumerate(default_by_grade.values):
    plt.text(i, value + 1, f"{value:.1f}%", ha="center")

plt.show()


# ----- 2. Default Rate by Income Group -----

plt.bar(default_by_income.index, default_by_income.values)

plt.xlabel("Income Group")
plt.ylabel("Default Rate (%)")
plt.title("Default Rate by Income Group")

for i, value in enumerate(default_by_income.values):
    plt.text(i, value + 1, f"{value:.1f}%", ha="center")

plt.show()


# ----- 3. Default Rate by Previous Default History -----

plt.bar(default_by_history.index, default_by_history.values)

plt.xlabel("Previous Default on File")
plt.ylabel("Default Rate (%)")
plt.title("Default Rate by Previous Default History")

for i, value in enumerate(default_by_history.values):
    plt.text(i, value + 1, f"{value:.1f}%", ha="center")

plt.show()


# ----- 4. Default Rate by Home Ownership -----

plt.bar(default_by_home.index, default_by_home.values)

plt.xlabel("Home Ownership")
plt.ylabel("Default Rate (%)")
plt.title("Default Rate by Home Ownership")

for i, value in enumerate(default_by_home.values):
    plt.text(i, value + 1, f"{value:.1f}%", ha="center")

plt.show()


# ----- 5. Default Rate by Loan Intent -----

plt.bar(default_by_intent.index, default_by_intent.values)

plt.xlabel("Loan Intent")
plt.ylabel("Default Rate (%)")
plt.title("Default Rate by Loan Intent")
plt.xticks(rotation=30)

for i, value in enumerate(default_by_intent.values):
    plt.text(i, value + 1, f"{value:.1f}%", ha="center")

plt.show()


# ----- 6. Average Loan Amount by Loan Status -----

plt.bar(
    ["Non-default", "Default"],
    average_loan_by_status.values
)

plt.xlabel("Loan Status")
plt.ylabel("Average Loan Amount")
plt.title("Average Loan Amount by Loan Status")

for i, value in enumerate(average_loan_by_status.values):
    plt.text(i, value + 100, f"{value:.2f}", ha="center")

plt.show()


# ----- 7. Average Loan-to-Income Ratio by Loan Status -----

plt.bar(
    ["Non-default", "Default"],
    average_ratio_by_status.values
)

plt.xlabel("Loan Status")
plt.ylabel("Average Loan-to-Income Ratio (%)")
plt.title("Average Loan-to-Income Ratio by Loan Status")

for i, value in enumerate(average_ratio_by_status.values):
    plt.text(i, value + 0.5, f"{value:.2f}%", ha="center")

plt.show()


# ----- 8. Average Interest Rate by Loan Status -----

plt.bar(
    ["Non-default", "Default"],
    average_interest_by_status.values
)

plt.xlabel("Loan Status")
plt.ylabel("Average Interest Rate (%)")
plt.title("Average Interest Rate by Loan Status")

for i, value in enumerate(average_interest_by_status.values):
    plt.text(i, value + 0.2, f"{value:.2f}%", ha="center")

plt.show()


# ============================================================
# 8. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("FINAL PROJECT SUMMARY")
print("=" * 50)

print("Total Loans:", total_loans)
print("Defaulted Loans:", defaulted_loans)
print("Default Rate:", round(default_rate, 2), "%")
print("Average Loan Amount:", round(average_loan, 2))
print("Average Interest Rate:", round(average_interest, 2), "%")

print("\nPython analysis completed successfully!")