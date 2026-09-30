-- ============================================================
-- CREDIT RISK & LOAN DEFAULT ANALYSIS
-- SQL DATA CLEANING, VALIDATION & RISK ANALYSIS
-- ============================================================

USE credit_risk;


-- ============================================================
-- 1. RAW DATA INSPECTION
-- ============================================================

DESCRIBE raw_loans;

SELECT COUNT(*) AS total_rows
FROM raw_loans;


-- ============================================================
-- 2. MISSING VALUE CHECK - RAW DATA
-- ============================================================

SELECT
    SUM(person_age = '') AS age_missing,
    SUM(person_income = '') AS income_missing,
    SUM(person_home_ownership = '') AS home_missing,
    SUM(person_emp_length = '') AS employment_missing,
    SUM(loan_intent = '') AS intent_missing,
    SUM(loan_grade = '') AS grade_missing,
    SUM(loan_amnt = '') AS amount_missing,
    SUM(loan_int_rate = '') AS interest_missing,
    SUM(loan_status = '') AS status_missing,
    SUM(loan_percent_income = '') AS percent_income_missing,
    SUM(cb_person_default_on_file = '') AS previous_default_missing,
    SUM(cb_person_cred_hist_length = '') AS credit_history_missing
FROM raw_loans;


-- ============================================================
-- 3. DUPLICATE RECORD CHECK
-- ============================================================

SELECT 
    COUNT(*) AS total_rows,
    COUNT(DISTINCT CONCAT_WS('|',
        person_age,
        person_income,
        person_home_ownership,
        person_emp_length,
        loan_intent,
        loan_grade,
        loan_amnt,
        loan_int_rate,
        loan_status,
        loan_percent_income,
        cb_person_default_on_file,
        cb_person_cred_hist_length
    )) AS unique_rows
FROM raw_loans;


SELECT
    person_age,
    person_income,
    person_home_ownership,
    person_emp_length,
    loan_intent,
    loan_grade,
    loan_amnt,
    loan_int_rate,
    loan_status,
    loan_percent_income,
    cb_person_default_on_file,
    cb_person_cred_hist_length,
    COUNT(*) AS duplicate_count
FROM raw_loans
GROUP BY
    person_age,
    person_income,
    person_home_ownership,
    person_emp_length,
    loan_intent,
    loan_grade,
    loan_amnt,
    loan_int_rate,
    loan_status,
    loan_percent_income,
    cb_person_default_on_file,
    cb_person_cred_hist_length
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC;


-- ============================================================
-- 4. DATA QUALITY CHECKS
-- ============================================================

-- Check for unrealistic ages

SELECT
    person_age,
    COUNT(*) AS total_records
FROM raw_loans
WHERE CAST(person_age AS UNSIGNED) > 100
GROUP BY person_age
ORDER BY person_age;


-- Check income range

SELECT
    MIN(CAST(person_income AS DECIMAL(15,2))) AS minimum_income,
    MAX(CAST(person_income AS DECIMAL(15,2))) AS maximum_income
FROM raw_loans;


-- ============================================================
-- 5. CREATE CLEANED DATASET
-- ============================================================

CREATE TABLE loans_cleaned AS
SELECT DISTINCT *
FROM raw_loans
WHERE CAST(person_age AS UNSIGNED) <= 100;


-- ============================================================
-- 6. ADD UNIQUE LOAN ID
-- ============================================================

ALTER TABLE loans_cleaned
ADD COLUMN loan_id INT AUTO_INCREMENT PRIMARY KEY FIRST;


-- ============================================================
-- 7. MISSING VALUE TREATMENT
-- ============================================================

SET SQL_SAFE_UPDATES = 0;


-- Replace missing interest rates with median value

UPDATE loans_cleaned
SET loan_int_rate = '10.99'
WHERE loan_int_rate = '';


-- Replace missing employment length with median value

UPDATE loans_cleaned
SET person_emp_length = '4'
WHERE person_emp_length = '';


-- Verify missing values were handled

SELECT
    SUM(person_emp_length = '') AS missing_emp_length,
    SUM(loan_int_rate = '') AS missing_interest_rate
FROM loans_cleaned;


-- ============================================================
-- 8. DATA TYPE CONVERSION
-- ============================================================

DESCRIBE loans_cleaned;


ALTER TABLE loans_cleaned
MODIFY person_age INT;

ALTER TABLE loans_cleaned
MODIFY person_income DECIMAL(15,2);

ALTER TABLE loans_cleaned
MODIFY person_emp_length DECIMAL(10,2);

ALTER TABLE loans_cleaned
MODIFY loan_amnt DECIMAL(15,2);

ALTER TABLE loans_cleaned
MODIFY loan_int_rate DECIMAL(10,2);

ALTER TABLE loans_cleaned
MODIFY loan_status INT;

ALTER TABLE loans_cleaned
MODIFY loan_percent_income DECIMAL(10,4);

ALTER TABLE loans_cleaned
MODIFY cb_person_cred_hist_length INT;


-- ============================================================
-- 9. FINAL DATA VALIDATION
-- ============================================================

DESCRIBE loans_cleaned;


SELECT
    COUNT(*) AS total_rows,
    MIN(person_age) AS min_age,
    MAX(person_age) AS max_age,
    MIN(person_income) AS min_income,
    MIN(loan_amnt) AS min_loan_amount,
    MIN(loan_int_rate) AS min_interest_rate,
    MIN(loan_percent_income) AS min_loan_percent_income
FROM loans_cleaned;


-- ============================================================
-- 10. OVERALL DEFAULT RISK
-- ============================================================

SELECT
    COUNT(*) AS total_loans,
    SUM(loan_status = 1) AS defaulted_loans,
    ROUND(
        SUM(loan_status = 1) * 100.0 / COUNT(*),
        2
    ) AS default_rate
FROM loans_cleaned;


-- ============================================================
-- 11. DEFAULT RISK BY LOAN GRADE
-- ============================================================

SELECT
    loan_grade,
    COUNT(*) AS total_loans,
    SUM(loan_status = 1) AS defaulted_loans,
    ROUND(
        SUM(loan_status = 1) * 100.0 / COUNT(*),
        2
    ) AS default_rate,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
    ROUND(AVG(loan_int_rate), 2) AS avg_interest_rate
FROM loans_cleaned
GROUP BY loan_grade
ORDER BY default_rate DESC;


-- ============================================================
-- 12. DEFAULT RISK BY LOAN INTENT
-- ============================================================

SELECT
    loan_intent,
    COUNT(*) AS total_loans,
    SUM(loan_status = 1) AS defaulted_loans,
    ROUND(
        SUM(loan_status = 1) * 100.0 / COUNT(*),
        2
    ) AS default_rate
FROM loans_cleaned
GROUP BY loan_intent
ORDER BY default_rate DESC;


-- ============================================================
-- 13. DEFAULT RISK BY HOME OWNERSHIP
-- ============================================================

SELECT
    person_home_ownership,
    COUNT(*) AS total_loans,
    SUM(loan_status = 1) AS defaulted_loans,
    ROUND(
        SUM(loan_status = 1) * 100.0 / COUNT(*),
        2
    ) AS default_rate
FROM loans_cleaned
GROUP BY person_home_ownership
ORDER BY default_rate DESC;


-- ============================================================
-- 14. INTEREST RATE BY LOAN STATUS
-- ============================================================

SELECT
    loan_status,
    COUNT(*) AS total_loans,
    ROUND(AVG(loan_int_rate), 2) AS average_interest_rate
FROM loans_cleaned
GROUP BY loan_status;


-- ============================================================
-- 15. INCOME GROUP RISK ANALYSIS
-- ============================================================

SELECT
    CASE
        WHEN person_income < 50000 THEN 'Low Income'
        WHEN person_income < 100000 THEN 'Medium Income'
        ELSE 'High Income'
    END AS income_group,
    COUNT(*) AS total_loans,
    SUM(loan_status = 1) AS defaulted_loans,
    ROUND(
        SUM(loan_status = 1) * 100.0 / COUNT(*),
        2
    ) AS default_rate,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
    ROUND(AVG(loan_percent_income) * 100, 2) AS avg_loan_income_ratio
FROM loans_cleaned
GROUP BY income_group
ORDER BY default_rate DESC;


-- ============================================================
-- 16. AVERAGE LOAN AMOUNT BY LOAN STATUS
-- ============================================================

SELECT
    loan_status,
    COUNT(*) AS total_loans,
    ROUND(AVG(loan_amnt), 2) AS average_loan_amount
FROM loans_cleaned
GROUP BY loan_status;


-- ============================================================
-- 17. LOAN-TO-INCOME RATIO BY LOAN STATUS
-- ============================================================

SELECT
    loan_status,
    ROUND(AVG(loan_percent_income) * 100, 2) AS average_loan_income_ratio
FROM loans_cleaned
GROUP BY loan_status;


-- ============================================================
-- 18. EMPLOYMENT LENGTH RISK
-- ============================================================

SELECT
    CASE
        WHEN person_emp_length <= 3 THEN 'Short'
        WHEN person_emp_length <= 7 THEN 'Medium'
        ELSE 'Long'
    END AS employment_group,
    COUNT(*) AS total_loans,
    SUM(loan_status = 1) AS defaulted_loans,
    ROUND(
        SUM(loan_status = 1) * 100.0 / COUNT(*),
        2
    ) AS default_rate
FROM loans_cleaned
GROUP BY employment_group
ORDER BY default_rate DESC;


-- ============================================================
-- 19. AGE GROUP RISK
-- ============================================================

SELECT
    CASE
        WHEN person_age <= 25 THEN 'Young'
        WHEN person_age <= 40 THEN 'Adult'
        ELSE 'Older'
    END AS age_group,
    COUNT(*) AS total_loans,
    SUM(loan_status = 1) AS defaulted_loans,
    ROUND(
        SUM(loan_status = 1) * 100.0 / COUNT(*),
        2
    ) AS default_rate
FROM loans_cleaned
GROUP BY age_group
ORDER BY default_rate DESC;


-- ============================================================
-- 20. PREVIOUS DEFAULT HISTORY
-- ============================================================

SELECT
    cb_person_default_on_file AS previous_default,
    COUNT(*) AS total_loans,
    SUM(loan_status = 1) AS defaulted_loans,
    ROUND(
        SUM(loan_status = 1) * 100.0 / COUNT(*),
        2
    ) AS default_rate
FROM loans_cleaned
GROUP BY cb_person_default_on_file
ORDER BY default_rate DESC;


-- ============================================================
-- 21. CREDIT HISTORY LENGTH
-- ============================================================

SELECT
    CASE
        WHEN cb_person_cred_hist_length <= 5 THEN 'Short'
        WHEN cb_person_cred_hist_length <= 10 THEN 'Medium'
        ELSE 'Long'
    END AS credit_history_group,
    COUNT(*) AS total_loans,
    SUM(loan_status = 1) AS defaulted_loans,
    ROUND(
        SUM(loan_status = 1) * 100.0 / COUNT(*),
        2
    ) AS default_rate
FROM loans_cleaned
GROUP BY credit_history_group
ORDER BY default_rate DESC;


-- ============================================================
-- 22. DEFAULTED LOAN EXPOSURE
-- ============================================================

SELECT
    COUNT(*) AS defaulted_loans,
    ROUND(SUM(loan_amnt), 2) AS total_defaulted_amount,
    ROUND(AVG(loan_amnt), 2) AS average_defaulted_loan
FROM loans_cleaned
WHERE loan_status = 1;


-- ============================================================
-- 23. OVERALL RISK SUMMARY VIEW
-- ============================================================

CREATE VIEW overall_risk_summary AS
SELECT
    COUNT(*) AS total_loans,
    SUM(loan_status = 1) AS defaulted_loans,
    ROUND(
        SUM(loan_status = 1) * 100.0 / COUNT(*),
        2
    ) AS default_rate,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
    ROUND(AVG(loan_int_rate), 2) AS avg_interest_rate,
    ROUND(AVG(loan_percent_income) * 100, 2) AS avg_loan_income_ratio,
    ROUND(
        SUM(
            CASE
                WHEN loan_status = 1 THEN loan_amnt
                ELSE 0
            END
        ),
        2
    ) AS total_defaulted_amount
FROM loans_cleaned;


-- ============================================================
-- 24. GRADE RISK SUMMARY VIEW
-- ============================================================

CREATE VIEW grade_risk_summary AS
SELECT
    loan_grade,
    COUNT(*) AS total_loans,
    SUM(loan_status = 1) AS defaulted_loans,
    ROUND(
        SUM(loan_status = 1) * 100.0 / COUNT(*),
        2
    ) AS default_rate,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
    ROUND(AVG(loan_int_rate), 2) AS avg_interest_rate
FROM loans_cleaned
GROUP BY loan_grade
ORDER BY default_rate DESC;


-- ============================================================
-- 25. INCOME RISK SUMMARY VIEW
-- ============================================================

CREATE VIEW income_risk_summary AS
SELECT
    CASE
        WHEN person_income < 50000 THEN 'Low Income'
        WHEN person_income < 100000 THEN 'Medium Income'
        ELSE 'High Income'
    END AS income_group,
    COUNT(*) AS total_loans,
    SUM(loan_status = 1) AS defaulted_loans,
    ROUND(
        SUM(loan_status = 1) * 100.0 / COUNT(*),
        2
    ) AS default_rate,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
    ROUND(AVG(loan_percent_income) * 100, 2) AS avg_loan_income_ratio
FROM loans_cleaned
GROUP BY income_group
ORDER BY default_rate DESC;


-- ============================================================
-- END OF SQL ANALYSIS
-- ============================================================