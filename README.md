# Week 7: Introduction to Statistics for Data Science
## Project: Statistical Business Analysis

A documented Python project for Developer Arena Week 7. It applies descriptive statistics, distribution analysis, correlation and covariance, hypothesis testing, confidence intervals and regression to the supplied sales dataset, with a separate supplementary customer-churn analysis.

### Project deliverables
- `main.py` — runnable end-to-end analysis script
- `statistical_analysis.ipynb` — beginner-friendly stepwise notebook
- `data/sales_data.csv` — supplied sales dataset (100 rows)
- `data/customer_churn.csv` — supplied customer churn dataset (500 rows)
- `outputs/` — descriptive statistics, correlation/covariance matrices, hypothesis test results, churn summary and crosstab
- `visualizations/` — histogram, regional boxplot, product comparison, correlation heatmap and churn chart
- `reports/analysis_report.pdf` — written statistical report with visuals
- `reports/analysis_documentation.docx` — editable project documentation
- `requirements.txt` — Python dependencies

### Install and run (VS Code)
1. Extract this project ZIP and open the folder in VS Code.
2. Select a Python 3 interpreter.
3. Terminal → New Terminal:
   ```powershell
   python -m pip install -r requirements.txt
   python main.py
   ```
4. View generated CSV files in `outputs/` and PNG charts in `visualizations/`.

### Week 7 topics implemented
- Descriptive statistics: mean, median, mode, standard deviation and variance
- Distribution analysis using histograms and Shapiro–Wilk normality test
- Pearson correlation and covariance
- Hypothesis testing using one-way ANOVA, Welch's t-test and chi-square
- Statistical significance and p-values at α = 0.05
- 95% confidence intervals and margin of error
- Linear regression with R-squared and RMSE
- Business interpretation and recommendations

### Dataset limitations
- The sales dataset does not contain a marketing-spend field. Therefore, marketing spend versus revenue cannot be analyzed from the supplied data. Quantity, price and total sales are used for correlation instead.
- `Total_Sales` is constructed as `Quantity × Price`. Regression using these values reconstructs an existing arithmetic relationship; it is not an independent predictive model.
- The churn dataset is analyzed separately. Customer IDs across the sales and churn files are not verified as matching identities.
- Sales value is not profit because cost, discounts and returns are not supplied.

### Run and interpret
Use a 5% significance level. A p-value below 0.05 is treated as statistically significant for the specified test. A p-value is not the probability that the null hypothesis is true.
