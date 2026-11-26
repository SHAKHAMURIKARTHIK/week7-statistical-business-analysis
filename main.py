"""
Week 7: Introduction to Statistics for Data Science
Project: Statistical Business Analysis
Run: python main.py
"""
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = BASE_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

ALPHA = 0.05

def load_data():
    sales = pd.read_csv(DATA_DIR / "sales_data.csv")
    churn = pd.read_csv(DATA_DIR / "customer_churn.csv")
    sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
    for col in ["Product", "Region", "Customer_ID"]:
        sales[col] = sales[col].astype("string").str.strip()
    for col in ["Quantity", "Price", "Total_Sales"]:
        sales[col] = pd.to_numeric(sales[col], errors="coerce")
    sales = sales.dropna(subset=["Date", "Product", "Region", "Customer_ID",
                                 "Quantity", "Price", "Total_Sales"]).copy()
    sales = sales[(sales["Quantity"] > 0) & (sales["Price"] >= 0) &
                  (sales["Total_Sales"] >= 0)].drop_duplicates()
    churn["CustomerID"] = churn["CustomerID"].astype("string").str.strip()
    return sales, churn

def save_fig(fig, filename):
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)

def main():
    sales, churn = load_data()
    print("\\n========== WEEK 7: STATISTICAL BUSINESS ANALYSIS ==========")
    print(f"Sales records: {len(sales)} | Churn records: {len(churn)}")
    print("\\nMissing values (sales):\\n", sales.isna().sum())
    print("\\nMissing values (churn):\\n", churn.isna().sum())

    # 1. Descriptive statistics
    metrics = ["Quantity", "Price", "Total_Sales"]
    desc = sales[metrics].agg(["count", "mean", "median", "std", "var", "min", "max"]).T
    desc["mode"] = sales[metrics].mode().iloc[0]
    desc.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")
    sales_summary = {
        "records": len(sales),
        "total_sales": sales["Total_Sales"].sum(),
        "mean_sales": sales["Total_Sales"].mean(),
        "median_sales": sales["Total_Sales"].median(),
        "mode_sales": sales["Total_Sales"].mode().iloc[0],
        "std_sales": sales["Total_Sales"].std()
    }

    # 2. Distribution and normality
    shapiro = stats.shapiro(sales["Total_Sales"])
    region_groups = [g["Total_Sales"].values for _, g in sales.groupby("Region")]
    product_groups = [g["Total_Sales"].values for _, g in sales.groupby("Product")]
    region_anova = stats.f_oneway(*region_groups)
    product_anova = stats.f_oneway(*product_groups)

    # 3. Correlation and covariance
    corr = sales[metrics].corr(method="pearson")
    cov = sales[metrics].cov()
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    cov.to_csv(OUTPUT_DIR / "covariance_matrix.csv")
    corr_sales = stats.pearsonr(sales["Quantity"], sales["Total_Sales"])
    corr_price = stats.pearsonr(sales["Price"], sales["Total_Sales"])

    # 4. Hypothesis tests
    east = sales.loc[sales["Region"].str.casefold() == "east", "Total_Sales"]
    west = sales.loc[sales["Region"].str.casefold() == "west", "Total_Sales"]
    east_west = stats.ttest_ind(east, west, equal_var=False)
    contract_table = pd.crosstab(churn["Contract"], churn["Churn"])
    chi_contract = stats.chi2_contingency(contract_table)
    contract_table.to_csv(OUTPUT_DIR / "contract_churn_crosstab.csv")

    # 5. Confidence intervals
    ci = stats.t.interval(0.95, len(sales)-1, loc=sales["Total_Sales"].mean(),
                          scale=stats.sem(sales["Total_Sales"]))
    churn_count = int(churn["Churn"].sum())
    churn_ci = stats.binomtest(churn_count, len(churn)).proportion_ci(
        confidence_level=0.95, method="wilson")

    # 6. Regression. Target is constructed as Quantity * Price; label this honestly.
    X = sales[["Quantity", "Price"]].copy()
    X["Quantity_x_Price"] = X["Quantity"] * X["Price"]
    y = sales["Total_Sales"]
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    regression = {
        "intercept": model.intercept_,
        "r_squared": r2_score(y, pred),
        "rmse": mean_squared_error(y, pred) ** 0.5
    }

    # 7. Churn supplementary analysis
    tenure_test = stats.ttest_ind(
        churn.loc[churn["Churn"] == 1, "Tenure"],
        churn.loc[churn["Churn"] == 0, "Tenure"],
        equal_var=False
    )
    churn_summary = churn.groupby("Churn", as_index=False).agg(
        Customers=("CustomerID", "count"),
        Average_Tenure=("Tenure", "mean"),
        Average_Monthly_Charges=("MonthlyCharges", "mean"),
        Average_Total_Charges=("TotalCharges", "mean")
    )
    churn_summary.to_csv(OUTPUT_DIR / "churn_summary.csv", index=False)

    tests = pd.DataFrame([
        ["Sales by region (one-way ANOVA)", region_anova.statistic, region_anova.pvalue,
         "Reject H0" if region_anova.pvalue < ALPHA else "Fail to reject H0"],
        ["Sales by product (one-way ANOVA)", product_anova.statistic, product_anova.pvalue,
         "Reject H0" if product_anova.pvalue < ALPHA else "Fail to reject H0"],
        ["East vs West sales (Welch t-test)", east_west.statistic, east_west.pvalue,
         "Reject H0" if east_west.pvalue < ALPHA else "Fail to reject H0"],
        ["Contract vs churn (chi-square)", chi_contract.statistic, chi_contract.pvalue,
         "Reject H0" if chi_contract.pvalue < ALPHA else "Fail to reject H0"],
    ], columns=["Test", "Statistic", "P_Value", "Decision"])
    tests.to_csv(OUTPUT_DIR / "hypothesis_test_results.csv", index=False)

    # Figures
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sales["Total_Sales"], bins=15, edgecolor="black")
    ax.set(title="Distribution of Transaction Sales", xlabel="Total Sales", ylabel="Frequency")
    save_fig(fig, "sales_histogram.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sales.boxplot(column="Total_Sales", by="Region", ax=ax)
    ax.set_title("Transaction Sales by Region"); fig.suptitle("")
    ax.set_xlabel("Region"); ax.set_ylabel("Total Sales")
    save_fig(fig, "sales_by_region_boxplot.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sales.groupby("Product")["Total_Sales"].mean().sort_values().plot(kind="bar", ax=ax)
    ax.set(title="Average Transaction Sales by Product", xlabel="Product", ylabel="Average Sales")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, "sales_by_product.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(corr.index)), labels=corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
    ax.set_title("Pearson Correlation Heatmap")
    fig.colorbar(im, ax=ax, label="Correlation")
    save_fig(fig, "correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    (churn.groupby("Contract")["Churn"].mean() * 100).plot(kind="bar", ax=ax)
    ax.set(title="Churn Rate by Contract Type", xlabel="Contract", ylabel="Churn Rate (%)")
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, "churn_by_contract.png")

    # Plain-text findings
    report = f"""WEEK 7: STATISTICAL BUSINESS ANALYSIS
Significance level: alpha = {ALPHA}

SALES DATA
Records: {len(sales)}
Total sales: {sales_summary['total_sales']:,.2f}
Mean transaction sales: {sales_summary['mean_sales']:,.2f}
Median transaction sales: {sales_summary['median_sales']:,.2f}
Mode transaction sales: {sales_summary['mode_sales']:,.2f}
Sample standard deviation: {sales_summary['std_sales']:,.2f}
95% confidence interval for mean sales: [{ci[0]:,.2f}, {ci[1]:,.2f}]
Margin of error: {(ci[1]-ci[0])/2:,.2f}
Shapiro-Wilk normality test: statistic={shapiro.statistic:.4f}, p={shapiro.pvalue:.6f}

CORRELATION
Pearson r (Quantity, Total_Sales): {corr_sales.statistic:.4f}, p={corr_sales.pvalue:.6f}
Pearson r (Price, Total_Sales): {corr_price.statistic:.4f}, p={corr_price.pvalue:.6f}

HYPOTHESIS TESTS (alpha={ALPHA})
{tests.to_string(index=False)}

CUSTOMER CHURN
Observed churn rate: {churn['Churn'].mean():.1%}
95% Wilson CI: [{churn_ci.low:.4f}, {churn_ci.high:.4f}]
Welch tenure t-test (churned vs retained): t={tenure_test.statistic:.4f}, p={tenure_test.pvalue:.6g}

REGRESSION
R-squared: {regression['r_squared']:.6f}
RMSE: {regression['rmse']:.6f}
Caution: Total_Sales is constructed as Quantity * Price. This regression is a mathematical reconstruction, not an independent forecasting model.

DATA LIMITATION
The supplied sales data has no Marketing_Spend column. Therefore marketing-spend-versus-revenue correlation or hypothesis testing cannot be conducted from these files. The churn dataset is analyzed separately; customer IDs in the two datasets are not verified as matching identities.
"""
    (OUTPUT_DIR / "hypothesis_tests_results.txt").write_text(report, encoding="utf-8")

    print("\\n--- KEY FINDINGS ---")
    print(f"Total sales: {sales_summary['total_sales']:,.2f}")
    print(f"Mean transaction sales: {sales_summary['mean_sales']:,.2f}")
    print(f"95% CI: [{ci[0]:,.2f}, {ci[1]:,.2f}]")
    print(f"Churn rate: {churn['Churn'].mean():.1%}")
    print("\\nHypothesis tests:\\n", tests.to_string(index=False))
    print("\\nCreated CSV outputs in outputs/ and charts in visualizations/")

if __name__ == "__main__":
    main()
