# Business Efficiency with Data Science: Churn and Demand Forecasting

This was a data science project where I looked at how a travel company (flights and hotel bookings) could use its data to make better decisions. I picked two use cases that matter a lot for this kind of business:

| Use case | Business question | Main method |
|---|---|---|
| **1. Customer churn** | Which customers are likely to leave, and why? | Exploratory analysis + XGBoost classifier |
| **2. Demand forecasting** | How much hotel and flight demand should we plan for? | ARIMA time-series forecasting + scenario planning |

Both use cases use **synthetic data** that I generated in Python to look like real-world data. Using synthetic data let me build the full pipeline, but it also limits what the results can tell us (more on this in [Honest notes](#honest-notes)).

The full write-up is in `Report summary.pdf`.

---

## Part 1: Customer churn

Keeping a customer is a lot cheaper than finding a new one. So the goal here was to understand what drives customers to leave and to build a model that flags customers who are at risk.

### The data

10,000 customers, one row each:

| Feature | Type | What it means |
|---|---|---|
| `CustomerID` | ID | Unique customer number (not used in analysis) |
| `Age` | Numeric | Customer's age |
| `ServiceFrequency` | Numeric | How often the customer used the service |
| `SubscriptionLength` | Numeric | How long they've been subscribed (months) |
| `SatisfactionScore` | Numeric, 0–10 | Customer satisfaction rating |
| `AverageSpending` | Numeric | Average amount spent |
| `LastActivityDays` | Numeric | Days since the customer was last active |
| `LoyaltyProgram` | Binary | 1 = in the loyalty program, 0 = not |
| `Complaints` | Numeric | Number of complaints filed |
| `Region` | Category | Urban, Suburban or Rural |
| `LastActivityDate` | Date | Date of last activity |
| `LastActivityMonth` | Date (derived) | Month of last activity |
| `Churn` | Binary (target) | 1 = customer left, 0 = stayed |

Some basic statistics for the key numeric features:

| Feature | Mean | Median | Std. dev. |
|---|---|---|---|
| Age | 43.5 | 43.0 | 14.9 |
| ServiceFrequency | 10.2 | 10.0 | 5.5 |
| AverageSpending | 2,748.7 | 2,741.5 | 1,299.6 |

### What I analysed

`churn_analysis.py` goes through the analysis in this order:

1. **Descriptive statistics:** mean, median, standard deviation and histograms for Age, ServiceFrequency and AverageSpending
2. **Correlation analysis:** a correlation heatmap of all the numeric features
3. **Demographics:** churn rate by age group (18–30, 31–40 … 71–80) and by region
4. **Behaviour:** box plots comparing churned and non-churned customers on service frequency, spending and days since last activity
5. **Churn over time:** monthly churn rate based on last activity month
6. **Satisfaction:** average satisfaction for churned vs non-churned customers, and churn rate for Low (0–3), Medium (4–6) and High (7–10) satisfaction groups
7. **Complaints:** churn rate for customers who did and didn't file a complaint
8. **Loyalty program:** churn rate, average spending and satisfaction for members vs non-members
9. **Subscription length:** how spending and satisfaction change as customers stay longer
10. **Rural, low-satisfaction customers:** filtering rural customers with a satisfaction score below 4 as a target group for retention
11. **XGBoost model:** predicting churn (details below)

### The churn model

| Setting | Value |
|---|---|
| Model | XGBoost (`binary:logistic`) |
| Features | Age, ServiceFrequency, SatisfactionScore, LastActivityDays, LoyaltyProgram, Complaints |
| Split | 70% train / 30% test, stratified by churn, `random_state=42` |
| Hyperparameters | learning rate 0.1, max depth 6, subsample 0.8, colsample_bytree 0.8, 100 boosting rounds |
| Threshold | 0.5 |
| Evaluation | accuracy, precision, recall, F1, classification report, feature importance |

### What I found

- **The correlations were all very weak.** For example, LastActivityDays vs Churn was −0.011, SatisfactionScore vs Churn was −0.007, and Complaints vs Churn was about 0.02. Being in the loyalty program was slightly negatively correlated with churn.
- **Churn rate by age group** was quite flat, from 18.5% (ages 41–50) to 21.9% (ages 51–60).
- **Complaints:** about 6,500 customers had filed a complaint and just over 3,000 hadn't. The churn rate was a little higher for customers who complained.
- **Loyalty program:** members churned slightly less than non-members.
- **Satisfaction:** churned customers had slightly lower satisfaction scores on average.
- **Subscription length:** spending and satisfaction went down slightly for customers with longer subscriptions, which could mean engagement fades over time.
- **XGBoost feature importance:** LastActivityDays came out as the most important feature, followed by Age, SatisfactionScore, ServiceFrequency and Complaints.

From these I suggested some business actions: expanding the loyalty program, fixing complaints faster, running targeted campaigns for inactive and low-spending customers, and re-checking churn drivers every quarter or six months.

---

## Part 2: Hotel and flight demand forecasting

If a travel company knows when demand is going to rise or drop, it can plan staff, rooms, flights and prices ahead of time instead of reacting late.

### The data

A synthetic monthly time series with three columns:

| Column | Meaning |
|---|---|
| `Time` | Month-end date (YYYY-MM-DD) |
| `Hotel_Demand` | Hotel demand for that month |
| `Flight_Demand` | Flight demand for that month |

I generated the series with an upward trend over the years and a seasonal pattern: higher demand in summer (June–August) and lower demand from September to February.

### What I did

1. **Visualised and decomposed** both series to see the trend and seasonality
2. **Forecast both series with ARIMA.** I chose ARIMA because it handles trend through differencing and short-term patterns through its AR and MA parts, and it doesn't need a huge dataset. The forecast plots show the actual data, the forecast and a confidence interval.
3. **Scenario planning:** I simulated a sudden **50% increase** and a **25% decrease** in demand to see how the business should react to surprises
4. **Resource allocation:** from the high-demand scenario I picked out the peak months (forecast values between about 81 and 90, in the synthetic year 2044). Assuming one resort holds 20 people, I worked out how many extra hotels would be needed each month, which came out at 4–5.

---

## What's in this repo

```
business-efficiency-DS/
├── README.md
├── churn_analysis.py      Part 1: churn analysis + XGBoost (exported from Google Colab)
└── Report summary.pdf     full write-up of both parts, with all the figures
```

Some things are **not here yet**:

- the code for Part 2 (ARIMA forecasting, scenarios, resource allocation)
- the two synthetic datasets, and the script I used to generate them

## Running the churn analysis

The script was exported from a Colab notebook, so it still mounts Google Drive.

1. Get `customer_churn_data.csv` (see above, not uploaded yet).
2. Remove the `drive.mount(...)` lines and change the file path:
   ```python
   df = pd.read_csv("data/customer_churn_data.csv")
   ```
3. Install the libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
4. Run it:
   ```bash
   python churn_analysis.py
   ```
   Every chart opens in its own window, one after another. It might be easier to run it as a notebook.


## What I'd do next

- Re-run the XGBoost model, add its scores and a majority-class baseline, and use class weights or threshold tuning for the churn class
- Switch to gain-based or SHAP feature importance
- Add significance tests for the group comparisons
- Upload the forecasting code and both datasets, plus the data generator
- Hold out the last 12 months to test the forecast, compare ARIMA with SARIMA, and report MAE and MAPE
- Try the same pipeline on a public real-world dataset (for example a telecom churn dataset or real hotel booking data)
