
#
Global Superstore Sales Forecasting & Analytics
#
**Project Overview**

This project focuses on analyzing historical sales data from the Global Superstore dataset and building a machine learning model to forecast future sales and predict profit margins. By combining Power BI for visualization and XGBoost for predictive modeling, the project provides actionable insights into business performance.

**Tech Stack**

Data Visualization: Power BI Desktop

Programming Language: Python

Libraries: Pandas, Numpy, XGBoost, Scikit-learn, Matplotlib, Seaborn

Version Control: Git & GitHub

**Machine Learning Approach**
- Sales Forecasting
  
Goal: Forecast monthly sales using historical data
Models: Prophet and XGBoost
Feature Engineering:
Time-based features (month, quarter)
Lag features (1, 3, 12 months)
Rolling statistics
Data Transformation: Log transformation to handle skewness
- Profit Analysis
  
Goal: Analyze the impact of discount on revenue and profitability
Approach: Simulated profit under different discount levels

**Model Evaluation:**

XGBoost outperformed Prophet in forecasting accuracy
WAPE reduced from 34.89% (Prophet) to 31.45% (XGBoost)
R² Score (XGBoost): 0.6070

**Key Takeaways**

XGBoost captures complex patterns better than Prophet for this dataset
Model performance varies across regions, indicating heterogeneous market behavior
Discount strategies significantly impact profitability and should be optimized carefully
