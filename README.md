<img width="1396" height="790" alt="image" src="https://github.com/user-attachments/assets/8ab18a85-aaba-4fdd-b4d9-2f8d44732218" />Global Superstore Sales Forecasting & Analytics 
- Project Overview
This project focuses on analyzing historical sales data from the Global Superstore dataset and building a machine learning model to forecast future sales and predict profit margins. By combining Power BI for visualization and XGBoost for predictive modeling, the project provides actionable insights into business performance.

- Tech Stack
Data Visualization: Power BI Desktop

Programming Language: Python

Libraries: Pandas, Numpy, XGBoost, Scikit-learn, Matplotlib, Seaborn

Version Control: Git & GitHub

Machine Learning Approach
We developed two main predictive models using the XGBoost Regressor:

Sales Forecasting:

Goal: Predict monthly sales for 2016 based on 2012-2015 historical data.

Feature Engineering: Time-based features (month, quarter), Lag features (1, 3, 12 months), and Rolling averages.

Data Transformation: Log transformation was applied to handle data skewness.

Profit Margin Prediction:

Goal: Estimate the profitability of each order.

Key Features: Shipping Cost Ratio, Ship Mode, Category, and Discount.

Model Evaluation:
WAPE (Weighted Absolute Percentage Error):31.45%

R2 Score: 0.6070
