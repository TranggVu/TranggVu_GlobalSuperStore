📊 Global Superstore Sales Forecasting & Analytics
📝 Project Overview
This project focuses on analyzing historical sales data from the Global Superstore dataset. The goal is to build a high-accuracy machine learning model to forecast future sales and predict profit margins.

By combining Power BI for deep-dive visualization and XGBoost for predictive modeling, the project provides actionable insights into business performance and optimization.

🛠 Tech Stack
Data Visualization: Power BI Desktop

Programming Language: Python 🐍

Libraries: Pandas, Numpy, XGBoost, Scikit-learn, Matplotlib, Seaborn

Version Control: Git & GitHub

📈 Business Insights (Power BI)
Dưới đây là Dashboard phân tích tổng quan tình hình kinh doanh, doanh thu và các phân khúc khách hàng:

Hình 1: Tổng quan phân tích doanh số và lợi nhuận trên Power BI

🤖 Machine Learning Approach
1. Sales Forecasting
Goal: Forecast monthly sales using historical data.

Models: Comparison between Prophet and XGBoost.

Feature Engineering: Time-based features (month, quarter), Lag features (1, 3, 12 months), and Rolling statistics.

Data Transformation: Log transformation applied to handle skewness.

Hình 2: Kết quả dự báo doanh thu hàng tháng

2. Profit Analysis
Goal: Analyze the impact of discount on revenue and profitability.

Approach: Simulated profit under different discount levels to find the optimal strategy.

Hình 3: Mô phỏng biến động lợi nhuận theo mức giảm giá (Discount)

🧪 Model Evaluation
Sau khi thử nghiệm, XGBoost cho thấy hiệu quả vượt trội trong việc nắm bắt các quy luật phức tạp của tập dữ liệu:

Accuracy: XGBoost outperformed Prophet in forecasting accuracy.

Error Reduction: WAPE reduced from 34.89% (Prophet) to 31.45% (XGBoost).

R² Score (XGBoost): 0.6070.

💡 Key Takeaways
Model Selection: XGBoost captures complex patterns better than Prophet for this specific retail dataset.

Regional Insights: Model performance varies across regions, indicating heterogeneous market behavior.

Strategic Optimization: Discount strategies significantly impact profitability and should be optimized carefully using data-driven simulations.
