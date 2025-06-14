💰 Gold Price Prediction with Machine Learning

✨ Project Title and Elevator Pitch
GoldPriceForecast is an interactive web application that leverages machine learning models to forecast future gold prices based on historical data and advanced feature engineering. It's your personal data-driven compass for navigating the gold market!

🌟 Detailed Description
This project provides a comprehensive platform for analyzing historical gold price trends, evaluating various machine learning models for prediction, and generating future price forecasts. It empowers users to make more informed decisions by offering insights into model performance and key influencing factors.

🎯 What it does:
Fetches Historical Data: Gathers real-time and historical gold price data using Yahoo Finance.
Preprocesses and Engineers Features: Cleans the data and creates a rich set of features, including technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands) and seasonal components.
Trains Multiple ML Models: Supports training and evaluating popular regression models like Linear Regression, Random Forest, Support Vector Regression (SVR), and XGBoost.
Evaluates Model Performance: Provides detailed metrics (RMSE, MAE, R², MAPE) and visual comparisons of actual vs. predicted prices for each model.
Generates Future Forecasts: Projects future gold prices for a user-defined number of days, complete with confidence intervals.
Database Integration: Allows users to save forecasts and model performance metrics for later review and comparison, enhancing historical tracking and analysis.
Interactive Visualizations: Presents data, predictions, and forecasts through engaging and insightful Plotly charts within a Streamlit interface.

💡 Why it's useful:
In the volatile financial markets, predicting asset prices is a challenging yet highly valuable endeavor. This application aims to provide:
Enhanced Decision Making: Helps investors and analysts gain data-driven insights into potential gold price movements.
Model Comparison: Allows users to compare the effectiveness of different machine learning algorithms on gold price data.
Understanding Influencing Factors: Through feature importance plots, users can identify which factors most significantly impact price predictions.
Historical Performance Tracking: The integrated database feature enables a persistent record of past forecasts and model accuracies.

🚀 Key Features:
Interactive Streamlit UI: User-friendly interface for seamless data exploration and model interaction.
Multiple Model Support: Choose from Linear Regression, Random Forest, SVR, and XGBoost.
Dynamic Date Range & Ticker Selection: Customize historical data fetching for various gold assets (e.g., GC=F, GLD).
Customizable Feature Engineering: Control the inclusion of technical indicators and seasonal features.
Comprehensive Model Evaluation: View RMSE, MAE, R², and MAPE for each trained model.
Detailed Prediction Plots: Visualize how well each model predicts actual prices.
Feature Importance Analysis: Understand which features are most influential for tree-based models.
Future Price Forecasting: Project gold prices with configurable forecast horizons and confidence intervals.
Persistent Data Storage: Save forecasts and model performance records to a PostgreSQL database.

⚙️ Tech Stack
This project is built primarily with Python, utilizing powerful libraries for data analysis, machine learning, and interactive web application development.
Category
Technology
Logo (Conceptual)
Language
Python
Web Framework
Streamlit
Data Processing
Pandas, NumPy


Machine Learning
Scikit-learn, XGBoost, TA-Lib (optional)


Visualization
Plotly
Database
PostgreSQL (via SQLAlchemy)
Data Source
yfinance
(No official logo available, but represents Yahoo Finance API)

📦 Installation and Usage
To get this project up and running on your local machine, follow these steps:
Prerequisites
Python 3.8+
pip (Python package installer)

Step-by-Step Installation:
1.Download Code

2.Create a virtual environment (recommended):
python -m venv venv


3.Activate the virtual environment:
On macOS/Linux:
source venv/bin/activate


On Windows:
.\venv\Scripts\activate


4.Install the required dependencies:
The project uses pyproject.toml for dependency management. Install them using pip:
pip install -e .

This command reads the pyproject.toml and installs all necessary packages, including numpy, pandas, plotly, scikit-learn, xgboost, streamlit, sqlalchemy, yfinance, and psycopg2-binary.

5.Set up the Database:
This application requires a PostgreSQL database.
Install PostgreSQL: If you don't have PostgreSQL installed, download it from https://www.postgresql.org/download/.
Create a Database: Create a new database, e.g., gold_price_db.
Set Environment Variable: The application connects to the database using an environment variable named DATABASE_URL. Set this variable in your terminal before running the app.

# Example for macOS/Linux (replace with your actual database credentials)
export DATABASE_URL="postgresql://user:password@localhost:5432/gold_price_db"

# Example for Windows (Command Prompt)
set DATABASE_URL="postgresql://user:password@localhost:5432/gold_price_db"

# Example for Windows (PowerShell)
$env:DATABASE_URL="postgresql://user:password@localhost:5432/gold_price_db"

Ensure the user has appropriate permissions for the database.
Usage
6.Run the Streamlit application:
With your virtual environment activated and DATABASE_URL set, navigate to the project's root directory and run:
streamlit run app.py


Interact with the Application:
The application will open in your web browser (usually http://localhost:5000).
Use the sidebar to adjust the date range for historical data, select gold tickers (USD or ETF), choose machine learning models, and configure feature engineering options.

Explore different tabs for Data Analysis, Model Performance, Detailed Predictions, Future Forecasts, and Saved Data.
Save your generated forecasts and model performance metrics to the database for future reference.

📁 Project Structure
The repository is organized into a modular structure to ensure maintainability and readability.
GoldPriceForecast/
├── app.py                      # Main Streamlit application file
├── data_processor.py           # Handles data fetching and preprocessing
├── feature_engineering.py      # Contains functions for creating ML features (technical/seasonal)
├── model_trainer.py            # Manages training of selected ML models
├── model_evaluator.py          # Calculates and displays model performance metrics
├── forecaster.py               # Implements future price forecasting logic
├── visualizer.py               # Functions for generating Plotly visualizations
├── database.py                 # Defines SQLAlchemy models and functions for database interaction
├── pyproject.toml              # Project dependencies and metadata
└── README.md                   # This file


app.py: The central file that orchestrates the Streamlit UI and integrates all other modules.
data_processor.py: Responsible for downloading gold price data from Yahoo Finance and performing initial cleaning and basic calculations (returns, volatility).
feature_engineering.py: Focuses on creating advanced features from the raw data, including various technical indicators (Moving Averages, RSI, MACD, Bollinger Bands) and time-based seasonal features (month, day of week, quarter). It includes a fallback for talib if not installed.
model_trainer.py: Handles splitting data into training and testing sets, scaling features (where necessary), and training the selected machine learning models.
model_evaluator.py: Calculates and presents standard regression metrics (RMSE, MAE, R², MAPE) to assess how well the models perform.
forecaster.py: Contains the logic for generating future gold price predictions based on the best-performing model, including an iterative forecasting approach and simplified confidence intervals.
visualizer.py: Provides functions to generate interactive charts using Plotly, such as historical candlestick charts, actual vs. predicted plots, and feature importance bar charts.
database.py: Manages the database schema using SQLAlchemy, defining tables for GoldPrice data, Forecasts, and ModelPerformance. It includes functions to save and retrieve data from a PostgreSQL database.
pyproject.toml: Specifies the project's metadata and its Python dependencies, ensuring a consistent environment setup.
