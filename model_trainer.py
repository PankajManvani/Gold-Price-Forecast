import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import streamlit as st

def train_models(data, selected_models, test_size=0.2, target_col='Close', random_state=42):
    """
    Train selected machine learning models on the provided data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Feature-engineered data
    selected_models : list
        List of model names to train
    test_size : float
        Proportion of the data to use for testing
    target_col : str
        Name of the target column to predict
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, trained_models_dict
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Define features and target
    features = df.drop([target_col], axis=1, errors='ignore')
    target = df[target_col]
    
    # Get the feature names for later use
    feature_names = features.columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, shuffle=False
    )
    
    # Scale the features (important for some models like SVR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to keep column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    
    # Initialize dictionary to store trained models
    trained_models = {}
    
    # Train the selected models
    for model_name in selected_models:
        with st.spinner(f"Training {model_name}..."):
            if model_name == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)  # Linear regression doesn't need scaling
                trained_models[model_name] = model
                
            elif model_name == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=random_state
                )
                model.fit(X_train, y_train)  # Random Forest doesn't need scaling
                trained_models[model_name] = model
                
            elif model_name == "Support Vector Regression":
                model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
                model.fit(X_train_scaled_df, y_train)  # SVR needs scaling
                # Create a wrapper for SVR to handle scaling internally
                class SVRWithScaling:
                    def __init__(self, model, scaler):
                        self.model = model
                        self.scaler = scaler
                        
                    def predict(self, X):
                        X_scaled = self.scaler.transform(X)
                        return self.model.predict(X_scaled)
                
                trained_models[model_name] = SVRWithScaling(model, scaler)
                
            elif model_name == "XGBoost":
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state
                )
                model.fit(X_train, y_train)  # XGBoost doesn't need scaling
                trained_models[model_name] = model
    
    return X_train, X_test, y_train, y_test, trained_models
