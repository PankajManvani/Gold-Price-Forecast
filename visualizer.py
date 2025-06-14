import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def plot_gold_price_history(data):
    """
    Create a candlestick chart of gold price history
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Historical gold price data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object for the chart
    """
    # Create figure
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'], 
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Gold Price'
    )])
    
    # Add moving averages if available
    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            mode='lines',
            name='20-day MA',
            line=dict(color='blue', width=1)
        ))
    
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_50'],
            mode='lines',
            name='50-day MA',
            line=dict(color='orange', width=1)
        ))
    
    if 'SMA_200' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_200'],
            mode='lines',
            name='200-day MA',
            line=dict(color='green', width=1)
        ))
    
    # Update layout
    fig.update_layout(
        title='Gold Price History',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def plot_predictions(model, X_test, y_test, model_name):
    """
    Plot model predictions against actual values
    
    Parameters:
    -----------
    model : object
        Trained machine learning model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Actual target values
    model_name : str
        Name of the model
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object for the chart
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Create figure
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test,
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add predicted prices
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_pred,
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', width=2)
    ))
    
    # Calculate error metrics
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    
    # Update layout
    fig.update_layout(
        title=f'{model_name} Predictions (RMSE: {rmse:.2f}, R²: {r2:.4f})',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        height=500
    )
    
    return fig

def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance for tree-based models
    
    Parameters:
    -----------
    model : object
        Trained machine learning model
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object for the chart
    """
    # Extract feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        # For models wrapped in classes like SVRWithScaling
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            importance = model.model.feature_importances_
        else:
            # Return empty figure if model doesn't support feature importance
            fig = go.Figure()
            fig.update_layout(
                title=f"Feature importance not available for {model_name}",
                height=400
            )
            return fig
    
    # Sort features by importance
    indices = np.argsort(importance)[-20:]  # Top 20 features
    
    # Create figure
    fig = go.Figure(go.Bar(
        x=importance[indices],
        y=[feature_names[i] for i in indices],
        orientation='h'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Top Features Importance - {model_name}',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=600,
        yaxis=dict(autorange="reversed")
    )
    
    return fig
