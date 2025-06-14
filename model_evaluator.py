import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test data
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target values
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics for each model
    """
    evaluation_results = {
        'Model': [],
        'RMSE': [],
        'MAE': [],
        'R²': [],
        'MAPE (%)': []
    }
    
    for model_name, model in models.items():
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Store results
        evaluation_results['Model'].append(model_name)
        evaluation_results['RMSE'].append(round(rmse, 4))
        evaluation_results['MAE'].append(round(mae, 4))
        evaluation_results['R²'].append(round(r2, 4))
        evaluation_results['MAPE (%)'].append(round(mape, 2))
    
    return evaluation_results

def get_model_accuracy_summary(models, X_test, y_test):
    """
    Generate a summary of model accuracy
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target values
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with model accuracy metrics
    """
    # Get evaluation results
    results = evaluate_models(models, X_test, y_test)
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Sort by RMSE (lower is better)
    df = df.sort_values('RMSE')
    
    return df
