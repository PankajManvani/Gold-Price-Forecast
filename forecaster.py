import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def forecast_future_prices(historical_data, model, forecast_days=30, window_size=14, include_technical=True, include_seasonal=True):
    """
    Forecast future gold prices using the trained model
    
    Parameters:
    -----------
    historical_data : pandas.DataFrame
        Historical data with engineered features
    model : object
        Trained machine learning model
    forecast_days : int
        Number of days to forecast
    window_size : int
        Size of rolling window for calculations
    include_technical : bool
        Whether to include technical indicators
    include_seasonal : bool
        Whether to include seasonal features
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with forecasted prices and dates
    """
    # Create a copy of the latest data for forecasting
    forecast_data = historical_data.copy()
    latest_date = forecast_data.index[-1]
    
    # Prepare data to store forecasts
    forecast_dates = [latest_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    # Convert to a business day calendar (skip weekends)
    business_dates = pd.bdate_range(start=forecast_dates[0], periods=forecast_days)
    
    # Create a DataFrame to store forecasts
    forecasts = pd.DataFrame(index=business_dates)
    forecasts['Forecast'] = np.nan
    
    # Get the columns used for prediction
    feature_columns = historical_data.columns.drop('Close', errors='ignore').tolist()
    
    # Get the last row of data (will be used to start forecasting)
    last_known_data = forecast_data.iloc[-1:].copy()
    
    # Iteratively predict future values
    current_data = last_known_data.copy()
    
    for i, date in enumerate(business_dates):
        # Prepare features for current prediction
        if i > 0:
            # Update date-related features
            if include_seasonal:
                # Set seasonal features
                current_data['Month'] = date.month
                current_data['Day'] = date.day
                current_data['DayOfWeek'] = date.dayofweek
                current_data['Quarter'] = date.quarter
                
                # One-hot encode month
                for month in range(1, 13):
                    current_data[f'Month_{month}'] = 1 if date.month == month else 0
                
                # One-hot encode day of week
                for dow in range(5):
                    current_data[f'DOW_{dow}'] = 1 if date.dayofweek == dow else 0
                
                # One-hot encode quarter
                for quarter in range(1, 5):
                    current_data[f'Quarter_{quarter}'] = 1 if date.quarter == quarter else 0
            
            # Update lagged features with previously predicted values
            current_data['Close_Lag1'] = forecasts['Forecast'].iloc[i-1] if i >= 1 else last_known_data['Close'].values[0]
            current_data['Close_Lag2'] = forecasts['Forecast'].iloc[i-2] if i >= 2 else last_known_data['Close_Lag1'].values[0]
            current_data['Close_Lag3'] = forecasts['Forecast'].iloc[i-3] if i >= 3 else last_known_data['Close_Lag2'].values[0]
            current_data['Close_Lag5'] = forecasts['Forecast'].iloc[i-5] if i >= 5 else last_known_data['Close_Lag4'].values[0] if 'Close_Lag4' in last_known_data.columns else last_known_data['Close_Lag3'].values[0]
            current_data['Close_Lag10'] = forecasts['Forecast'].iloc[i-10] if i >= 10 else last_known_data['Close_Lag9'].values[0] if 'Close_Lag9' in last_known_data.columns else last_known_data['Close_Lag5'].values[0]
        
        # Make prediction
        prediction_features = current_data[feature_columns]
        
        try:
            # Ensure features match the expected format
            feature_columns_set = set(feature_columns)
            missing_columns = feature_columns_set - set(prediction_features.columns)
            extra_columns = set(prediction_features.columns) - feature_columns_set
            
            # Add missing columns with default values
            for col in missing_columns:
                prediction_features[col] = 0
                
            # Remove extra columns
            if extra_columns:
                prediction_features = prediction_features.drop(columns=list(extra_columns))
                
            # Ensure columns are in the same order as during training
            prediction_features = prediction_features[feature_columns]
            
            # Convert to numpy array if needed
            if hasattr(model, 'predict') and 'sklearn' in str(type(model)):
                X = prediction_features.values.reshape(1, -1)
                prediction = model.predict(X)
            else:
                # For other model types (like SVRWithScaling wrapper)
                prediction = model.predict(prediction_features)
            
            # Get the prediction value, handle different return shapes
            pred_value = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
            
            # Store the prediction
            forecasts.loc[date, 'Forecast'] = pred_value
            
            # Update current data with the new prediction for next iteration
            current_data = current_data.copy()
            current_data['Close'] = pred_value
            
            # Calculate returns based on new prediction
            if i > 0:
                prev_close = forecasts['Forecast'].iloc[i-1]
                current_data['Returns'] = ((pred_value / prev_close) - 1) * 100
                current_data['LogReturns'] = np.log(pred_value / prev_close) * 100
            
            # Update technical indicators if needed (simplified)
            if include_technical and i > 0:
                # Simple moving averages - just approximate them
                recent_values = list(forecasts['Forecast'].iloc[max(0, i-4):i+1])
                if i < 4:  # Add historical values to complete 5-day window
                    historical_vals = list(historical_data['Close'].iloc[-(5-len(recent_values)):].values)
                    values_5d = historical_vals + recent_values
                else:
                    values_5d = recent_values
                
                current_data['SMA_5'] = np.mean(values_5d)
                
                # Approximate other indicators based on current forecast context
                if 'RSI' in current_data.columns:
                    # Simplified RSI approximation
                    current_data['RSI'] = 50 + current_data['Returns'] / 2
                
                if 'BB_Upper' in current_data.columns:
                    # Simplified Bollinger Bands approximation
                    current_data['BB_Middle'] = current_data['SMA_5']
                    current_data['BB_Upper'] = current_data['SMA_5'] * 1.05  # Just an approximation
                    current_data['BB_Lower'] = current_data['SMA_5'] * 0.95  # Just an approximation
            
        except Exception as e:
            # If prediction fails, use the last known/predicted value
            last_value = last_known_data['Close'].values[0] if i == 0 else forecasts['Forecast'].iloc[i-1]
            forecasts.loc[date, 'Forecast'] = last_value
            current_data['Close'] = last_value
            print(f"Prediction error on day {i+1}: {str(e)}")
    
    # Add confidence intervals (simplified)
    # In a real application, better uncertainty methods would be used
    forecast_std = historical_data['Close'].pct_change().std() * 100
    forecast_range = np.sqrt(np.arange(1, forecast_days + 1)) * forecast_std * 0.01 * forecasts['Forecast']
    
    forecasts['Lower Bound'] = forecasts['Forecast'] - forecast_range
    forecasts['Upper Bound'] = forecasts['Forecast'] + forecast_range
    
    return forecasts
