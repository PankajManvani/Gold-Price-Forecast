import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import json
import os

from data_processor import fetch_gold_data, preprocess_data
from feature_engineering import engineer_features
from model_trainer import train_models
from model_evaluator import evaluate_models
from forecaster import forecast_future_prices
from visualizer import plot_predictions, plot_feature_importance, plot_gold_price_history
from database import save_forecast, save_model_performance, get_forecasts, get_forecast, get_model_performances

# Page configuration
st.set_page_config(
    page_title="Gold Price Prediction App",
    page_icon="💰",
    layout="wide"
)

# Main title
st.title("Gold Price Prediction Application")
st.write("This application uses machine learning models to predict gold prices based on historical data.")

# Sidebar for inputs
st.sidebar.header("Settings")

# Date range selection
start_date = st.sidebar.date_input(
    "Start Date",
    datetime.now() - timedelta(days=3*365)  # 3 years ago by default
)
end_date = st.sidebar.date_input(
    "End Date",
    datetime.now()
)

# Convert to string format needed for yfinance
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Selecting gold ticker
ticker_options = {
    "Gold (USD)": "GC=F",
    "Gold ETF (GLD)": "GLD"
}
selected_ticker = st.sidebar.selectbox(
    "Select Gold Ticker",
    list(ticker_options.keys())
)
ticker = ticker_options[selected_ticker]

# Model selection (allow multiple)
available_models = ["Linear Regression", "Random Forest", "Support Vector Regression", "XGBoost"]
selected_models = st.sidebar.multiselect(
    "Select Models for Prediction",
    available_models,
    default=["Linear Regression", "Random Forest"]
)

# Train-test split ratio
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

# Feature engineering options
feature_options = st.sidebar.expander("Feature Engineering Options")
with feature_options:
    window_size = st.slider("Rolling Window Size (days)", 5, 30, 14)
    include_technical = st.checkbox("Include Technical Indicators", True)
    include_seasonal = st.checkbox("Include Seasonal Features", True)

# Forecast settings
forecast_options = st.sidebar.expander("Forecast Settings")
with forecast_options:
    forecast_days = st.slider("Forecast Days", 1, 90, 30)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Analysis", "Model Performance", "Predictions", "Forecast", "Saved Data"])

# Attempt to load data
with st.spinner("Fetching gold price data..."):
    try:
        # Fetch data
        gold_data = fetch_gold_data(ticker, start_date_str, end_date_str)
        
        if gold_data is None or gold_data.empty:
            st.error("Failed to fetch gold price data. Please check your internet connection or try a different date range.")
            st.stop()
        
        # Preprocessing
        preprocessed_data = preprocess_data(gold_data)
        
        # Feature engineering
        feature_data = engineer_features(
            preprocessed_data, 
            window_size=window_size, 
            include_technical=include_technical,
            include_seasonal=include_seasonal
        )
            
        # Show data analysis tab content
        with tab1:
            st.subheader("Historical Gold Price Data")
            st.dataframe(preprocessed_data.head())
            
            st.subheader("Gold Price Chart")
            fig = plot_gold_price_history(preprocessed_data)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Statistical Summary")
            st.write(preprocessed_data.describe())
            
            st.subheader("Engineered Features")
            st.dataframe(feature_data.head())
            
            # Correlation heatmap
            if st.checkbox("Show Correlation Heatmap"):
                st.subheader("Correlation Between Features")
                corr = feature_data.corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    colorscale='Viridis',
                    zmin=-1, zmax=1
                ))
                fig.update_layout(height=600, width=800)
                st.plotly_chart(fig, use_container_width=True)
        
        # Don't proceed to modeling if no models selected
        if not selected_models:
            st.warning("Please select at least one model for prediction.")
            st.stop()
            
        # Train models with selected options
        with st.spinner("Training selected models..."):
            X_train, X_test, y_train, y_test, models = train_models(
                feature_data, 
                selected_models, 
                test_size=test_size
            )
                
        # Model evaluation
        with tab2:
            st.subheader("Model Performance Metrics")
            evaluation_results = evaluate_models(models, X_test, y_test)
            
            # Display metrics
            metrics_df = pd.DataFrame(evaluation_results)
            st.dataframe(metrics_df)
            
            # Plot predictions vs actual for each model
            st.subheader("Model Predictions vs Actual")
            for model_name in models:
                fig = plot_predictions(models[model_name], X_test, y_test, model_name)
                st.plotly_chart(fig, use_container_width=True)
                
            # Feature importance for models that support it
            st.subheader("Feature Importance")
            for model_name in models:
                if model_name in ["Random Forest", "XGBoost"]:
                    fig = plot_feature_importance(models[model_name], X_train.columns, model_name)
                    st.plotly_chart(fig, use_container_width=True)
            
        # Detailed predictions
        with tab3:
            st.subheader("Detailed Prediction Analysis")
            
            # Create predictions DataFrame
            predictions_df = pd.DataFrame({'Actual': y_test})
            for model_name in models:
                predictions_df[f'{model_name} Prediction'] = models[model_name].predict(X_test)
                predictions_df[f'{model_name} Error'] = abs(predictions_df['Actual'] - predictions_df[f'{model_name} Prediction'])
                
            st.dataframe(predictions_df)
            
            # Combined plot of all models
            st.subheader("All Models vs Actual")
            fig = go.Figure()
            
            # Add actual prices
            fig.add_trace(go.Scatter(
                x=predictions_df.index,
                y=predictions_df['Actual'],
                mode='lines',
                name='Actual Price',
                line=dict(color='black', width=2)
            ))
            
            # Add predictions for each model
            colors = ['blue', 'red', 'green', 'purple']
            for i, model_name in enumerate(models):
                fig.add_trace(go.Scatter(
                    x=predictions_df.index,
                    y=predictions_df[f'{model_name} Prediction'],
                    mode='lines',
                    name=f'{model_name} Prediction',
                    line=dict(color=colors[i % len(colors)])
                ))
                
            fig.update_layout(
                title='Model Predictions vs Actual Gold Prices',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                legend_title='Legend',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
                
        # Future forecast
        with tab4:
            st.subheader(f"Gold Price Forecast for Next {forecast_days} Days")
            
            # Select model for forecasting
            forecast_model = st.selectbox(
                "Select Model for Forecasting",
                list(models.keys())
            )
            
            # Get forecast
            with st.spinner("Generating forecast..."):
                forecast_result = forecast_future_prices(
                    feature_data,
                    models[forecast_model], 
                    forecast_days,
                    window_size=window_size,
                    include_technical=include_technical,
                    include_seasonal=include_seasonal
                )
                
                # Display forecast
                st.dataframe(forecast_result)
                
                # Plot forecast
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=preprocessed_data.index[-60:],  # Last 60 days
                    y=preprocessed_data['Close'][-60:],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue')
                ))
                
                # Forecast data
                fig.add_trace(go.Scatter(
                    x=forecast_result.index,
                    y=forecast_result['Forecast'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red')
                ))
                
                # Add confidence intervals if available
                if 'Lower Bound' in forecast_result.columns and 'Upper Bound' in forecast_result.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast_result.index.tolist() + forecast_result.index.tolist()[::-1],
                        y=forecast_result['Upper Bound'].tolist() + forecast_result['Lower Bound'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence Interval'
                    ))
                
                fig.update_layout(
                    title=f'Gold Price Forecast - Next {forecast_days} Days',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    legend_title='Legend',
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("⚠️ Disclaimer: These predictions are based on historical patterns and may not account for unexpected market events.")
                
                # Save forecast option
                save_forecast_option = st.checkbox("Save this forecast to database")
                if save_forecast_option:
                    forecast_name = st.text_input("Enter a name for this forecast", f"{selected_ticker} {forecast_model} Forecast {datetime.now().strftime('%Y-%m-%d')}")
                    if st.button("Save Forecast"):
                        # Save the forecast to database
                        forecast_id = save_forecast(
                            forecast_name=forecast_name, 
                            model_name=forecast_model,
                            ticker=ticker,
                            start_date=start_date,
                            end_date=end_date,
                            forecast_days=forecast_days,
                            parameters={
                                'window_size': window_size,
                                'include_technical': include_technical,
                                'include_seasonal': include_seasonal,
                                'test_size': test_size
                            },
                            forecast_df=forecast_result
                        )
                        
                        if forecast_id:
                            st.success(f"Forecast saved successfully with ID: {forecast_id}")
                        else:
                            st.error("Failed to save forecast")
        
        # Show saved data tab regardless of other operations
        with tab5:
            st.subheader("Database Management")
            
            # Create tabs for different database functions
            db_tab1, db_tab2 = st.tabs(["Saved Forecasts", "Model Performance"])
            
            with db_tab1:
                st.subheader("Previously Saved Forecasts")
                
                # Get all forecasts
                forecasts = get_forecasts()
                
                if not forecasts:
                    st.info("No forecasts saved yet. Generate and save a forecast to see it here.")
                else:
                    # Create a summary table
                    forecast_summary = []
                    for f in forecasts:
                        forecast_summary.append({
                            'ID': f['id'],
                            'Name': f['forecast_name'],
                            'Model': f['model_name'],
                            'Ticker': f['ticker'],
                            'Start Date': f['start_date'].split('T')[0],
                            'End Date': f['end_date'].split('T')[0],
                            'Forecast Days': f['forecast_days'],
                            'Date Created': f['date_created'].split('T')[0]
                        })
                    
                    # Display summary table
                    summary_df = pd.DataFrame(forecast_summary)
                    st.dataframe(summary_df)
                    
                    # Option to view a specific forecast
                    selected_forecast_id = st.selectbox(
                        "Select a forecast to view details",
                        options=[f['ID'] for f in forecast_summary],
                        format_func=lambda x: f"ID: {x} - {next((f['Name'] for f in forecast_summary if f['ID'] == x), '')}"
                    )
                    
                    if st.button("View Selected Forecast"):
                        # Get the forecast details
                        forecast = get_forecast(selected_forecast_id)
                        
                        if forecast:
                            st.subheader(f"Forecast: {forecast['forecast_name']}")
                            
                            # Display forecast details
                            st.write(f"**Model:** {forecast['model_name']}")
                            st.write(f"**Ticker:** {forecast['ticker']}")
                            st.write(f"**Date Range:** {forecast['start_date'].split('T')[0]} to {forecast['end_date'].split('T')[0]}")
                            st.write(f"**Forecast Days:** {forecast['forecast_days']}")
                            st.write(f"**Created On:** {forecast['date_created'].split('T')[0]}")
                            
                            # Parameters
                            if forecast['parameters']:
                                st.subheader("Parameters Used")
                                for key, value in forecast['parameters'].items():
                                    st.write(f"**{key}:** {value}")
                            
                            # Convert forecast data to DataFrame
                            forecast_data = forecast['forecast_data']
                            df = pd.DataFrame(forecast_data)
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.set_index('date')
                            
                            # Display forecast data
                            st.subheader("Forecast Results")
                            st.dataframe(df)
                            
                            # Plot forecast
                            st.subheader("Forecast Chart")
                            fig = go.Figure()
                            
                            # Forecast data
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['Forecast'],
                                mode='lines',
                                name='Forecast',
                                line=dict(color='red')
                            ))
                            
                            # Add confidence intervals if available
                            if 'Lower Bound' in df.columns and 'Upper Bound' in df.columns:
                                fig.add_trace(go.Scatter(
                                    x=df.index.tolist() + df.index.tolist()[::-1],
                                    y=df['Upper Bound'].tolist() + df['Lower Bound'].tolist()[::-1],
                                    fill='toself',
                                    fillcolor='rgba(255,0,0,0.2)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='95% Confidence Interval'
                                ))
                            
                            fig.update_layout(
                                title=f'{forecast["forecast_name"]}',
                                xaxis_title='Date',
                                yaxis_title='Price (USD)',
                                legend_title='Legend',
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Failed to retrieve forecast")
            
            with db_tab2:
                st.subheader("Model Performance History")
                
                # Get all model performances
                performances = get_model_performances()
                
                if not performances:
                    st.info("No model performance data saved yet.")
                    
                    # Option to save current model performances
                    if 'evaluation_results' in locals() and st.button("Save Current Model Performance"):
                        # Save performance metrics for each model
                        saved_ids = []
                        for i, model_name in enumerate(evaluation_results['Model']):
                            perf_id = save_model_performance(
                                model_name=model_name,
                                ticker=ticker,
                                start_date=start_date,
                                end_date=end_date,
                                test_size=test_size,
                                rmse=evaluation_results['RMSE'][i],
                                mae=evaluation_results['MAE'][i],
                                r2=evaluation_results['R²'][i],
                                mape=evaluation_results['MAPE (%)'][i]
                            )
                            if perf_id:
                                saved_ids.append(perf_id)
                        
                        if saved_ids:
                            st.success(f"Saved performance metrics for {len(saved_ids)} models")
                        else:
                            st.error("Failed to save performance metrics")
                else:
                    # Create a summary table
                    perf_summary = []
                    for p in performances:
                        perf_summary.append({
                            'ID': p['id'],
                            'Model': p['model_name'],
                            'Ticker': p['ticker'],
                            'Date Range': f"{p['start_date'].split('T')[0]} to {p['end_date'].split('T')[0]}",
                            'Test Size': f"{int(p['test_size'] * 100)}%",
                            'RMSE': round(p['rmse'], 4),
                            'MAE': round(p['mae'], 4),
                            'R²': round(p['r2'], 4),
                            'MAPE (%)': round(p['mape'], 2),
                            'Date Created': p['date_created'].split('T')[0]
                        })
                    
                    # Display summary table
                    summary_df = pd.DataFrame(perf_summary)
                    st.dataframe(summary_df)
                    
                    # Compare performance across models
                    st.subheader("Performance Comparison")
                    
                    # Group by model and calculate average metrics
                    model_groups = {}
                    for p in perf_summary:
                        model = p['Model']
                        if model not in model_groups:
                            model_groups[model] = {
                                'RMSE': [], 'MAE': [], 'R²': [], 'MAPE (%)': []
                            }
                        model_groups[model]['RMSE'].append(p['RMSE'])
                        model_groups[model]['MAE'].append(p['MAE'])
                        model_groups[model]['R²'].append(p['R²'])
                        model_groups[model]['MAPE (%)'].append(p['MAPE (%)'])
                    
                    # Calculate averages
                    avg_metrics = []
                    for model, metrics in model_groups.items():
                        avg_metrics.append({
                            'Model': model,
                            'Avg RMSE': round(sum(metrics['RMSE']) / len(metrics['RMSE']), 4),
                            'Avg MAE': round(sum(metrics['MAE']) / len(metrics['MAE']), 4),
                            'Avg R²': round(sum(metrics['R²']) / len(metrics['R²']), 4),
                            'Avg MAPE (%)': round(sum(metrics['MAPE (%)']) / len(metrics['MAPE (%)']), 2)
                        })
                    
                    # Sort by RMSE (lowest first)
                    avg_metrics.sort(key=lambda x: x['Avg RMSE'])
                    
                    # Display average metrics
                    avg_df = pd.DataFrame(avg_metrics)
                    st.dataframe(avg_df)
                    
                    # Bar chart comparison
                    st.subheader("Model Comparison Chart")
                    fig = go.Figure()
                    
                    models = [m['Model'] for m in avg_metrics]
                    
                    # Add bars for each metric
                    fig.add_trace(go.Bar(
                        x=models,
                        y=[m['Avg RMSE'] for m in avg_metrics],
                        name='Avg RMSE',
                        marker_color='indianred'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=models,
                        y=[m['Avg MAPE (%)'] / 10 for m in avg_metrics],  # Scale down for better visibility
                        name='Avg MAPE (%) / 10',
                        marker_color='lightsalmon'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=models,
                        y=[m['Avg R²'] for m in avg_metrics],
                        name='Avg R²',
                        marker_color='royalblue'
                    ))
                    
                    fig.update_layout(
                        title='Average Model Performance Metrics',
                        xaxis_title='Model',
                        yaxis_title='Value',
                        barmode='group',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Option to save current model performances
                    if 'evaluation_results' in locals() and st.button("Save Current Model Performance"):
                        # Save performance metrics for each model
                        saved_ids = []
                        for i, model_name in enumerate(evaluation_results['Model']):
                            perf_id = save_model_performance(
                                model_name=model_name,
                                ticker=ticker,
                                start_date=start_date,
                                end_date=end_date,
                                test_size=test_size,
                                rmse=evaluation_results['RMSE'][i],
                                mae=evaluation_results['MAE'][i],
                                r2=evaluation_results['R²'][i],
                                mape=evaluation_results['MAPE (%)'][i]
                            )
                            if perf_id:
                                saved_ids.append(perf_id)
                        
                        if saved_ids:
                            st.success(f"Saved performance metrics for {len(saved_ids)} models")
                        else:
                            st.error("Failed to save performance metrics")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try again with different parameters or check your internet connection.")

# Footer
st.markdown("---")
st.write("Gold Price Prediction App | Machine Learning Model")
