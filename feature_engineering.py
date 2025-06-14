import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def engineer_features(data, window_size=14, include_technical=True, include_seasonal=True):
    """
    Engineer features for gold price prediction
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Preprocessed gold price data
    window_size : int
        Size of the rolling window for calculations
    include_technical : bool
        Whether to include technical indicators
    include_seasonal : bool
        Whether to include seasonal features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with engineered features
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Check if talib is available, use alternative calculations if not
    try:
        import talib as ta
        has_talib = True
    except ImportError:
        has_talib = False
    
    # Technical indicators
    if include_technical:
        if has_talib:
            # Moving averages
            df['SMA_5'] = ta.SMA(df['Close'].values, timeperiod=5)
            df['SMA_20'] = ta.SMA(df['Close'].values, timeperiod=20)
            df['SMA_50'] = ta.SMA(df['Close'].values, timeperiod=50)
            df['SMA_200'] = ta.SMA(df['Close'].values, timeperiod=200)
            
            # Moving average crossovers
            df['MA_Cross_5_20'] = df['SMA_5'] - df['SMA_20']
            df['MA_Cross_20_50'] = df['SMA_20'] - df['SMA_50']
            
            # Exponential moving averages
            df['EMA_12'] = ta.EMA(df['Close'].values, timeperiod=12)
            df['EMA_26'] = ta.EMA(df['Close'].values, timeperiod=26)
            
            # MACD
            macd, macd_signal, macd_hist = ta.MACD(df['Close'].values, 
                                                  fastperiod=12, 
                                                  slowperiod=26, 
                                                  signalperiod=9)
            df['MACD'] = macd
            df['MACD_Signal'] = macd_signal
            df['MACD_Hist'] = macd_hist
            
            # RSI
            df['RSI'] = ta.RSI(df['Close'].values, timeperiod=window_size)
            
            # Bollinger Bands
            upper, middle, lower = ta.BBANDS(df['Close'].values, 
                                           timeperiod=window_size, 
                                           nbdevup=2, 
                                           nbdevdn=2, 
                                           matype=0)
            df['BB_Upper'] = upper
            df['BB_Middle'] = middle
            df['BB_Lower'] = lower
            df['BB_Width'] = (upper - lower) / middle
            
            # Stochastic oscillator
            slowk, slowd = ta.STOCH(df['High'].values, 
                                  df['Low'].values, 
                                  df['Close'].values, 
                                  fastk_period=14, 
                                  slowk_period=3, 
                                  slowk_matype=0, 
                                  slowd_period=3, 
                                  slowd_matype=0)
            df['Stoch_K'] = slowk
            df['Stoch_D'] = slowd
            
            # Commodity Channel Index
            df['CCI'] = ta.CCI(df['High'].values, 
                             df['Low'].values, 
                             df['Close'].values, 
                             timeperiod=window_size)
            
            # Average Directional Index
            df['ADX'] = ta.ADX(df['High'].values, 
                             df['Low'].values, 
                             df['Close'].values, 
                             timeperiod=window_size)
            
        else:
            # Manual calculations for common indicators if talib is not available
            # Moving averages
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Moving average crossovers
            df['MA_Cross_5_20'] = df['SMA_5'] - df['SMA_20']
            df['MA_Cross_20_50'] = df['SMA_20'] - df['SMA_50']
            
            # Exponential moving averages
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            
            # Simple MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Simple RSI implementation
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=window_size).mean()
            avg_loss = loss.rolling(window=window_size).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=window_size).mean()
            df['BB_Std'] = df['Close'].rolling(window=window_size).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Add rolling statistics
    df['Close_Min_5d'] = df['Close'].rolling(window=5).min()
    df['Close_Max_5d'] = df['Close'].rolling(window=5).max()
    df['Close_Std_5d'] = df['Close'].rolling(window=5).std()
    
    df['Close_Min_20d'] = df['Close'].rolling(window=20).min()
    df['Close_Max_20d'] = df['Close'].rolling(window=20).max()
    df['Close_Std_20d'] = df['Close'].rolling(window=20).std()
    
    # Price distance from moving averages (in %)
    df['Close_SMA5_Dist'] = (df['Close'] / df['SMA_5'] - 1) * 100
    df['Close_SMA20_Dist'] = (df['Close'] / df['SMA_20'] - 1) * 100
    df['Close_SMA50_Dist'] = (df['Close'] / df['SMA_50'] - 1) * 100
    
    # Seasonal features
    if include_seasonal:
        # Extract date features
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['DayOfWeek'] = df.index.dayofweek
        df['Quarter'] = df.index.quarter
        
        # One-hot encode month and day of week for seasonality
        for month in range(1, 13):
            df[f'Month_{month}'] = (df['Month'] == month).astype(int)
            
        for dow in range(5):  # Trading days (0-4 for Monday-Friday)
            df[f'DOW_{dow}'] = (df['DayOfWeek'] == dow).astype(int)
            
        # Create quarter features
        for quarter in range(1, 5):
            df[f'Quarter_{quarter}'] = (df['Quarter'] == quarter).astype(int)
    
    # Lagged features
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)
    df['Close_Lag5'] = df['Close'].shift(5)
    df['Close_Lag10'] = df['Close'].shift(10)
    
    # Return lags
    df['Return_Lag1'] = df['Returns'].shift(1)
    df['Return_Lag2'] = df['Returns'].shift(2)
    df['Return_Lag3'] = df['Returns'].shift(3)
    
    # Volatility lag
    df['Volatility_Lag1'] = df['Volatility'].shift(1)
    
    # Volume-based features (if available)
    if 'Volume' in df.columns:
        # Volume moving averages
        df['Volume_SMA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
        
        # Volume to moving average ratio
        df['Volume_Ratio_SMA5'] = df['Volume'] / df['Volume_SMA5']
        df['Volume_Ratio_SMA20'] = df['Volume'] / df['Volume_SMA20']
        
        # Price-volume relationship
        df['Price_Volume_Corr_5d'] = df['Close'].rolling(window=5).corr(df['Volume'])
        df['Price_Volume_Corr_20d'] = df['Close'].rolling(window=20).corr(df['Volume'])
    
    # Drop unnecessary columns for model training
    df = df.drop(['Year', 'Month', 'Day', 'DayOfWeek', 'Quarter'], axis=1, errors='ignore')
    
    # Drop NaN values
    df = df.dropna()
    
    return df
