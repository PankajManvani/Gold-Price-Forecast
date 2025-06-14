import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

# Get database URL from environment variable
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create base class for models
Base = declarative_base()

# Define models
class GoldPrice(Base):
    """Gold price historical data"""
    __tablename__ = 'gold_prices'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date.isoformat(),
            'ticker': self.ticker,
            'open': self.open_price,
            'high': self.high_price,
            'low': self.low_price,
            'close': self.close_price,
            'volume': self.volume
        }

class Forecast(Base):
    """Saved forecast results"""
    __tablename__ = 'forecasts'
    
    id = Column(Integer, primary_key=True)
    date_created = Column(DateTime, default=datetime.now)
    forecast_name = Column(String(100), nullable=False)
    model_name = Column(String(50), nullable=False)
    ticker = Column(String(10), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    forecast_days = Column(Integer, nullable=False)
    parameters = Column(Text, nullable=True)  # JSON string of parameters
    forecast_data = Column(Text, nullable=False)  # JSON string of forecast results
    
    def to_dict(self):
        return {
            'id': self.id,
            'date_created': self.date_created.isoformat(),
            'forecast_name': self.forecast_name,
            'model_name': self.model_name,
            'ticker': self.ticker,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'forecast_days': self.forecast_days,
            'parameters': json.loads(self.parameters) if self.parameters else {},
            'forecast_data': json.loads(self.forecast_data)
        }

class ModelPerformance(Base):
    """Model performance metrics"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    date_created = Column(DateTime, default=datetime.now)
    model_name = Column(String(50), nullable=False)
    ticker = Column(String(10), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    test_size = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)  # Root Mean Squared Error
    mae = Column(Float, nullable=False)   # Mean Absolute Error
    r2 = Column(Float, nullable=False)    # R-squared
    mape = Column(Float, nullable=False)  # Mean Absolute Percentage Error
    
    def to_dict(self):
        return {
            'id': self.id,
            'date_created': self.date_created.isoformat(),
            'model_name': self.model_name,
            'ticker': self.ticker,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'test_size': self.test_size,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'mape': self.mape
        }

# Create tables
Base.metadata.create_all(engine)

# Create session factory
Session = sessionmaker(bind=engine)

# Database functions
def save_gold_data(df, ticker):
    """
    Save gold price data to database
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing gold price data
    ticker : str
        Ticker symbol
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        session = Session()
        
        # Check if data already exists
        existing_dates = session.query(GoldPrice.date).filter(GoldPrice.ticker == ticker).all()
        existing_dates = [d[0].date() for d in existing_dates]
        
        # Prepare records to insert
        records = []
        for index, row in df.iterrows():
            # Skip if already exists
            if index.date() in existing_dates:
                continue
                
            gold_price = GoldPrice(
                date=index,
                ticker=ticker,
                open_price=row['Open'],
                high_price=row['High'],
                low_price=row['Low'],
                close_price=row['Close'],
                volume=row['Volume'] if 'Volume' in row else None
            )
            records.append(gold_price)
        
        # Insert new records
        if records:
            session.add_all(records)
            session.commit()
        
        session.close()
        return True
        
    except Exception as e:
        print(f"Error saving gold data: {str(e)}")
        if session:
            session.rollback()
            session.close()
        return False

def get_gold_data(ticker, start_date, end_date):
    """
    Get gold price data from database
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    start_date : datetime
        Start date
    end_date : datetime
        End date
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing gold price data, or None if not found
    """
    try:
        # Use raw SQL query to fetch data directly rather than through ORM to avoid Ticker objects
        conn = engine.connect()
        query = f"""
            SELECT date, open_price, high_price, low_price, close_price, volume 
            FROM gold_prices 
            WHERE ticker = '{ticker}' 
            AND date >= '{start_date}' 
            AND date <= '{end_date}'
            ORDER BY date
        """
        df = pd.read_sql(query, conn, parse_dates=['date'])
        conn.close()
        
        if df.empty:
            return None
        
        # Set index and rename columns
        df = df.set_index('date')
        df = df.rename(columns={
            'open_price': 'Open',
            'high_price': 'High', 
            'low_price': 'Low', 
            'close_price': 'Close',
            'volume': 'Volume'
        })
        
        # Ensure data types match yfinance output
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        
        return df
        
    except Exception as e:
        print(f"Error getting gold data: {str(e)}")
        # No session to close with direct SQL approach
        return None

def save_forecast(forecast_name, model_name, ticker, start_date, end_date, 
                  forecast_days, parameters, forecast_df):
    """
    Save forecast results to database
    
    Parameters:
    -----------
    forecast_name : str
        Name for this forecast
    model_name : str
        Name of the model used
    ticker : str
        Ticker symbol
    start_date : datetime
        Start date of historical data
    end_date : datetime
        End date of historical data
    forecast_days : int
        Number of days forecasted
    parameters : dict
        Dictionary of parameters used
    forecast_df : pandas.DataFrame
        DataFrame containing forecast results
    
    Returns:
    --------
    int
        ID of saved forecast, or None if error
    """
    try:
        session = Session()
        
        # Convert DataFrame to JSON
        forecast_json = forecast_df.reset_index()
        forecast_json['date'] = forecast_json['index'].astype(str)
        forecast_json = forecast_json.drop('index', axis=1).to_dict(orient='records')
        
        # Create forecast record
        forecast = Forecast(
            forecast_name=forecast_name,
            model_name=model_name,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            forecast_days=forecast_days,
            parameters=json.dumps(parameters) if parameters else None,
            forecast_data=json.dumps(forecast_json)
        )
        
        session.add(forecast)
        session.commit()
        
        forecast_id = forecast.id
        session.close()
        
        return forecast_id
        
    except Exception as e:
        print(f"Error saving forecast: {str(e)}")
        if session:
            session.rollback()
            session.close()
        return None

def get_forecasts():
    """
    Get all saved forecasts
    
    Returns:
    --------
    list
        List of forecast dictionaries
    """
    try:
        session = Session()
        forecasts = session.query(Forecast).order_by(Forecast.date_created.desc()).all()
        result = [f.to_dict() for f in forecasts]
        session.close()
        return result
    except Exception as e:
        print(f"Error getting forecasts: {str(e)}")
        if session:
            session.close()
        return []

def get_forecast(forecast_id):
    """
    Get a specific forecast by ID
    
    Parameters:
    -----------
    forecast_id : int
        ID of the forecast
    
    Returns:
    --------
    dict
        Forecast dictionary, or None if not found
    """
    try:
        session = Session()
        forecast = session.query(Forecast).filter(Forecast.id == forecast_id).first()
        session.close()
        
        if forecast:
            return forecast.to_dict()
        return None
        
    except Exception as e:
        print(f"Error getting forecast: {str(e)}")
        if session:
            session.close()
        return None

def save_model_performance(model_name, ticker, start_date, end_date, test_size, 
                           rmse, mae, r2, mape):
    """
    Save model performance metrics
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    ticker : str
        Ticker symbol
    start_date : datetime
        Start date of data
    end_date : datetime
        End date of data
    test_size : float
        Proportion of data used for testing
    rmse : float
        Root Mean Squared Error
    mae : float
        Mean Absolute Error
    r2 : float
        R-squared
    mape : float
        Mean Absolute Percentage Error
    
    Returns:
    --------
    int
        ID of saved performance record, or None if error
    """
    try:
        session = Session()
        
        # Create performance record
        performance = ModelPerformance(
            model_name=model_name,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            test_size=test_size,
            rmse=rmse,
            mae=mae,
            r2=r2,
            mape=mape
        )
        
        session.add(performance)
        session.commit()
        
        perf_id = performance.id
        session.close()
        
        return perf_id
        
    except Exception as e:
        print(f"Error saving model performance: {str(e)}")
        if session:
            session.rollback()
            session.close()
        return None

def get_model_performances():
    """
    Get all model performance records
    
    Returns:
    --------
    list
        List of performance dictionaries
    """
    try:
        session = Session()
        performances = session.query(ModelPerformance).order_by(
            ModelPerformance.date_created.desc()
        ).all()
        result = [p.to_dict() for p in performances]
        session.close()
        return result
    except Exception as e:
        print(f"Error getting model performances: {str(e)}")
        if session:
            session.close()
        return []