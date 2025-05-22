import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from ..src.backtesting_engine import BacktestingEngine, BacktestResult
from ..src.portfolio_manager import PortfolioManager
from ..src.risk_manager import RiskManager

@pytest.fixture
def portfolio_manager():
    return PortfolioManager()

@pytest.fixture
def risk_manager(portfolio_manager):
    return RiskManager(portfolio_manager)

@pytest.fixture
def backtesting_engine(portfolio_manager, risk_manager):
    return BacktestingEngine(portfolio_manager, risk_manager)

@pytest.fixture
def sample_historical_data():
    # Create sample historical data
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, len(dates)),
        'high': np.random.normal(102, 1, len(dates)),
        'low': np.random.normal(98, 1, len(dates)),
        'close': np.random.normal(100, 1, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    return data

@pytest.fixture
def sample_strategy():
    async def strategy(symbol: str, data: pd.DataFrame, params: dict) -> list:
        signals = []
        if len(data) < 2:
            return signals
        
        # Simple moving average crossover strategy
        data['sma20'] = data['close'].rolling(window=20).mean()
        data['sma50'] = data['close'].rolling(window=50).mean()
        
        if len(data) >= 50:
            last_row = data.iloc[-1]
            prev_row = data.iloc[-2]
            
            # Buy signal: SMA20 crosses above SMA50
            if prev_row['sma20'] <= prev_row['sma50'] and last_row['sma20'] > last_row['sma50']:
                signals.append({
                    'action': 'buy',
                    'quantity': 100,
                    'price': last_row['close']
                })
            # Sell signal: SMA20 crosses below SMA50
            elif prev_row['sma20'] >= prev_row['sma50'] and last_row['sma20'] < last_row['sma50']:
                signals.append({
                    'action': 'sell',
                    'quantity': 100,
                    'price': last_row['close']
                })
        
        return signals
    return strategy

@pytest.mark.asyncio
async def test_load_historical_data(backtesting_engine, sample_historical_data):
    # Test loading valid data
    success = await backtesting_engine.load_historical_data('AAPL', sample_historical_data)
    assert success
    
    # Test loading data with missing columns
    invalid_data = sample_historical_data.drop('volume', axis=1)
    success = await backtesting_engine.load_historical_data('GOOGL', invalid_data)
    assert not success

@pytest.mark.asyncio
async def test_set_strategy(backtesting_engine, sample_strategy):
    # Test setting strategy
    success = await backtesting_engine.set_strategy(sample_strategy)
    assert success
    
    # Test setting strategy with parameters
    params = {'window': 20, 'quantity': 100}
    success = await backtesting_engine.set_strategy(sample_strategy, params)
    assert success

@pytest.mark.asyncio
async def test_run_backtest(backtesting_engine, sample_historical_data, sample_strategy):
    # Load data and set strategy
    await backtesting_engine.load_historical_data('AAPL', sample_historical_data)
    await backtesting_engine.set_strategy(sample_strategy)
    
    # Run backtest
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    initial_capital = 10000.0
    
    result = await backtesting_engine.run_backtest(start_date, end_date, initial_capital)
    
    assert result is not None
    assert isinstance(result, BacktestResult)
    assert result.strategy_name == sample_strategy.__name__
    assert result.start_date == start_date
    assert result.end_date == end_date
    assert result.initial_capital == initial_capital
    assert result.final_capital > 0
    assert result.total_trades >= 0

@pytest.mark.asyncio
async def test_get_historical_data(backtesting_engine, sample_historical_data):
    # Load data
    await backtesting_engine.load_historical_data('AAPL', sample_historical_data)
    
    # Get full data
    data = await backtesting_engine.get_historical_data('AAPL')
    assert data is not None
    assert len(data) == len(sample_historical_data)
    
    # Get data for date range
    start_date = datetime(2023, 1, 10)
    end_date = datetime(2023, 1, 20)
    data = await backtesting_engine.get_historical_data('AAPL', start_date, end_date)
    assert data is not None
    assert len(data) == 11  # 11 days inclusive
    
    # Test non-existent symbol
    data = await backtesting_engine.get_historical_data('INVALID')
    assert data is None

@pytest.mark.asyncio
async def test_get_available_symbols(backtesting_engine, sample_historical_data):
    # Load data for multiple symbols
    await backtesting_engine.load_historical_data('AAPL', sample_historical_data)
    await backtesting_engine.load_historical_data('GOOGL', sample_historical_data)
    
    symbols = await backtesting_engine.get_available_symbols()
    assert len(symbols) == 2
    assert 'AAPL' in symbols
    assert 'GOOGL' in symbols

@pytest.mark.asyncio
async def test_get_current_date(backtesting_engine, sample_historical_data, sample_strategy):
    # Load data and set strategy
    await backtesting_engine.load_historical_data('AAPL', sample_historical_data)
    await backtesting_engine.set_strategy(sample_strategy)
    
    # Check date before backtest
    assert await backtesting_engine.get_current_date() is None
    
    # Run backtest
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    await backtesting_engine.run_backtest(start_date, end_date, 10000.0)
    
    # Check date after backtest
    assert await backtesting_engine.get_current_date() == end_date

@pytest.mark.asyncio
async def test_error_handling(backtesting_engine):
    # Test running backtest without strategy
    result = await backtesting_engine.run_backtest(
        datetime(2023, 1, 1),
        datetime(2023, 1, 31),
        10000.0
    )
    assert result is None
    
    # Test running backtest without data
    async def dummy_strategy(symbol, data, params):
        return []
    
    await backtesting_engine.set_strategy(dummy_strategy)
    result = await backtesting_engine.run_backtest(
        datetime(2023, 1, 1),
        datetime(2023, 1, 31),
        10000.0
    )
    assert result is None 