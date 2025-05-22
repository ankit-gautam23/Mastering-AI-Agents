import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from ..src.performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from ..src.portfolio_manager import PortfolioManager

@pytest.fixture
def portfolio_manager():
    return PortfolioManager()

@pytest.fixture
def performance_analyzer(portfolio_manager):
    return PerformanceAnalyzer(portfolio_manager)

@pytest.fixture
def sample_trades():
    return [
        {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.0,
            'realized_pnl': 1000.0,
            'timestamp': datetime.now() - timedelta(days=30)
        },
        {
            'symbol': 'AAPL',
            'quantity': -100,
            'price': 160.0,
            'realized_pnl': 1000.0,
            'timestamp': datetime.now() - timedelta(days=20)
        },
        {
            'symbol': 'GOOGL',
            'quantity': 50,
            'price': 2000.0,
            'realized_pnl': -500.0,
            'timestamp': datetime.now() - timedelta(days=15)
        },
        {
            'symbol': 'GOOGL',
            'quantity': -50,
            'price': 1900.0,
            'realized_pnl': -500.0,
            'timestamp': datetime.now() - timedelta(days=5)
        }
    ]

@pytest.mark.asyncio
async def test_calculate_performance_metrics(performance_analyzer, portfolio_manager, sample_trades):
    # Add sample trades to portfolio manager
    for trade in sample_trades:
        await portfolio_manager.update_position(
            trade['symbol'],
            trade['quantity'],
            trade['price']
        )
    
    # Calculate metrics
    metrics = await performance_analyzer.calculate_performance_metrics()
    
    assert metrics is not None
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_return == 1000.0  # Sum of all realized P&L
    assert metrics.total_trades == 4
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 2
    assert metrics.win_rate == 0.5

@pytest.mark.asyncio
async def test_generate_performance_report(performance_analyzer, portfolio_manager, sample_trades):
    # Add sample trades
    for trade in sample_trades:
        await portfolio_manager.update_position(
            trade['symbol'],
            trade['quantity'],
            trade['price']
        )
    
    # Generate report
    report = await performance_analyzer.generate_performance_report()
    
    assert report is not None
    assert 'summary' in report
    assert 'trade_statistics' in report
    assert 'portfolio_statistics' in report
    assert 'timestamp' in report
    
    # Check summary metrics
    assert 'total_return' in report['summary']
    assert 'annualized_return' in report['summary']
    assert 'sharpe_ratio' in report['summary']
    assert 'max_drawdown' in report['summary']
    
    # Check trade statistics
    assert report['trade_statistics']['total_trades'] == 4
    assert report['trade_statistics']['winning_trades'] == 2
    assert report['trade_statistics']['losing_trades'] == 2

@pytest.mark.asyncio
async def test_plot_performance_charts(performance_analyzer, portfolio_manager, sample_trades):
    # Add sample trades
    for trade in sample_trades:
        await portfolio_manager.update_position(
            trade['symbol'],
            trade['quantity'],
            trade['price']
        )
    
    # Generate charts
    figures = await performance_analyzer.plot_performance_charts()
    
    assert isinstance(figures, dict)
    assert 'equity_curve' in figures
    assert 'drawdown' in figures
    assert 'monthly_returns' in figures
    
    # Check figure types
    for fig in figures.values():
        assert hasattr(fig, 'axes')
        assert len(fig.axes) > 0

@pytest.mark.asyncio
async def test_performance_metrics_with_date_filter(performance_analyzer, portfolio_manager, sample_trades):
    # Add sample trades
    for trade in sample_trades:
        await portfolio_manager.update_position(
            trade['symbol'],
            trade['quantity'],
            trade['price']
        )
    
    # Calculate metrics with date filter
    start_date = datetime.now() - timedelta(days=25)
    metrics = await performance_analyzer.calculate_performance_metrics(start_date)
    
    assert metrics is not None
    assert metrics.total_trades == 2  # Only trades after start_date

@pytest.mark.asyncio
async def test_empty_trade_history(performance_analyzer):
    # Test with no trades
    metrics = await performance_analyzer.calculate_performance_metrics()
    assert metrics is None
    
    report = await performance_analyzer.generate_performance_report()
    assert report is None
    
    figures = await performance_analyzer.plot_performance_charts()
    assert figures == {}

@pytest.mark.asyncio
async def test_error_handling(performance_analyzer, portfolio_manager):
    # Test with invalid portfolio state
    portfolio_manager.positions = None
    
    metrics = await performance_analyzer.calculate_performance_metrics()
    assert metrics is None
    
    report = await performance_analyzer.generate_performance_report()
    assert report is None
    
    figures = await performance_analyzer.plot_performance_charts()
    assert figures == {} 