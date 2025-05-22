import pytest
import asyncio
from datetime import datetime
from src.risk_manager import RiskManager, RiskMetrics
from src.portfolio_manager import PortfolioManager

@pytest.fixture
async def portfolio_manager():
    return PortfolioManager(initial_balance=10000.0)

@pytest.fixture
async def risk_manager(portfolio_manager):
    return RiskManager(portfolio_manager)

@pytest.mark.asyncio
async def test_calculate_position_risk(risk_manager):
    """Test calculating position risk metrics."""
    # Setup initial portfolio
    await risk_manager.portfolio_manager.update_position("AAPL", 10.0, 150.0, True)
    
    # Calculate risk for new position
    risk_metrics = await risk_manager.calculate_position_risk("GOOGL", 5.0, 200.0)
    assert risk_metrics is not None
    assert 'position_size_pct' in risk_metrics
    assert 'var_95' in risk_metrics
    assert 'var_99' in risk_metrics
    assert 'volatility' in risk_metrics
    assert 'max_loss' in risk_metrics

@pytest.mark.asyncio
async def test_validate_trade(risk_manager):
    """Test trade validation against risk limits."""
    # Test valid trade
    valid = await risk_manager.validate_trade("AAPL", 5.0, 150.0, True)
    assert valid
    
    # Test trade exceeding position size limit
    risk_manager.risk_limits['max_position_size'] = 0.01  # 1% limit
    valid = await risk_manager.validate_trade("AAPL", 100.0, 150.0, True)
    assert not valid

@pytest.mark.asyncio
async def test_calculate_portfolio_risk(risk_manager):
    """Test calculating portfolio risk metrics."""
    # Setup portfolio with positions
    await risk_manager.portfolio_manager.update_position("AAPL", 10.0, 150.0, True)
    await risk_manager.portfolio_manager.update_position("GOOGL", 5.0, 200.0, True)
    
    # Calculate portfolio risk
    risk_metrics = await risk_manager.calculate_portfolio_risk()
    assert risk_metrics is not None
    assert isinstance(risk_metrics, RiskMetrics)
    assert risk_metrics.var_95 > 0
    assert risk_metrics.var_99 > risk_metrics.var_95
    assert risk_metrics.max_drawdown <= 0
    assert risk_metrics.volatility > 0

@pytest.mark.asyncio
async def test_update_risk_limits(risk_manager):
    """Test updating risk limits."""
    # Get initial limits
    initial_limits = risk_manager.get_risk_limits()
    
    # Update limits
    new_limits = {
        'max_position_size': 0.05,
        'max_drawdown': 0.15
    }
    success = await risk_manager.update_risk_limits(new_limits)
    assert success
    
    # Verify updates
    updated_limits = risk_manager.get_risk_limits()
    assert updated_limits['max_position_size'] == 0.05
    assert updated_limits['max_drawdown'] == 0.15
    assert updated_limits['max_leverage'] == initial_limits['max_leverage']  # Unchanged

@pytest.mark.asyncio
async def test_risk_limits_persistence(risk_manager):
    """Test that risk limits persist between operations."""
    # Set custom limits
    await risk_manager.update_risk_limits({
        'max_position_size': 0.05,
        'max_drawdown': 0.15
    })
    
    # Validate trade with new limits
    valid = await risk_manager.validate_trade("AAPL", 100.0, 150.0, True)
    assert not valid  # Should fail due to position size limit

@pytest.mark.asyncio
async def test_correlation_check(risk_manager):
    """Test correlation check with existing positions."""
    # Setup initial position
    await risk_manager.portfolio_manager.update_position("AAPL", 10.0, 150.0, True)
    
    # Check correlation for new position
    # Note: This is currently a placeholder that always returns True
    valid = await risk_manager._check_correlation("GOOGL")
    assert valid

@pytest.mark.asyncio
async def test_error_handling(risk_manager):
    """Test error handling in risk calculations."""
    # Test with invalid portfolio state
    risk_manager.portfolio_manager.cash_balance = -1000.0
    risk_metrics = await risk_manager.calculate_position_risk("AAPL", 5.0, 150.0)
    assert risk_metrics is None
    
    # Test with invalid risk limits
    await risk_manager.update_risk_limits({
        'max_position_size': -0.1  # Invalid negative value
    })
    valid = await risk_manager.validate_trade("AAPL", 5.0, 150.0, True)
    assert not valid 