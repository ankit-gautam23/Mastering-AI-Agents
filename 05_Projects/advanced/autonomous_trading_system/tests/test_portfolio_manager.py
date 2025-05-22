import pytest
from datetime import datetime
from src.portfolio_manager import PortfolioManager, Position, PortfolioMetrics

@pytest.fixture
def portfolio_manager():
    return PortfolioManager(initial_balance=10000.0)

@pytest.fixture
def sample_position():
    return Position(
        symbol="AAPL",
        quantity=10.0,
        average_price=150.0,
        current_price=155.0,
        unrealized_pnl=50.0,
        realized_pnl=0.0,
        last_updated=datetime.now()
    )

@pytest.mark.asyncio
async def test_update_position_buy(portfolio_manager):
    """Test updating position for buy trade."""
    # Test initial buy
    success = await portfolio_manager.update_position("AAPL", 10.0, 150.0, True)
    assert success
    position = await portfolio_manager.get_position("AAPL")
    assert position.quantity == 10.0
    assert position.average_price == 150.0
    assert portfolio_manager.cash_balance == 8500.0  # 10000 - (10 * 150)

    # Test additional buy
    success = await portfolio_manager.update_position("AAPL", 5.0, 160.0, True)
    assert success
    position = await portfolio_manager.get_position("AAPL")
    assert position.quantity == 15.0
    assert position.average_price == 153.33  # (10*150 + 5*160) / 15
    assert portfolio_manager.cash_balance == 7700.0  # 8500 - (5 * 160)

@pytest.mark.asyncio
async def test_update_position_sell(portfolio_manager):
    """Test updating position for sell trade."""
    # Setup initial position
    await portfolio_manager.update_position("AAPL", 10.0, 150.0, True)
    
    # Test sell
    success = await portfolio_manager.update_position("AAPL", 5.0, 160.0, False)
    assert success
    position = await portfolio_manager.get_position("AAPL")
    assert position.quantity == 5.0
    assert position.average_price == 150.0
    assert position.realized_pnl == 50.0  # (160 - 150) * 5
    assert portfolio_manager.cash_balance == 8500.0  # 7700 + (5 * 160)

    # Test sell more than available
    success = await portfolio_manager.update_position("AAPL", 10.0, 160.0, False)
    assert not success
    position = await portfolio_manager.get_position("AAPL")
    assert position.quantity == 5.0  # Unchanged

@pytest.mark.asyncio
async def test_update_prices(portfolio_manager):
    """Test updating position prices."""
    # Setup initial position
    await portfolio_manager.update_position("AAPL", 10.0, 150.0, True)
    
    # Update prices
    await portfolio_manager.update_prices({"AAPL": 160.0})
    position = await portfolio_manager.get_position("AAPL")
    assert position.current_price == 160.0
    assert position.unrealized_pnl == 100.0  # (160 - 150) * 10

@pytest.mark.asyncio
async def test_get_portfolio_metrics(portfolio_manager):
    """Test getting portfolio metrics."""
    # Setup initial positions
    await portfolio_manager.update_position("AAPL", 10.0, 150.0, True)
    await portfolio_manager.update_position("GOOGL", 5.0, 200.0, True)
    
    # Update prices
    await portfolio_manager.update_prices({
        "AAPL": 160.0,
        "GOOGL": 210.0
    })
    
    metrics = await portfolio_manager.get_portfolio_metrics()
    assert metrics is not None
    assert metrics.total_value == 10000.0  # Initial balance
    assert metrics.cash_balance == 5000.0  # 10000 - (10*150 + 5*200)
    assert metrics.unrealized_pnl == 200.0  # (160-150)*10 + (210-200)*5
    assert metrics.realized_pnl == 0.0
    assert metrics.total_pnl == 200.0

@pytest.mark.asyncio
async def test_get_trade_history(portfolio_manager):
    """Test getting trade history."""
    # Setup trades
    await portfolio_manager.update_position("AAPL", 10.0, 150.0, True)
    await portfolio_manager.update_position("AAPL", 5.0, 160.0, False)
    
    # Get all trades
    trades = await portfolio_manager.get_trade_history()
    assert len(trades) == 2
    assert trades[0]["symbol"] == "AAPL"
    assert trades[0]["is_buy"] == True
    assert trades[1]["is_buy"] == False
    
    # Get trades for specific symbol
    aapl_trades = await portfolio_manager.get_trade_history("AAPL")
    assert len(aapl_trades) == 2
    
    # Get trades for non-existent symbol
    other_trades = await portfolio_manager.get_trade_history("GOOGL")
    assert len(other_trades) == 0

@pytest.mark.asyncio
async def test_calculate_position_size(portfolio_manager):
    """Test calculating position size."""
    # Setup initial position
    await portfolio_manager.update_position("AAPL", 10.0, 150.0, True)
    
    # Calculate position size
    size = await portfolio_manager.calculate_position_size("AAPL", 160.0)
    assert size > 0
    assert size <= portfolio_manager.cash_balance / 160.0

def test_get_position_count(portfolio_manager):
    """Test getting position count."""
    assert portfolio_manager.get_position_count() == 0
    
    # Add positions
    asyncio.run(portfolio_manager.update_position("AAPL", 10.0, 150.0, True))
    asyncio.run(portfolio_manager.update_position("GOOGL", 5.0, 200.0, True))
    
    assert portfolio_manager.get_position_count() == 2

def test_get_trade_count(portfolio_manager):
    """Test getting trade count."""
    assert portfolio_manager.get_trade_count() == 0
    
    # Add trades
    asyncio.run(portfolio_manager.update_position("AAPL", 10.0, 150.0, True))
    asyncio.run(portfolio_manager.update_position("AAPL", 5.0, 160.0, False))
    
    assert portfolio_manager.get_trade_count() == 2 