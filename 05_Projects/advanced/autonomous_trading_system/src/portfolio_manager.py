from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class Position:
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    last_updated: datetime
    metadata: Dict[str, Any] = None

@dataclass
class PortfolioMetrics:
    total_value: float
    cash_balance: float
    margin_used: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    last_updated: datetime
    metadata: Dict[str, Any] = None

class PortfolioManager:
    def __init__(self, initial_balance: float = 10000.0):
        self.logger = logging.getLogger(__name__)
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def update_position(self, symbol: str, quantity: float, price: float, is_buy: bool) -> bool:
        """
        Update position after trade.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Trade price
            is_buy: Whether this is a buy trade
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                if symbol not in self.positions:
                    if not is_buy:
                        self.logger.warning(f"Cannot sell {symbol}: no position")
                        return False
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=0.0,
                        average_price=0.0,
                        current_price=price,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        last_updated=datetime.now()
                    )
                
                position = self.positions[symbol]
                trade_value = quantity * price
                
                if is_buy:
                    # Update position for buy
                    new_quantity = position.quantity + quantity
                    new_cost = (position.quantity * position.average_price) + trade_value
                    position.quantity = new_quantity
                    position.average_price = new_cost / new_quantity
                    self.cash_balance -= trade_value
                else:
                    # Update position for sell
                    if quantity > position.quantity:
                        self.logger.warning(f"Cannot sell {quantity} {symbol}: only {position.quantity} available")
                        return False
                    
                    # Calculate realized P&L
                    realized_pnl = (price - position.average_price) * quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= quantity
                    self.cash_balance += trade_value
                
                position.current_price = price
                position.last_updated = datetime.now()
                
                # Record trade
                self.trade_history.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'is_buy': is_buy,
                    'timestamp': datetime.now(),
                    'realized_pnl': realized_pnl if not is_buy else 0.0
                })
                
                return True
        except Exception as e:
            self.logger.error(f"Failed to update position: {e}")
            return False

    async def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update position prices.
        
        Args:
            prices: Dictionary of symbol to price
        """
        try:
            async with self._lock:
                for symbol, price in prices.items():
                    if symbol in self.positions:
                        position = self.positions[symbol]
                        position.current_price = price
                        position.unrealized_pnl = (price - position.average_price) * position.quantity
                        position.last_updated = datetime.now()
        except Exception as e:
            self.logger.error(f"Failed to update prices: {e}")

    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position by symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position if found, None otherwise
        """
        return self.positions.get(symbol)

    async def list_positions(self) -> List[Position]:
        """
        List all positions.
        
        Returns:
            List of positions
        """
        return list(self.positions.values())

    async def get_portfolio_metrics(self) -> PortfolioMetrics:
        """
        Get portfolio metrics.
        
        Returns:
            Portfolio metrics
        """
        try:
            async with self._lock:
                total_value = self.cash_balance
                unrealized_pnl = 0.0
                
                for position in self.positions.values():
                    position_value = position.quantity * position.current_price
                    total_value += position_value
                    unrealized_pnl += position.unrealized_pnl
                
                realized_pnl = sum(trade['realized_pnl'] for trade in self.trade_history)
                total_pnl = unrealized_pnl + realized_pnl
                
                # Calculate win rate
                winning_trades = len([t for t in self.trade_history if t['realized_pnl'] > 0])
                total_trades = len(self.trade_history)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
                
                # Calculate Sharpe ratio (placeholder)
                sharpe_ratio = 0.0  # TODO: Implement proper calculation
                
                # Calculate maximum drawdown
                max_drawdown = 0.0  # TODO: Implement proper calculation
                
                return PortfolioMetrics(
                    total_value=total_value,
                    cash_balance=self.cash_balance,
                    margin_used=0.0,  # TODO: Implement margin tracking
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=realized_pnl,
                    total_pnl=total_pnl,
                    win_rate=win_rate,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    last_updated=datetime.now()
                )
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio metrics: {e}")
            return None

    async def get_trade_history(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get trade history.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of trades
        """
        if symbol:
            return [trade for trade in self.trade_history if trade['symbol'] == symbol]
        return self.trade_history

    async def calculate_position_size(self, symbol: str, price: float, risk_per_trade: float = 0.02) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            symbol: Trading symbol
            price: Current price
            risk_per_trade: Maximum risk per trade as fraction of portfolio
            
        Returns:
            Recommended position size
        """
        try:
            metrics = await self.get_portfolio_metrics()
            if not metrics:
                return 0.0
            
            # Calculate maximum risk amount
            max_risk = metrics.total_value * risk_per_trade
            
            # TODO: Implement proper position sizing logic
            # This is a simple example that uses 1% of portfolio per trade
            position_size = (metrics.total_value * 0.01) / price
            
            return position_size
        except Exception as e:
            self.logger.error(f"Failed to calculate position size: {e}")
            return 0.0

    def get_position_count(self) -> int:
        """
        Get number of positions.
        
        Returns:
            Number of positions
        """
        return len(self.positions)

    def get_trade_count(self) -> int:
        """
        Get number of trades.
        
        Returns:
            Number of trades
        """
        return len(self.trade_history) 