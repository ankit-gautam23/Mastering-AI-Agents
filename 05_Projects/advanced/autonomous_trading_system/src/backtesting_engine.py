from typing import Dict, List, Optional, Any, Callable
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from .portfolio_manager import PortfolioManager
from .risk_manager import RiskManager
from .performance_analyzer import PerformanceAnalyzer

@dataclass
class BacktestResult:
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    profit_factor: float
    metadata: Dict[str, Any] = None

class BacktestingEngine:
    def __init__(self, portfolio_manager: PortfolioManager, risk_manager: RiskManager):
        self.logger = logging.getLogger(__name__)
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.performance_analyzer = PerformanceAnalyzer(portfolio_manager)
        self._lock = asyncio.Lock()
        self._historical_data: Dict[str, pd.DataFrame] = {}
        self._current_date: Optional[datetime] = None
        self._strategy: Optional[Callable] = None
        self._strategy_params: Dict[str, Any] = {}

    async def load_historical_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Load historical price data for backtesting.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with historical data (must include 'timestamp', 'open', 'high', 'low', 'close', 'volume')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns in historical data for {symbol}")
                return False
            
            # Ensure timestamp is datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
            data.sort_index(inplace=True)
            
            self._historical_data[symbol] = data
            return True
        except Exception as e:
            self.logger.error(f"Failed to load historical data for {symbol}: {e}")
            return False

    async def set_strategy(self, strategy: Callable, params: Dict[str, Any] = None) -> bool:
        """
        Set the trading strategy to backtest.
        
        Args:
            strategy: Strategy function that takes (symbol, data, params) and returns trade signals
            params: Optional strategy parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._strategy = strategy
            self._strategy_params = params or {}
            return True
        except Exception as e:
            self.logger.error(f"Failed to set strategy: {e}")
            return False

    async def run_backtest(self, start_date: datetime, end_date: datetime, initial_capital: float) -> Optional[BacktestResult]:
        """
        Run backtest for the specified period.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Initial capital for backtest
            
        Returns:
            Backtest result if successful, None otherwise
        """
        try:
            if not self._strategy:
                self.logger.error("No strategy set for backtesting")
                return None
            
            # Initialize portfolio
            self.portfolio_manager.cash_balance = initial_capital
            self.portfolio_manager.positions.clear()
            self.portfolio_manager.trade_history.clear()
            
            # Get date range
            dates = pd.date_range(start_date, end_date, freq='D')
            
            # Run simulation
            for date in dates:
                self._current_date = date
                
                # Get data up to current date
                for symbol, data in self._historical_data.items():
                    current_data = data[data.index <= date]
                    if len(current_data) == 0:
                        continue
                    
                    # Get strategy signals
                    signals = await self._strategy(symbol, current_data, self._strategy_params)
                    if not signals:
                        continue
                    
                    # Execute trades based on signals
                    for signal in signals:
                        if signal['action'] == 'buy':
                            # Validate trade against risk limits
                            if await self.risk_manager.validate_trade(
                                symbol,
                                signal['quantity'],
                                current_data.iloc[-1]['close'],
                                True
                            ):
                                await self.portfolio_manager.update_position(
                                    symbol,
                                    signal['quantity'],
                                    current_data.iloc[-1]['close']
                                )
                        elif signal['action'] == 'sell':
                            await self.portfolio_manager.update_position(
                                symbol,
                                -signal['quantity'],
                                current_data.iloc[-1]['close']
                            )
                
                # Update portfolio with current prices
                for symbol in self.portfolio_manager.positions:
                    if symbol in self._historical_data:
                        current_price = self._historical_data[symbol].loc[date, 'close']
                        await self.portfolio_manager.update_prices({symbol: current_price})
            
            # Calculate performance metrics
            metrics = await self.performance_analyzer.calculate_performance_metrics()
            if not metrics:
                return None
            
            # Get final portfolio value
            portfolio_metrics = await self.portfolio_manager.get_portfolio_metrics()
            
            return BacktestResult(
                strategy_name=self._strategy.__name__,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=portfolio_metrics.total_value,
                total_return=metrics.total_return,
                annualized_return=metrics.annualized_return,
                sharpe_ratio=metrics.sharpe_ratio,
                max_drawdown=metrics.max_drawdown,
                win_rate=metrics.win_rate,
                total_trades=metrics.total_trades,
                winning_trades=metrics.winning_trades,
                losing_trades=metrics.losing_trades,
                average_win=metrics.average_win,
                average_loss=metrics.average_loss,
                profit_factor=metrics.profit_factor,
                metadata={
                    'strategy_params': self._strategy_params,
                    'symbols_traded': list(self._historical_data.keys())
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to run backtest: {e}")
            return None

    async def get_historical_data(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol within the specified date range.
        
        Args:
            symbol: Trading symbol
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            DataFrame with historical data if available, None otherwise
        """
        try:
            if symbol not in self._historical_data:
                return None
            
            data = self._historical_data[symbol]
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            return data
        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None

    async def get_available_symbols(self) -> List[str]:
        """
        Get list of symbols with loaded historical data.
        
        Returns:
            List of available symbols
        """
        return list(self._historical_data.keys())

    async def get_current_date(self) -> Optional[datetime]:
        """
        Get current simulation date.
        
        Returns:
            Current date if simulation is running, None otherwise
        """
        return self._current_date 