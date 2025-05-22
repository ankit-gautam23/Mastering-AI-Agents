from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .portfolio_manager import PortfolioManager

@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_holding_period: timedelta
    total_trades: int
    winning_trades: int
    losing_trades: int
    last_updated: datetime
    metadata: Dict[str, Any] = None

class PerformanceAnalyzer:
    def __init__(self, portfolio_manager: PortfolioManager):
        self.logger = logging.getLogger(__name__)
        self.portfolio_manager = portfolio_manager
        self._lock = asyncio.Lock()

    async def calculate_performance_metrics(self, start_date: Optional[datetime] = None) -> Optional[PerformanceMetrics]:
        """
        Calculate performance metrics for the portfolio.
        
        Args:
            start_date: Optional start date for analysis
            
        Returns:
            Performance metrics if successful, None otherwise
        """
        try:
            # Get trade history
            trades = await self.portfolio_manager.get_trade_history()
            if not trades:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            if start_date:
                df = df[df['timestamp'] >= start_date]
            
            if len(df) == 0:
                return None
            
            # Calculate returns
            df['return'] = df['realized_pnl']
            total_return = df['return'].sum()
            
            # Calculate annualized return
            days = (df['timestamp'].max() - df['timestamp'].min()).days
            annualized_return = (1 + total_return) ** (365/days) - 1 if days > 0 else 0
            
            # Calculate Sharpe ratio
            daily_returns = df.groupby(df['timestamp'].dt.date)['return'].sum()
            risk_free_rate = 0.02  # Placeholder
            excess_returns = daily_returns - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if len(daily_returns) > 1 else 0
            
            # Calculate Sortino ratio
            downside_returns = daily_returns[daily_returns < 0]
            sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # Calculate trade statistics
            winning_trades = df[df['realized_pnl'] > 0]
            losing_trades = df[df['realized_pnl'] < 0]
            
            win_rate = len(winning_trades) / len(df) if len(df) > 0 else 0
            profit_factor = abs(winning_trades['realized_pnl'].sum() / losing_trades['realized_pnl'].sum()) if len(losing_trades) > 0 else float('inf')
            
            average_win = winning_trades['realized_pnl'].mean() if len(winning_trades) > 0 else 0
            average_loss = losing_trades['realized_pnl'].mean() if len(losing_trades) > 0 else 0
            largest_win = winning_trades['realized_pnl'].max() if len(winning_trades) > 0 else 0
            largest_loss = losing_trades['realized_pnl'].min() if len(losing_trades) > 0 else 0
            
            # Calculate average holding period
            holding_periods = []
            for symbol in df['symbol'].unique():
                symbol_trades = df[df['symbol'] == symbol].sort_values('timestamp')
                if len(symbol_trades) >= 2:
                    for i in range(0, len(symbol_trades)-1, 2):
                        if i+1 < len(symbol_trades):
                            holding_periods.append(symbol_trades.iloc[i+1]['timestamp'] - symbol_trades.iloc[i]['timestamp'])
            
            average_holding_period = sum(holding_periods, timedelta()) / len(holding_periods) if holding_periods else timedelta()
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                average_holding_period=average_holding_period,
                total_trades=len(df),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                last_updated=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {e}")
            return None

    async def generate_performance_report(self, start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            start_date: Optional start date for analysis
            
        Returns:
            Dictionary containing performance report
        """
        try:
            metrics = await self.calculate_performance_metrics(start_date)
            if not metrics:
                return None
            
            # Get portfolio metrics
            portfolio_metrics = await self.portfolio_manager.get_portfolio_metrics()
            
            # Generate report
            report = {
                'summary': {
                    'total_return': f"{metrics.total_return:.2%}",
                    'annualized_return': f"{metrics.annualized_return:.2%}",
                    'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                    'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
                    'max_drawdown': f"{metrics.max_drawdown:.2%}",
                    'win_rate': f"{metrics.win_rate:.2%}",
                    'profit_factor': f"{metrics.profit_factor:.2f}"
                },
                'trade_statistics': {
                    'total_trades': metrics.total_trades,
                    'winning_trades': metrics.winning_trades,
                    'losing_trades': metrics.losing_trades,
                    'average_win': f"${metrics.average_win:.2f}",
                    'average_loss': f"${metrics.average_loss:.2f}",
                    'largest_win': f"${metrics.largest_win:.2f}",
                    'largest_loss': f"${metrics.largest_loss:.2f}",
                    'average_holding_period': str(metrics.average_holding_period)
                },
                'portfolio_statistics': {
                    'total_value': f"${portfolio_metrics.total_value:.2f}",
                    'cash_balance': f"${portfolio_metrics.cash_balance:.2f}",
                    'unrealized_pnl': f"${portfolio_metrics.unrealized_pnl:.2f}",
                    'realized_pnl': f"${portfolio_metrics.realized_pnl:.2f}"
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return report
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return None

    async def plot_performance_charts(self, start_date: Optional[datetime] = None) -> Dict[str, plt.Figure]:
        """
        Generate performance visualization charts.
        
        Args:
            start_date: Optional start date for analysis
            
        Returns:
            Dictionary of matplotlib figures
        """
        try:
            # Get trade history
            trades = await self.portfolio_manager.get_trade_history()
            if not trades:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            if start_date:
                df = df[df['timestamp'] >= start_date]
            
            if len(df) == 0:
                return {}
            
            # Calculate daily returns
            daily_returns = df.groupby(df['timestamp'].dt.date)['return'].sum()
            cumulative_returns = (1 + daily_returns).cumprod()
            
            # Create figures
            figures = {}
            
            # Equity curve
            fig, ax = plt.subplots(figsize=(12, 6))
            cumulative_returns.plot(ax=ax)
            ax.set_title('Equity Curve')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            figures['equity_curve'] = fig
            
            # Drawdown chart
            fig, ax = plt.subplots(figsize=(12, 6))
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            drawdowns.plot(ax=ax)
            ax.set_title('Drawdown')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown')
            figures['drawdown'] = fig
            
            # Monthly returns heatmap
            fig, ax = plt.subplots(figsize=(12, 6))
            monthly_returns = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month]).sum()
            monthly_returns = monthly_returns.unstack()
            sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax)
            ax.set_title('Monthly Returns')
            figures['monthly_returns'] = fig
            
            return figures
        except Exception as e:
            self.logger.error(f"Failed to generate performance charts: {e}")
            return {} 