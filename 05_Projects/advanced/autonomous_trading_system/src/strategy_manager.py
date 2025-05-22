from typing import Dict, List, Optional, Any, Callable
import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Signal:
    symbol: str
    timestamp: datetime
    type: str  # "BUY" or "SELL"
    price: float
    quantity: float
    confidence: float
    metadata: Dict[str, Any] = None

class Strategy(ABC):
    def __init__(self, name: str, symbols: List[str]):
        self.name = name
        self.symbols = symbols
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def analyze(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Analyze market data and generate trading signal.
        
        Args:
            data: Market data to analyze
            
        Returns:
            Trading signal if generated, None otherwise
        """
        pass

    @abstractmethod
    async def update_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Update strategy parameters.
        
        Args:
            parameters: New parameter values
            
        Returns:
            True if successful, False otherwise
        """
        pass

class MovingAverageCrossover(Strategy):
    def __init__(self, symbols: List[str], short_window: int = 20, long_window: int = 50):
        super().__init__("Moving Average Crossover", symbols)
        self.short_window = short_window
        self.long_window = long_window

    async def analyze(self, data: pd.DataFrame) -> Optional[Signal]:
        try:
            if len(data) < self.long_window:
                return None

            # Calculate moving averages
            short_ma = data['close'].rolling(window=self.short_window).mean()
            long_ma = data['close'].rolling(window=self.long_window).mean()

            # Generate signals
            if short_ma.iloc[-2] <= long_ma.iloc[-2] and short_ma.iloc[-1] > long_ma.iloc[-1]:
                return Signal(
                    symbol=data.index.name,
                    timestamp=datetime.now(),
                    type="BUY",
                    price=data['close'].iloc[-1],
                    quantity=1.0,  # TODO: Implement position sizing
                    confidence=0.7,
                    metadata={'short_ma': short_ma.iloc[-1], 'long_ma': long_ma.iloc[-1]}
                )
            elif short_ma.iloc[-2] >= long_ma.iloc[-2] and short_ma.iloc[-1] < long_ma.iloc[-1]:
                return Signal(
                    symbol=data.index.name,
                    timestamp=datetime.now(),
                    type="SELL",
                    price=data['close'].iloc[-1],
                    quantity=1.0,  # TODO: Implement position sizing
                    confidence=0.7,
                    metadata={'short_ma': short_ma.iloc[-1], 'long_ma': long_ma.iloc[-1]}
                )
            return None
        except Exception as e:
            self.logger.error(f"Error analyzing data: {e}")
            return None

    async def update_parameters(self, parameters: Dict[str, Any]) -> bool:
        try:
            if 'short_window' in parameters:
                self.short_window = parameters['short_window']
            if 'long_window' in parameters:
                self.long_window = parameters['long_window']
            return True
        except Exception as e:
            self.logger.error(f"Error updating parameters: {e}")
            return False

class RSIStrategy(Strategy):
    def __init__(self, symbols: List[str], period: int = 14, overbought: float = 70, oversold: float = 30):
        super().__init__("RSI Strategy", symbols)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    async def analyze(self, data: pd.DataFrame) -> Optional[Signal]:
        try:
            if len(data) < self.period + 1:
                return None

            # Calculate RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Generate signals
            if rsi.iloc[-2] > self.oversold and rsi.iloc[-1] <= self.oversold:
                return Signal(
                    symbol=data.index.name,
                    timestamp=datetime.now(),
                    type="BUY",
                    price=data['close'].iloc[-1],
                    quantity=1.0,  # TODO: Implement position sizing
                    confidence=0.6,
                    metadata={'rsi': rsi.iloc[-1]}
                )
            elif rsi.iloc[-2] < self.overbought and rsi.iloc[-1] >= self.overbought:
                return Signal(
                    symbol=data.index.name,
                    timestamp=datetime.now(),
                    type="SELL",
                    price=data['close'].iloc[-1],
                    quantity=1.0,  # TODO: Implement position sizing
                    confidence=0.6,
                    metadata={'rsi': rsi.iloc[-1]}
                )
            return None
        except Exception as e:
            self.logger.error(f"Error analyzing data: {e}")
            return None

    async def update_parameters(self, parameters: Dict[str, Any]) -> bool:
        try:
            if 'period' in parameters:
                self.period = parameters['period']
            if 'overbought' in parameters:
                self.overbought = parameters['overbought']
            if 'oversold' in parameters:
                self.oversold = parameters['oversold']
            return True
        except Exception as e:
            self.logger.error(f"Error updating parameters: {e}")
            return False

class StrategyManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.strategies: Dict[str, Strategy] = {}
        self.signal_handlers: List[Callable] = []
        self._lock = asyncio.Lock()

    async def add_strategy(self, strategy: Strategy) -> bool:
        """
        Add trading strategy.
        
        Args:
            strategy: Strategy to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                if strategy.name in self.strategies:
                    self.logger.warning(f"Strategy {strategy.name} already exists")
                    return False
                    
                self.strategies[strategy.name] = strategy
                return True
        except Exception as e:
            self.logger.error(f"Failed to add strategy: {e}")
            return False

    async def remove_strategy(self, strategy_name: str) -> bool:
        """
        Remove trading strategy.
        
        Args:
            strategy_name: Name of strategy to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                if strategy_name not in self.strategies:
                    self.logger.warning(f"Strategy {strategy_name} not found")
                    return False
                    
                del self.strategies[strategy_name]
                return True
        except Exception as e:
            self.logger.error(f"Failed to remove strategy: {e}")
            return False

    async def get_strategy(self, strategy_name: str) -> Optional[Strategy]:
        """
        Get strategy by name.
        
        Args:
            strategy_name: Name of strategy
            
        Returns:
            Strategy if found, None otherwise
        """
        return self.strategies.get(strategy_name)

    async def list_strategies(self) -> List[Strategy]:
        """
        List all strategies.
        
        Returns:
            List of all strategies
        """
        return list(self.strategies.values())

    async def register_signal_handler(self, handler: Callable) -> bool:
        """
        Register signal handler.
        
        Args:
            handler: Signal handler function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                self.signal_handlers.append(handler)
                return True
        except Exception as e:
            self.logger.error(f"Failed to register signal handler: {e}")
            return False

    async def unregister_signal_handler(self, handler: Callable) -> bool:
        """
        Unregister signal handler.
        
        Args:
            handler: Signal handler function to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                if handler in self.signal_handlers:
                    self.signal_handlers.remove(handler)
                return True
        except Exception as e:
            self.logger.error(f"Failed to unregister signal handler: {e}")
            return False

    async def process_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Process market data through strategies.
        
        Args:
            symbol: Trading symbol
            data: Market data to process
        """
        try:
            data.index.name = symbol
            for strategy in self.strategies.values():
                if symbol in strategy.symbols:
                    signal = await strategy.analyze(data)
                    if signal:
                        for handler in self.signal_handlers:
                            try:
                                await handler(signal)
                            except Exception as e:
                                self.logger.error(f"Error in signal handler: {e}")
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")

    async def update_strategy_parameters(self, strategy_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Update strategy parameters.
        
        Args:
            strategy_name: Name of strategy
            parameters: New parameter values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            strategy = await self.get_strategy(strategy_name)
            if strategy:
                return await strategy.update_parameters(parameters)
            return False
        except Exception as e:
            self.logger.error(f"Failed to update strategy parameters: {e}")
            return False 