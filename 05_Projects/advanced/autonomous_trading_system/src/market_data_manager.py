from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
import websockets
import json

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: Dict[str, Any] = None

class MarketDataManager:
    def __init__(self, websocket_url: str = "wss://stream.binance.com:9443/ws"):
        self.logger = logging.getLogger(__name__)
        self.websocket_url = websocket_url
        self.websocket = None
        self.subscribers: Dict[str, List[callable]] = {}
        self.data_buffer: Dict[str, List[MarketData]] = {}
        self.running = False
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        """
        Connect to market data stream.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            self.running = True
            asyncio.create_task(self._process_messages())
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to market data stream: {e}")
            return False

    async def disconnect(self) -> None:
        """
        Disconnect from market data stream.
        """
        self.running = False
        if self.websocket:
            await self.websocket.close()

    async def subscribe(self, symbol: str, callback: callable) -> bool:
        """
        Subscribe to market data for symbol.
        
        Args:
            symbol: Trading symbol
            callback: Callback function for data updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                if symbol not in self.subscribers:
                    self.subscribers[symbol] = []
                    self.data_buffer[symbol] = []
                    
                    # Subscribe to websocket stream
                    subscribe_msg = {
                        "method": "SUBSCRIBE",
                        "params": [f"{symbol.lower()}@kline_1m"],
                        "id": 1
                    }
                    await self.websocket.send(json.dumps(subscribe_msg))
                
                self.subscribers[symbol].append(callback)
                return True
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False

    async def unsubscribe(self, symbol: str, callback: callable) -> bool:
        """
        Unsubscribe from market data for symbol.
        
        Args:
            symbol: Trading symbol
            callback: Callback function to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                if symbol in self.subscribers:
                    self.subscribers[symbol].remove(callback)
                    if not self.subscribers[symbol]:
                        # Unsubscribe from websocket stream
                        unsubscribe_msg = {
                            "method": "UNSUBSCRIBE",
                            "params": [f"{symbol.lower()}@kline_1m"],
                            "id": 1
                        }
                        await self.websocket.send(json.dumps(unsubscribe_msg))
                        del self.subscribers[symbol]
                        del self.data_buffer[symbol]
                return True
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from {symbol}: {e}")
            return False

    async def get_historical_data(self, symbol: str, interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            limit: Number of data points
            
        Returns:
            DataFrame with historical data
        """
        try:
            # TODO: Implement API call to get historical data
            # This is a placeholder that returns random data
            dates = pd.date_range(end=datetime.now(), periods=limit, freq=interval)
            data = {
                'open': np.random.normal(100, 1, limit),
                'high': np.random.normal(101, 1, limit),
                'low': np.random.normal(99, 1, limit),
                'close': np.random.normal(100, 1, limit),
                'volume': np.random.normal(1000, 100, limit)
            }
            df = pd.DataFrame(data, index=dates)
            return df
        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {e}")
            return pd.DataFrame()

    async def _process_messages(self) -> None:
        """
        Process incoming websocket messages.
        """
        while self.running:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if 'k' in data:
                    kline = data['k']
                    symbol = kline['s']
                    
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(kline['t'] / 1000),
                        open=float(kline['o']),
                        high=float(kline['h']),
                        low=float(kline['l']),
                        close=float(kline['c']),
                        volume=float(kline['v']),
                        metadata={'is_closed': kline['x']}
                    )
                    
                    async with self._lock:
                        if symbol in self.data_buffer:
                            self.data_buffer[symbol].append(market_data)
                            # Keep only last 1000 data points
                            if len(self.data_buffer[symbol]) > 1000:
                                self.data_buffer[symbol] = self.data_buffer[symbol][-1000:]
                            
                            # Notify subscribers
                            for callback in self.subscribers[symbol]:
                                try:
                                    await callback(market_data)
                                except Exception as e:
                                    self.logger.error(f"Error in subscriber callback: {e}")
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                await asyncio.sleep(1)

    def get_data_buffer(self, symbol: str) -> List[MarketData]:
        """
        Get buffered market data for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of market data points
        """
        return self.data_buffer.get(symbol, [])

    def is_connected(self) -> bool:
        """
        Check if connected to market data stream.
        
        Returns:
            True if connected, False otherwise
        """
        return self.running and self.websocket is not None 