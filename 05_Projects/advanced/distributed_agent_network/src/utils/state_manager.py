from typing import Dict, Any, Optional
import json
import logging
import asyncio
from datetime import datetime
import redis.asyncio as redis
from dataclasses import dataclass, asdict

@dataclass
class State:
    key: str
    value: Any
    version: int = 1
    last_modified: datetime = None
    metadata: Dict[str, Any] = None

class StateManager:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.logger = logging.getLogger(__name__)
        self.redis = redis.from_url(redis_url)
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        """
        Connect to state store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to state store: {e}")
            return False

    async def disconnect(self) -> None:
        """
        Disconnect from state store.
        """
        await self.redis.close()

    async def set_state(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> bool:
        """
        Set state value.
        
        Args:
            key: State key
            value: State value
            metadata: Optional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                # Get current version
                current = await self.get_state(key)
                version = (current.version + 1) if current else 1
                
                state = State(
                    key=key,
                    value=value,
                    version=version,
                    last_modified=datetime.now(),
                    metadata=metadata
                )
                
                await self.redis.set(
                    key,
                    json.dumps(asdict(state), default=str)
                )
                return True
        except Exception as e:
            self.logger.error(f"Failed to set state: {e}")
            return False

    async def get_state(self, key: str) -> Optional[State]:
        """
        Get state value.
        
        Args:
            key: State key
            
        Returns:
            State if found, None otherwise
        """
        try:
            data = await self.redis.get(key)
            if data:
                state_dict = json.loads(data)
                return State(**state_dict)
            return None
        except Exception as e:
            self.logger.error(f"Failed to get state: {e}")
            return None

    async def delete_state(self, key: str) -> bool:
        """
        Delete state value.
        
        Args:
            key: State key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete state: {e}")
            return False

    async def list_states(self, pattern: str = "*") -> Dict[str, State]:
        """
        List all states matching pattern.
        
        Args:
            pattern: Key pattern to match
            
        Returns:
            Dictionary of matching states
        """
        try:
            states = {}
            async for key in self.redis.scan_iter(match=pattern):
                state = await self.get_state(key)
                if state:
                    states[key] = state
            return states
        except Exception as e:
            self.logger.error(f"Failed to list states: {e}")
            return {}

    async def watch_state(self, key: str, callback) -> bool:
        """
        Watch state for changes.
        
        Args:
            key: State key to watch
            callback: Callback function to invoke on change
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(f"__keyspace@0__:{key}")
            
            async def watch():
                while True:
                    try:
                        message = await pubsub.get_message(ignore_subscribe_messages=True)
                        if message and message["type"] == "message":
                            state = await self.get_state(key)
                            await callback(state)
                    except Exception as e:
                        self.logger.error(f"Error watching state: {e}")
                        await asyncio.sleep(1)
            
            asyncio.create_task(watch())
            return True
        except Exception as e:
            self.logger.error(f"Failed to watch state: {e}")
            return False

    async def get_state_version(self, key: str) -> Optional[int]:
        """
        Get state version.
        
        Args:
            key: State key
            
        Returns:
            Version number if found, None otherwise
        """
        state = await self.get_state(key)
        return state.version if state else None

    async def compare_and_set(self, key: str, value: Any, expected_version: int) -> bool:
        """
        Compare and set state value.
        
        Args:
            key: State key
            value: New state value
            expected_version: Expected current version
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                current = await self.get_state(key)
                if not current or current.version != expected_version:
                    return False
                    
                return await self.set_state(key, value)
        except Exception as e:
            self.logger.error(f"Failed to compare and set state: {e}")
            return False 