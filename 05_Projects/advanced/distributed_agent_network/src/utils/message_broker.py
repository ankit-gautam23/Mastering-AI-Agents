from typing import Dict, Callable, Optional
import json
import logging
import asyncio
import aio_pika
from aio_pika import connect_robust, Message, DeliveryMode
from aio_pika.patterns import RPC

class MessageBroker:
    def __init__(self, url: str = "amqp://guest:guest@localhost/"):
        self.url = url
        self.logger = logging.getLogger(__name__)
        self.connection = None
        self.channel = None
        self.rpc = None
        self.handlers: Dict[str, Callable] = {}

    async def connect(self) -> bool:
        """
        Connect to message broker.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.connection = await connect_robust(self.url)
            self.channel = await self.connection.channel()
            self.rpc = await RPC.create(self.channel)
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to message broker: {e}")
            return False

    async def disconnect(self) -> None:
        """
        Disconnect from message broker.
        """
        if self.connection:
            await self.connection.close()

    async def publish(self, topic: str, message: Dict) -> bool:
        """
        Publish message to topic.
        
        Args:
            topic: Topic to publish to
            message: Message to publish
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.channel.default_exchange.publish(
                Message(
                    body=json.dumps(message).encode(),
                    delivery_mode=DeliveryMode.PERSISTENT
                ),
                routing_key=topic
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish message: {e}")
            return False

    async def subscribe(self, topic: str, handler: Callable) -> bool:
        """
        Subscribe to topic.
        
        Args:
            topic: Topic to subscribe to
            handler: Message handler function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            queue = await self.channel.declare_queue(topic)
            self.handlers[topic] = handler
            
            async def process_message(message):
                async with message.process():
                    try:
                        payload = json.loads(message.body.decode())
                        await handler(payload)
                    except Exception as e:
                        self.logger.error(f"Error processing message: {e}")
            
            await queue.consume(process_message)
            return True
        except Exception as e:
            self.logger.error(f"Failed to subscribe to topic: {e}")
            return False

    async def unsubscribe(self, topic: str) -> bool:
        """
        Unsubscribe from topic.
        
        Args:
            topic: Topic to unsubscribe from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if topic in self.handlers:
                del self.handlers[topic]
            await self.channel.queue_delete(topic)
            return True
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from topic: {e}")
            return False

    async def call_rpc(self, method: str, params: Dict = None) -> Optional[Dict]:
        """
        Make RPC call.
        
        Args:
            method: Method to call
            params: Method parameters
            
        Returns:
            RPC response if successful, None otherwise
        """
        try:
            response = await self.rpc.call(method, params or {})
            return response
        except Exception as e:
            self.logger.error(f"RPC call failed: {e}")
            return None

    async def register_rpc_handler(self, method: str, handler: Callable) -> bool:
        """
        Register RPC handler.
        
        Args:
            method: Method name
            handler: Handler function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.rpc.register(method, handler)
            return True
        except Exception as e:
            self.logger.error(f"Failed to register RPC handler: {e}")
            return False

    async def unregister_rpc_handler(self, method: str) -> bool:
        """
        Unregister RPC handler.
        
        Args:
            method: Method name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.rpc.unregister(method)
            return True
        except Exception as e:
            self.logger.error(f"Failed to unregister RPC handler: {e}")
            return False

    def get_connection(self) -> Optional[aio_pika.Connection]:
        """
        Get message broker connection.
        
        Returns:
            Connection object if connected, None otherwise
        """
        return self.connection

    def get_channel(self) -> Optional[aio_pika.Channel]:
        """
        Get message broker channel.
        
        Returns:
            Channel object if connected, None otherwise
        """
        return self.channel 