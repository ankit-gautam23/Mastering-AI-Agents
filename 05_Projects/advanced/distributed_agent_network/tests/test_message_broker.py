import pytest
import asyncio
import json
from datetime import datetime
from src.utils.message_broker import MessageBroker

@pytest.fixture
async def message_broker():
    broker = MessageBroker()
    connected = await broker.connect()
    assert connected
    yield broker
    await broker.disconnect()

@pytest.mark.asyncio
async def test_publish_subscribe(message_broker):
    """Test publishing and subscribing to messages."""
    received_messages = []
    
    async def message_handler(message):
        received_messages.append(message)
    
    # Subscribe to test topic
    subscribed = await message_broker.subscribe("test_topic", message_handler)
    assert subscribed
    
    # Publish test message
    test_message = {"data": "test", "timestamp": datetime.now().isoformat()}
    published = await message_broker.publish("test_topic", test_message)
    assert published
    
    # Wait for message processing
    await asyncio.sleep(1)
    assert len(received_messages) == 1
    assert received_messages[0]["data"] == test_message["data"]

@pytest.mark.asyncio
async def test_unsubscribe(message_broker):
    """Test unsubscribing from topic."""
    received_messages = []
    
    async def message_handler(message):
        received_messages.append(message)
    
    # Subscribe and then unsubscribe
    subscribed = await message_broker.subscribe("test_topic", message_handler)
    assert subscribed
    
    unsubscribed = await message_broker.unsubscribe("test_topic")
    assert unsubscribed
    
    # Publish message after unsubscribe
    test_message = {"data": "test"}
    published = await message_broker.publish("test_topic", test_message)
    assert published
    
    # Wait for message processing
    await asyncio.sleep(1)
    assert len(received_messages) == 0

@pytest.mark.asyncio
async def test_rpc_call(message_broker):
    """Test RPC calls."""
    async def test_handler(params):
        return {"result": params["input"] * 2}
    
    # Register RPC handler
    registered = await message_broker.register_rpc_handler("test_method", test_handler)
    assert registered
    
    # Make RPC call
    result = await message_broker.call_rpc("test_method", {"input": 5})
    assert result["result"] == 10
    
    # Unregister handler
    unregistered = await message_broker.unregister_rpc_handler("test_method")
    assert unregistered

@pytest.mark.asyncio
async def test_connection_management(message_broker):
    """Test connection management."""
    # Check connection
    connection = message_broker.get_connection()
    assert connection is not None
    
    # Check channel
    channel = message_broker.get_channel()
    assert channel is not None
    
    # Disconnect and reconnect
    await message_broker.disconnect()
    connected = await message_broker.connect()
    assert connected

@pytest.mark.asyncio
async def test_error_handling(message_broker):
    """Test error handling."""
    # Test invalid topic
    published = await message_broker.publish("", {"data": "test"})
    assert not published
    
    # Test invalid message
    published = await message_broker.publish("test_topic", None)
    assert not published
    
    # Test invalid RPC call
    result = await message_broker.call_rpc("nonexistent_method", {})
    assert result is None

@pytest.mark.asyncio
async def test_multiple_subscribers(message_broker):
    """Test multiple subscribers to same topic."""
    received_messages_1 = []
    received_messages_2 = []
    
    async def handler1(message):
        received_messages_1.append(message)
    
    async def handler2(message):
        received_messages_2.append(message)
    
    # Subscribe two handlers
    await message_broker.subscribe("test_topic", handler1)
    await message_broker.subscribe("test_topic", handler2)
    
    # Publish message
    test_message = {"data": "test"}
    await message_broker.publish("test_topic", test_message)
    
    # Wait for message processing
    await asyncio.sleep(1)
    assert len(received_messages_1) == 1
    assert len(received_messages_2) == 1
    assert received_messages_1[0] == received_messages_2[0] 