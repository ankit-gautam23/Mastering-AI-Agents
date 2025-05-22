import pytest
import asyncio
from datetime import datetime
from src.utils.state_manager import StateManager, State

@pytest.fixture
async def state_manager():
    manager = StateManager()
    connected = await manager.connect()
    assert connected
    yield manager
    await manager.disconnect()

@pytest.mark.asyncio
async def test_set_get_state(state_manager):
    """Test setting and getting state."""
    # Set state
    test_value = {"data": "test", "timestamp": datetime.now().isoformat()}
    set_success = await state_manager.set_state("test_key", test_value)
    assert set_success
    
    # Get state
    state = await state_manager.get_state("test_key")
    assert state is not None
    assert state.key == "test_key"
    assert state.value == test_value
    assert state.version == 1

@pytest.mark.asyncio
async def test_delete_state(state_manager):
    """Test deleting state."""
    # Set state
    await state_manager.set_state("test_key", "test_value")
    
    # Delete state
    delete_success = await state_manager.delete_state("test_key")
    assert delete_success
    
    # Verify deletion
    state = await state_manager.get_state("test_key")
    assert state is None

@pytest.mark.asyncio
async def test_list_states(state_manager):
    """Test listing states."""
    # Set multiple states
    states = {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3"
    }
    
    for key, value in states.items():
        await state_manager.set_state(key, value)
    
    # List all states
    all_states = await state_manager.list_states()
    assert len(all_states) >= len(states)
    
    # List states with pattern
    pattern_states = await state_manager.list_states("key*")
    assert len(pattern_states) == len(states)

@pytest.mark.asyncio
async def test_state_versioning(state_manager):
    """Test state versioning."""
    # Set initial state
    await state_manager.set_state("test_key", "value1")
    state1 = await state_manager.get_state("test_key")
    assert state1.version == 1
    
    # Update state
    await state_manager.set_state("test_key", "value2")
    state2 = await state_manager.get_state("test_key")
    assert state2.version == 2
    assert state2.value == "value2"

@pytest.mark.asyncio
async def test_compare_and_set(state_manager):
    """Test compare and set operation."""
    # Set initial state
    await state_manager.set_state("test_key", "value1")
    
    # Get current version
    current_version = await state_manager.get_state_version("test_key")
    assert current_version == 1
    
    # Compare and set with correct version
    success = await state_manager.compare_and_set("test_key", "value2", current_version)
    assert success
    
    # Compare and set with incorrect version
    success = await state_manager.compare_and_set("test_key", "value3", current_version)
    assert not success

@pytest.mark.asyncio
async def test_state_watching(state_manager):
    """Test state watching."""
    received_states = []
    
    async def state_changed(state):
        received_states.append(state)
    
    # Start watching
    watch_success = await state_manager.watch_state("test_key", state_changed)
    assert watch_success
    
    # Set state multiple times
    for i in range(3):
        await state_manager.set_state("test_key", f"value{i}")
        await asyncio.sleep(0.1)
    
    # Wait for notifications
    await asyncio.sleep(1)
    assert len(received_states) >= 3

@pytest.mark.asyncio
async def test_state_metadata(state_manager):
    """Test state metadata."""
    metadata = {
        "source": "test",
        "timestamp": datetime.now().isoformat()
    }
    
    # Set state with metadata
    await state_manager.set_state("test_key", "test_value", metadata)
    
    # Get state and verify metadata
    state = await state_manager.get_state("test_key")
    assert state.metadata == metadata

@pytest.mark.asyncio
async def test_concurrent_updates(state_manager):
    """Test concurrent state updates."""
    async def update_state():
        for i in range(10):
            await state_manager.set_state("test_key", f"value{i}")
            await asyncio.sleep(0.01)
    
    # Run multiple concurrent updates
    tasks = [update_state() for _ in range(3)]
    await asyncio.gather(*tasks)
    
    # Verify final state
    state = await state_manager.get_state("test_key")
    assert state is not None
    assert state.version > 1 