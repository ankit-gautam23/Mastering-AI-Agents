import pytest
import asyncio
from datetime import datetime, timedelta
from src.utils.task_scheduler import TaskScheduler, Task

@pytest.fixture
async def task_scheduler():
    scheduler = TaskScheduler()
    await scheduler.start()
    yield scheduler
    await scheduler.stop()

@pytest.mark.asyncio
async def test_add_remove_task(task_scheduler):
    """Test adding and removing tasks."""
    # Create test task
    task = Task(
        task_id="test_task",
        name="Test Task",
        handler=lambda: None,
        interval=timedelta(seconds=1)
    )
    
    # Add task
    added = await task_scheduler.add_task(task)
    assert added
    assert task_scheduler.get_task_count() == 1
    
    # Remove task
    removed = await task_scheduler.remove_task(task.task_id)
    assert removed
    assert task_scheduler.get_task_count() == 0

@pytest.mark.asyncio
async def test_periodic_task(task_scheduler):
    """Test periodic task execution."""
    execution_count = 0
    
    async def periodic_handler():
        nonlocal execution_count
        execution_count += 1
    
    # Create periodic task
    task = Task(
        task_id="periodic_task",
        name="Periodic Task",
        handler=periodic_handler,
        interval=timedelta(seconds=1)
    )
    
    # Add task
    await task_scheduler.add_task(task)
    
    # Wait for multiple executions
    await asyncio.sleep(3.5)
    assert execution_count >= 3

@pytest.mark.asyncio
async def test_delayed_task(task_scheduler):
    """Test delayed task execution."""
    execution_count = 0
    
    async def delayed_handler():
        nonlocal execution_count
        execution_count += 1
    
    # Create delayed task
    task = Task(
        task_id="delayed_task",
        name="Delayed Task",
        handler=delayed_handler,
        start_time=datetime.now() + timedelta(seconds=2)
    )
    
    # Add task
    await task_scheduler.add_task(task)
    
    # Check initial state
    assert execution_count == 0
    
    # Wait for execution
    await asyncio.sleep(3)
    assert execution_count == 1

@pytest.mark.asyncio
async def test_task_retry(task_scheduler):
    """Test task retry on failure."""
    execution_count = 0
    
    async def failing_handler():
        nonlocal execution_count
        execution_count += 1
        raise Exception("Task failed")
    
    # Create failing task
    task = Task(
        task_id="failing_task",
        name="Failing Task",
        handler=failing_handler,
        interval=timedelta(seconds=1),
        max_retries=3,
        retry_delay=timedelta(seconds=1)
    )
    
    # Add task
    await task_scheduler.add_task(task)
    
    # Wait for retries
    await asyncio.sleep(5)
    assert execution_count == 3

@pytest.mark.asyncio
async def test_get_task(task_scheduler):
    """Test getting task by ID."""
    # Create test task
    task = Task(
        task_id="test_task",
        name="Test Task",
        handler=lambda: None,
        interval=timedelta(seconds=1)
    )
    
    # Add task
    await task_scheduler.add_task(task)
    
    # Get task
    retrieved_task = await task_scheduler.get_task(task.task_id)
    assert retrieved_task is not None
    assert retrieved_task.task_id == task.task_id
    assert retrieved_task.name == task.name

@pytest.mark.asyncio
async def test_list_tasks(task_scheduler):
    """Test listing all tasks."""
    # Create multiple tasks
    tasks = [
        Task(
            task_id=f"task_{i}",
            name=f"Task {i}",
            handler=lambda: None,
            interval=timedelta(seconds=1)
        )
        for i in range(3)
    ]
    
    # Add tasks
    for task in tasks:
        await task_scheduler.add_task(task)
    
    # List tasks
    task_list = await task_scheduler.list_tasks()
    assert len(task_list) == 3
    assert all(task.task_id in [t.task_id for t in task_list] for task in tasks)

@pytest.mark.asyncio
async def test_scheduler_lifecycle(task_scheduler):
    """Test scheduler start/stop lifecycle."""
    execution_count = 0
    
    async def handler():
        nonlocal execution_count
        execution_count += 1
    
    # Create task
    task = Task(
        task_id="lifecycle_task",
        name="Lifecycle Task",
        handler=handler,
        interval=timedelta(seconds=1)
    )
    
    # Add task
    await task_scheduler.add_task(task)
    
    # Wait for some executions
    await asyncio.sleep(2)
    initial_count = execution_count
    
    # Stop scheduler
    await task_scheduler.stop()
    assert not task_scheduler.is_running()
    
    # Wait and verify no more executions
    await asyncio.sleep(2)
    assert execution_count == initial_count
    
    # Restart scheduler
    await task_scheduler.start()
    assert task_scheduler.is_running()
    
    # Wait for more executions
    await asyncio.sleep(2)
    assert execution_count > initial_count 