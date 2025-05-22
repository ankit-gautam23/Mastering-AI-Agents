from typing import Dict, List, Optional, Any, Callable
import asyncio
import logging
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass

@dataclass
class Task:
    task_id: str
    name: str
    handler: Callable
    schedule: Optional[str] = None  # Cron-like schedule
    interval: Optional[timedelta] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_retries: int = 3
    retry_delay: timedelta = timedelta(seconds=5)
    metadata: Dict[str, Any] = None

class TaskScheduler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tasks: Dict[str, Task] = {}
        self.running = False
        self._task_handles: Dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        """
        Start the task scheduler.
        """
        self.running = True
        for task in self.tasks.values():
            await self._schedule_task(task)

    async def stop(self) -> None:
        """
        Stop the task scheduler.
        """
        self.running = False
        for handle in self._task_handles.values():
            handle.cancel()
        self._task_handles.clear()

    async def add_task(self, task: Task) -> bool:
        """
        Add a new task to the scheduler.
        
        Args:
            task: Task to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if task.task_id in self.tasks:
                self.logger.warning(f"Task {task.task_id} already exists")
                return False
                
            self.tasks[task.task_id] = task
            if self.running:
                await self._schedule_task(task)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add task: {e}")
            return False

    async def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from the scheduler.
        
        Args:
            task_id: ID of task to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if task_id not in self.tasks:
                self.logger.warning(f"Task {task_id} not found")
                return False
                
            if task_id in self._task_handles:
                self._task_handles[task_id].cancel()
                del self._task_handles[task_id]
                
            del self.tasks[task_id]
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove task: {e}")
            return False

    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task if found, None otherwise
        """
        return self.tasks.get(task_id)

    async def list_tasks(self) -> List[Task]:
        """
        List all tasks.
        
        Returns:
            List of all tasks
        """
        return list(self.tasks.values())

    async def _schedule_task(self, task: Task) -> None:
        """
        Schedule a task for execution.
        
        Args:
            task: Task to schedule
        """
        try:
            if task.schedule:
                # TODO: Implement cron-like scheduling
                pass
            elif task.interval:
                self._task_handles[task.task_id] = asyncio.create_task(
                    self._run_periodic_task(task)
                )
            elif task.start_time:
                delay = (task.start_time - datetime.now()).total_seconds()
                if delay > 0:
                    self._task_handles[task.task_id] = asyncio.create_task(
                        self._run_delayed_task(task, delay)
                    )
        except Exception as e:
            self.logger.error(f"Failed to schedule task: {e}")

    async def _run_periodic_task(self, task: Task) -> None:
        """
        Run a periodic task.
        
        Args:
            task: Task to run
        """
        while self.running:
            try:
                await task.handler()
                await asyncio.sleep(task.interval.total_seconds())
            except Exception as e:
                self.logger.error(f"Error running periodic task: {e}")
                await asyncio.sleep(task.retry_delay.total_seconds())

    async def _run_delayed_task(self, task: Task, delay: float) -> None:
        """
        Run a delayed task.
        
        Args:
            task: Task to run
            delay: Delay in seconds
        """
        try:
            await asyncio.sleep(delay)
            if self.running:
                await task.handler()
        except Exception as e:
            self.logger.error(f"Error running delayed task: {e}")

    def is_running(self) -> bool:
        """
        Check if scheduler is running.
        
        Returns:
            True if running, False otherwise
        """
        return self.running

    def get_task_count(self) -> int:
        """
        Get number of scheduled tasks.
        
        Returns:
            Number of tasks
        """
        return len(self.tasks) 