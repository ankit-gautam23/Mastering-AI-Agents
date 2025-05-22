import pytest
from datetime import datetime, timedelta
from src.task import Task

@pytest.fixture
def sample_task():
    return Task(
        title="Test Task",
        description="Test Description",
        creator_id="test_user",
        due_date=datetime.now() + timedelta(days=1),
        priority="high"
    )

def test_task_creation(sample_task):
    # TODO: Implement test for task creation
    # 1. Test all fields are set correctly
    # 2. Test default values
    pass

def test_task_update(sample_task):
    # TODO: Implement test for task update
    # 1. Test successful update
    # 2. Test with invalid updates
    # 3. Test timestamp update
    pass

def test_add_comment(sample_task):
    # TODO: Implement test for add_comment
    # 1. Test successful comment addition
    # 2. Test comment format
    # 3. Test timestamp update
    pass

def test_add_tag(sample_task):
    # TODO: Implement test for add_tag
    # 1. Test successful tag addition
    # 2. Test duplicate tag handling
    # 3. Test invalid tag handling
    pass

def test_remove_tag(sample_task):
    # TODO: Implement test for remove_tag
    # 1. Test successful tag removal
    # 2. Test non-existent tag handling
    pass

def test_is_overdue(sample_task):
    # TODO: Implement test for is_overdue
    # 1. Test overdue task
    # 2. Test non-overdue task
    # 3. Test task without due date
    pass 